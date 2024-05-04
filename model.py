# based on https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat

from functools import partial
from threading import Thread
from types import SimpleNamespace
from typing import Iterator, Optional

import json5
import torch
import yaml
from peft import PeftModel
import peft.tuners.lora.layer as lora_layer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from tqdm import tqdm
from trl import SFTTrainer

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))
REASONING = "".join(["<BLAH>"] * config.reasoning_tokens)


def get_model_and_tokenizer(
    model_name: str,
    adapter_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    quantize: bool = False,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_use_double_quant: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # add special <BLAH> token
    if config.reasoning_tokens > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<BLAH>"]})
        model.resize_token_embeddings(len(tokenizer))

    if adapter_name is not None:
        model = PeftModel.from_pretrained(model, adapter_name, device_map="auto")

    return model, tokenizer


def get_prompt(
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for user_message, assistant_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_input_token_length(
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
) -> int:
    prompt = get_prompt(tokenizer, message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ]
    return input_ids.shape[-1]


def run(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
) -> Iterator[str]:
    prompt = get_prompt(tokenizer, message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


def extract_answer(text):
    text = text[text.find("{") : text.rfind("}") + 1]
    try:
        # JSON5 is a little less picky than JSON
        answer = json5.loads(text)["answer"]
    except:
        answer = None
    return answer


class StopAfterToken(StoppingCriteria):
    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return input_ids[0][-1] == self.token_id


def get_answer(messages, pipeline, num_beams=None, force_answer=False):
    assistant_messages = [
        message
        for message in range(len(messages))
        if messages[message]["role"] == "assistant"
    ]

    for _, assistant_message in enumerate(assistant_messages):
        if force_answer and _ == len(assistant_messages) - 1:
            prompt = pipeline.tokenizer.apply_chat_template(
                messages[:assistant_message]
                + [{"role": "assistant", "content": "PLACEHOLDER"}],
                tokenize=False,
            )
            prompt = prompt[: prompt.rfind("PLACEHOLDER")] + f"{REASONING}\n```json"
            stopping_criteria = StoppingCriteriaList(
                [
                    StopAfterToken(
                        pipeline.tokenizer.vocab.get(
                            "}Ċ", pipeline.tokenizer.vocab["}"]
                        )
                    )
                ]
            )
        else:
            prompt = pipeline.tokenizer.apply_chat_template(
                messages[:assistant_message], tokenize=False, add_generation_prompt=True
            )
            stopping_criteria = None

        response = pipeline(
            prompt,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=1,
            max_new_tokens=512,
            temperature=None,
            top_p=None,
            stopping_criteria=stopping_criteria,
        )[0]["generated_text"]
        response = response[len(prompt) :].strip()
        messages[assistant_message] = {"role": "assistant", "content": response}

    return extract_answer(response), response


class SquadSFTTrainer(SFTTrainer):
    def __init__(self, answer_start_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens
        self.stopping_criteria = StoppingCriteriaList(
            [StopAfterToken(self.tokenizer.vocab.get("}Ċ", self.tokenizer.vocab["}"]))]
        )

    def evaluate(self, **kwargs):
        def cast_hook(dtype, module, input):
            input = input[0].to(dtype)
            return (input,)

        accuracy = 0
        for item in tqdm(self.eval_dataset, desc="Evaluating"):
            input_ids = torch.tensor(item["input_ids"]).to(self.model.device)
            attention_mask = torch.tensor(item["attention_mask"]).to(self.model.device)
            answer_start_tokens = self.answer_start_tokens.to(self.model.device)
            window = input_ids.unfold(0, answer_start_tokens.shape[0], 1)
            answer_start = (window == answer_start_tokens).all(dim=1).nonzero()[-1, 0]
            answers = extract_answer(
                self.tokenizer.decode(
                    item["input_ids"][answer_start:], skip_special_tokens=True
                )
            )

            # NFI why this is necessary here but not during training
            hook_handles = []
            for _, module in self.model.named_modules():
                if isinstance(module, lora_layer.Linear):
                    hook_handles.append(
                        module.register_forward_pre_hook(
                            partial(cast_hook, self.model.dtype)
                        )
                    )

            output = self.model.generate(
                input_ids=input_ids[
                    : answer_start + len(self.answer_start_tokens)
                ].unsqueeze(0),
                attention_mask=attention_mask[
                    : answer_start + len(self.answer_start_tokens)
                ].unsqueeze(0),
                do_sample=False,
                num_return_sequences=1,
                max_new_tokens=512,
                temperature=None,
                top_p=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for hook_handle in hook_handles:
                hook_handle.remove()

            model_answer = extract_answer(
                self.tokenizer.decode(
                    output[0, answer_start:], skip_special_tokens=True
                )
            )
            accuracy += 1 if model_answer is not None and model_answer in answers else 0

        accuracy /= len(self.eval_dataset)
        self.log({"eval_accuracy": accuracy})
        return {"eval_accuracy": accuracy}
