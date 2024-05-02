# based on https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat

from threading import Thread
from typing import Iterator, Optional

import json5
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def get_model_and_tokenizer(
    model_name: str,
    adapter_name: str,
    tokenizer_name: Optional[str] = None,
    quantize: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    if adapter_name is not None:
        model = PeftModel.from_pretrained(model, adapter_name, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name
    )

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
    text = text[text.find("{") :]
    text = text[: text.find("}") + 1]
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
            prompt = prompt[: prompt.rfind("PLACEHOLDER")] + "```json"
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
