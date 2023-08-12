# based on https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat

import json
import random
from dataclasses import dataclass, field
from typing import Iterator, Optional

import gradio as gr
import torch
from datasets import load_dataset
from transformers import HfArgumentParser

from model import (
    DEFAULT_SYSTEM_PROMPT,
    get_model_and_tokenizer,
    get_input_token_length,
    run,
)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default=None,
    )
    quantize: Optional[bool] = field(default=False)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

DESCRIPTION = f"""
# LLaMA SQuAD

![LLaMA SQuAD](https://github.com/teticio/llama-squad/blob/main/llama_squad.png?raw=true)

#### Base model: {script_args.model_name}
#### Adapter checkpoints: {script_args.adapter_name}
#### Choose a random question, press submit and then see if the answer is one of the correct ones below. You can then ask it to "Please explain your reasoning".
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


LICENSE = """
<p/>

---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return "", message


def display_input(
    message: str, history: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    history.append((message, ""))
    return history


def delete_prev_fn(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ""
    return history, message or ""


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(
        model,
        tokenizer,
        message,
        history,
        system_prompt,
        max_new_tokens,
    )
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, "")]
    for response in generator:
        yield history + [(message, response)]


def check_input_token_length(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> None:
    input_token_length = get_input_token_length(
        tokenizer, message, chat_history, system_prompt
    )
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(
            f"The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
        )


def get_random_question_and_answers():
    item = random.choice(dataset)
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = ["?"]
    return f"""\
Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```
If the answer is not in the context, the answer should be "?".
Context: {item["context"]}
Question: {item["question"]}""", json.dumps(
        answers
    )


model, tokenizer = get_model_and_tokenizer(
    model_name=script_args.model_name,
    adapter_name=script_args.adapter_name,
    tokenizer_name=script_args.tokenizer_name,
    quantize=script_args.quantize,
)

dataset = load_dataset("squad_v2")["validation"]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        chatbot = gr.Chatbot(label="Chatbot")
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder="Type a message...",
                scale=10,
            )
            submit_button = gr.Button("Submit", variant="primary", scale=1, min_width=0)

    with gr.Row():
        retry_button = gr.Button("🔄  Retry", variant="secondary")
        undo_button = gr.Button("↩️ Undo", variant="secondary")
        clear_button = gr.Button("🗑️  Clear", variant="secondary")
        random_question_button = gr.Button("Random Question", variant="secondary")

    with gr.Row():
        answers = gr.Textbox(
            label="Answers",
            scale=10,
        )

    saved_input = gr.State()

    with gr.Accordion(label="Advanced options", open=False):
        system_prompt = gr.Textbox(
            label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=6
        )
        max_new_tokens = gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )

    gr.Markdown(LICENSE)

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = (
        submit_button.click(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        )
        .then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        )
        .then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        )
        .success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
            ],
            outputs=chatbot,
            api_name=False,
        )
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

    random_question_button.click(
        fn=get_random_question_and_answers,
        outputs=[textbox, answers],
        api_name=False,
        queue=False,
    )

demo.queue(max_size=20).launch()
