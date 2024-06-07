import json
from dataclasses import dataclass, field
from textwrap import dedent
from types import SimpleNamespace
from typing import Optional

import yaml
from datasets import DatasetDict, load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    prompt: Optional[str] = field(
        default="single_turn",
        metadata={"help": "single_turn, multi_turn"},
    )
    validation_ratio: Optional[float] = field(
        default=0.005,
        metadata={"help": "Validation ratio"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for the random number generator"},
    )


config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))
# Llama 3 has several <|reserved_special_token_...|> that could be used instead
REASONING = (
    "".join([f"<blah_{i}>" for i in range(config.num_reasoning_tokens)])
    if config.multiple_reasoning_tokens
    else "".join(["<blah>"] * config.num_reasoning_tokens)
)
NO_RESPONSE = "?"


def is_exact_match(model_answer, answers):
    return model_answer is not None and model_answer in answers


def get_single_turn_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = [NO_RESPONSE]
    if not all_answers:
        answers = answers[0]
    answers = json.dumps(answers)

    return {
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": dedent(
                    f"""\
                    Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
                    ```json
                    {{
                      "answer": ...
                    }}
                    ```
                    If the answer is not in the context, the answer should be "{NO_RESPONSE}".
                    Context: {context}
                    Question: {question}"""
                ),
            },
            {
                "role": "assistant",
                "content": dedent(
                    f"""\
                    {REASONING}
                    ```json
                    {{
                      "answer": {answers}
                    }}
                    ```"""
                ),
            },
        ]
    }


def get_multi_turn_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answer = False
        answers = [NO_RESPONSE]
    else:
        answer = True
    if not all_answers:
        answers = answers[0]
    answers = json.dumps(answers)

    return {
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": dedent(
                    f"""\
                    Does the the following context contain enough information to be able to answer the question? Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
                    ```json
                    {{
                      "answer": ... # "yes" or "no"
                    }}
                    ```
                    Context: {context}
                    Question: {question}"""
                ),
            },
            {
                "role": "assistant",
                "content": dedent(
                    f"""\
                    {REASONING}
                    ```json
                    {{
                      "answer": "{'yes' if answer else 'no'}"
                    }}
                    ```"""
                ),
            },
            {
                "role": "user",
                "content": dedent(
                    f"""\
                    If the question has an answer in the context, extract the minimal span word for word from the context that best answers the question, otherwise the answer should be "{NO_RESPONSE}". Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
                    ```json
                    {{
                      "answer": ...
                    }}
                    ```"""
                ),
            },
            {
                "role": "assistant",
                "content": dedent(
                    f"""\
                    {REASONING}
                    ```json
                    {{
                      "answer": {answers}
                    }}
                    ```"""
                ),
            },
        ]
    }


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    instruction = {
        "single_turn": get_single_turn_prompt_and_response,
        "multi_turn": get_multi_turn_prompt_and_response,
    }[script_args.prompt]

    squad_dataset = load_dataset("squad_v2")
    dataset = squad_dataset["train"].train_test_split(
        test_size=script_args.validation_ratio,
        seed=script_args.seed,
    )
    train_dataset = dataset["train"].map(instruction)
    val_dataset = dataset["test"].map(instruction, fn_kwargs={"all_answers": True})
    test_dataset = squad_dataset["validation"].map(
        instruction, fn_kwargs={"all_answers": True}
    )
    dataset = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )
    dataset.save_to_disk(config.dataset_name)
