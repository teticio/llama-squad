from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser

from utils import DEFAULT_SYSTEM_PROMPT, get_prompt


@dataclass
class ScriptArguments:
    base: Optional[bool] = field(
        default=False,
        metadata={"help": "Use base instruction format."},
    )
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )


SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
REASONING_LENGTH = 100
reasoning = " ".join(["blah"] * REASONING_LENGTH)


def get_base_prompt_and_response(item):
    context = item["context"]
    question = item["question"]
    answer = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else "?"
    return {
        "text": get_prompt(
            f"""\
Use the following context to answer the question and give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```
Context: {context}
Question: {question}""",
            [],
            SYSTEM_PROMPT,
        )
        + f""" \
```json
{{
  "answer": "{answer}"
}}
``` </s>"""
    }


def get_prompt_and_response(item):
    context = item["context"]
    question = item["question"]
    answer = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else "?"
    return {
        "text": get_prompt(
            """\
Now give the answer in JSON format as follows:
```json
{
  "answer": ...
}
```
""",
            [
                (
                    f"""\
Use the following context to answer the question.
Context: {context}
Question: {question}""",
                    f"""\
{reasoning}""",
                ),
                (
                    f"""\
Extract the minimal span word for word from the context that best answers the question.
        """,
                    f"""\
{answer}""",
                ),
            ],
            SYSTEM_PROMPT,
        )
        + f""" \
```json
{{
  "answer": "{answer}"
}}
```
 </s>"""
    }


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

squad_dataset = load_dataset("squad_v2")
instruction = (
    get_base_prompt_and_response if script_args.base else get_prompt_and_response
)
train_dataset = squad_dataset["train"].map(instruction)
test_dataset = squad_dataset["validation"].map(instruction)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset.save_to_disk(script_args.dataset)
