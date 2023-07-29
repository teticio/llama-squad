from datasets import Dataset, DatasetDict, load_dataset

from utils import DEFAULT_SYSTEM_PROMPT, get_prompt


SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
REASONING_LENGTH = 100
reasoning = " ".join(["blah"] * REASONING_LENGTH)


def get_prompt_and_response(item):
    context = item["context"]
    question = item["question"]
    answer = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else "?"
    return (
        get_prompt(
            """\
Now give the answer in JSON format as follows:
```json
{
  "answer": ...
}""",
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
}} </s>"""
    )


squad_dataset = load_dataset("squad_v2")
train_data = {
    "text": [get_prompt_and_response(item) for item in squad_dataset["train"]]
}
test_data = {
    "text": [get_prompt_and_response(item) for item in squad_dataset["validation"]]
}
dataset = DatasetDict(
    {"train": Dataset.from_dict(train_data), "test": Dataset.from_dict(test_data)}
)
dataset.save_to_disk("data/squad_v2")
