import csv
import json
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser, pipeline


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="deepset/deberta-v3-large-squad2")
    batch_size: Optional[int] = field(default=8)
    output_csv_file: Optional[str] = field(default="results/results_encoder.csv")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset("squad_v2")["validation"]
pipe = pipeline(
    "question-answering",
    model=script_args.model_name,
    tokenizer=script_args.model_name,
    device=0,
)


def process_batch(batch):
    inputs = [
        {"question": q, "context": c}
        for q, c in zip(batch["question"], batch["context"])
    ]
    outputs = pipe(inputs)
    if not isinstance(outputs, list):
        outputs = [outputs]
    result = {key: [output[key] for output in outputs] for key in outputs[0].keys()}
    return result


results = dataset.map(process_batch, batched=True, batch_size=script_args.batch_size)

with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Context",
            "Question",
            "Correct answers",
            "Model answer",
            "Exact match",
        ]
    )

    for result in tqdm(results):
        context = result["context"]
        question = result["question"]
        answers = result["answers"]["text"]
        if len(answers) == 0:
            answers = ["?"]
        model_answer = "?" if result["score"] < 0.5 else result["answer"].strip()
        if model_answer[-1] in [".", ",", ";"]:
            model_answer = model_answer[:-1]
        exact_match = model_answer in answers

        writer.writerow(
            [context, question, json.dumps(answers), model_answer, exact_match]
        )
        file.flush()
