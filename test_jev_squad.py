import asyncio
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import yaml
from datasets import load_from_disk
from tqdm.asyncio import tqdm
from transformers import HfArgumentParser
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, NoulCriteria

from create_squad_dataset import NO_RESPONSE, is_exact_match

# A Choice question accepts up to 255 options
MAX_OPTIONS = 255
NONE = "none"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="jev-1.13.0")
    output_csv_file: Optional[str] = field(default="results/results_jev.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=None)
    split: Optional[str] = field(default="test")
    threshold: Optional[float] = field(
        default=0.8,
        metadata={"help": "Minimum probability that the question is answerable"},
    )
    max_answer_tokens: Optional[int] = field(default=30)
    num_boundaries: Optional[int] = field(
        default=15,
        metadata={"help": "Number of candidate start and end tokens to refine"},
    )
    num_examples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of training examples to show when refining"},
    )
    relative_probability: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Choose the shortest span with at least this fraction of the highest probability"
        },
    )
    concurrency: Optional[int] = field(default=32)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logger = logging.getLogger("test_jev_squad")
logger.setLevel(level=logging.DEBUG if script_args.debug else logging.INFO)

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))
dataset = load_from_disk(config.dataset_name)
# Jev is not fine-tuned, so we show it some examples of what a minimal span looks like
examples = [
    {"question": sample["question"], "answer": sample["answers"]["text"][0]}
    for sample in dataset["train"].shuffle(seed=42).select(range(100))
    if len(sample["answers"]["text"]) > 0
][: script_args.num_examples]
dataset = dataset[script_args.split]
if script_args.shuffle:
    dataset = dataset.shuffle(seed=script_args.seed)
if script_args.num_samples is not None and script_args.num_samples < len(dataset):
    dataset = dataset.select(range(script_args.num_samples))


def token_id(i):
    return f"T{i}"


def get_questions(question, chunks):
    # Jev does not generate text, so we ask it to point to the start and the end of
    # the span, in the same way as an encoder model
    questions = {
        "answerable": Noul(
            instructions=f'Does `context` contain the answer to the question: "{question}"?',
            criteria=NoulCriteria(
                true="The context explicitly states the answer to exactly this question",
                false="The context does not state the answer, or the question is about "
                "something different from what the context says",
            ),
        )
    }
    for k, chunk in enumerate(chunks):
        criteria = {token_id(i): None for i in chunk}
        if len(chunks) > 1:
            criteria[NONE] = "The span is not among these tokens"
        for boundary, position in (("start", "first"), ("end", "last")):
            questions[f"{boundary}_{k}"] = Choice(
                instructions=f"Each token in `context` is enclosed between [id] and "
                f"[/id] markers. Which is the {position} token of the minimal span of "
                f'`context` that best answers the question: "{question}"?',
                criteria=criteria,
            )
    return questions


def get_span(response, chunks, max_answer_tokens):
    best = (0, None, None)
    for k, chunk in enumerate(chunks):
        start = response.choices[f"start_{k}"].probabilities
        end = response.choices[f"end_{k}"].probabilities
        for i in chunk:
            p_start = start.get(token_id(i), 0)
            if p_start <= best[0]:
                continue
            for j in range(i, min(i + max_answer_tokens, chunk[-1] + 1)):
                p = p_start * end.get(token_id(j), 0)
                if p > best[0]:
                    best = (p, i, j)
    return best


def get_candidates(response, chunks, start, end, num_boundaries):
    # The most likely boundaries, together with those neighbouring the best span
    k = next(k for k, chunk in enumerate(chunks) if start in chunk)
    candidates = []
    for boundary, best in (("start", start), ("end", end)):
        probabilities = response.choices[f"{boundary}_{k}"].probabilities
        ranked = sorted(
            chunks[k],
            key=lambda i: (
                -probabilities.get(token_id(i), 0) - (abs(i - best) <= 2),
                abs(i - best),
            ),
        )
        candidates.append(sorted(ranked[:num_boundaries]))
    return candidates


def get_refine_question(question, spans):
    # Jev is better at choosing between literal excerpts than at pointing to tokens
    return Choice(
        instructions={
            "task": "Choose the excerpt of `context` that a SQuAD annotator would "
            "highlight as the answer to `question`: the minimal span, word for word, "
            "that answers it.",
            "examples": examples,
            "question": question,
        },
        criteria={span: None for span in spans},
    )


async def get_answer(client, semaphore, sample):
    context = sample["context"]
    tokens = list(re.finditer(r"\w+|[^\w\s]", context))
    # Leave room for the "none" option when there is more than one chunk
    chunk_size = MAX_OPTIONS if len(tokens) <= MAX_OPTIONS else MAX_OPTIONS - 1
    chunks = [
        range(i, min(i + chunk_size, len(tokens)))
        for i in range(0, len(tokens), chunk_size)
    ]
    state = {
        "context": " ".join(
            f"[{token_id(i)}]{token.group()}[/{token_id(i)}]"
            for i, token in enumerate(tokens)
        )
    }
    async with semaphore:
        response = await client.system_one(
            state, get_questions(sample["question"], chunks)
        )

    answerable = response.nouls["answerable"].noul
    _, start, end = get_span(response, chunks, script_args.max_answer_tokens)
    if start is None:
        return answerable, NO_RESPONSE
    span = context[tokens[start].start() : tokens[end].end()]
    if script_args.num_boundaries == 0:
        return answerable, span

    starts, ends = get_candidates(
        response, chunks, start, end, script_args.num_boundaries
    )
    spans = sorted(
        {
            context[tokens[i].start() : tokens[j].end()]
            for i in starts
            for j in ends
            if i <= j < i + script_args.max_answer_tokens
        },
        key=len,
    )
    async with semaphore:
        response = await client.system_one(
            {"context": context},
            {"span": get_refine_question(sample["question"], spans)},
        )
    # Jev tends to choose spans that contain the answer but are longer than necessary
    probabilities = response.choices["span"].probabilities
    min_probability = script_args.relative_probability * max(probabilities.values())
    return answerable, next(
        span for span in spans if probabilities.get(span, 0) >= min_probability
    )


async def main():
    semaphore = asyncio.Semaphore(script_args.concurrency)
    async with AsyncTypeSafeClient(
        api_key=os.environ.get("JEV_API_KEY"), model=script_args.model_name
    ) as client:
        results = await tqdm.gather(
            *[get_answer(client, semaphore, sample) for sample in dataset]
        )

    with open(script_args.output_csv_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Context",
                "Question",
                "Correct answers",
                "Model answer",
                "Exact match",
                "Answerable",
                "Span",
            ]
        )

        for sample, (answerable, span) in zip(dataset, results):
            answers = sample["answers"]["text"]
            if len(answers) == 0:
                answers = [NO_RESPONSE]
            model_answer = span if answerable >= script_args.threshold else NO_RESPONSE
            logger.debug("Correct answers: %s", answers)
            logger.debug("Model answer: %s (%.2f)", model_answer, answerable)
            exact_match = is_exact_match(model_answer, answers)

            writer.writerow(
                [
                    sample["context"],
                    sample["question"],
                    json.dumps(answers),
                    model_answer,
                    exact_match,
                    answerable,
                    span,
                ]
            )


asyncio.run(main())
