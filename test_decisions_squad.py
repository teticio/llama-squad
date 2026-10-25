import asyncio
import csv
import json
import logging
import random
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import yaml
from datasets import load_from_disk
from tqdm.asyncio import tqdm
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from transformers import HfArgumentParser

from create_squad_dataset import NO_RESPONSE, is_exact_match

# A choice question accepts up to 255 choices
MAX_OPTIONS = 255
NONE = "none"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-6-luna")
    output_csv_file: Optional[str] = field(default="results/results_decisions.csv")
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
        default=1.0,
        metadata={
            "help": "Choose the shortest span with at least this fraction of the highest probability"
        },
    )
    concurrency: Optional[int] = field(default=16)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logger = logging.getLogger("test_decisions_squad")
logger.setLevel(level=logging.DEBUG if script_args.debug else logging.INFO)

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))
dataset = load_from_disk(config.dataset_name)
# The model is not fine-tuned, so we show it some examples of what a minimal span looks like
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


def get_input(state):
    return [{"role": "user", "content": [{"type": "input_json", "data": state}]}]


async def decide(client, semaphore, state, questions):
    errors = 0
    while True:
        try:
            async with semaphore:
                return await client.decisions.create(
                    model=script_args.model_name,
                    input=get_input(state),
                    questions=questions,
                )
        except RateLimitError as e:
            if e.code in ("insufficient_quota", "credit_balance_exhausted"):
                raise
            # Rate limits are expected, so keep trying with jitter to avoid starvation
            await asyncio.sleep(random.uniform(1, 10))
        except (APIConnectionError, InternalServerError) as e:
            errors += 1
            if errors > 10:
                raise
            logger.warning("%s: sleeping for %s seconds", type(e).__name__, 2**errors)
            await asyncio.sleep(2**errors)


def get_answers(decision):
    # Refusals have no probabilities, so they are treated as zero
    return {
        answer.name: (
            answer.probability
            if answer.type == "predicate"
            else {p.value: p.probability for p in answer.probabilities}
        )
        for answer in decision.answers
        if answer.type != "refusal"
    }


def get_questions(question, chunks):
    # The Decisions API does not generate text, so we ask it to point to the start
    # and the end of the span, in the same way as an encoder model
    questions = [
        {
            "type": "predicate",
            "name": "answerable",
            "instructions": f'Does `context` contain the answer to the question: "{question}"? '
            "Answer yes only if the context explicitly states the answer to exactly this "
            "question, and no if the context does not state the answer, or the question "
            "is about something different from what the context says.",
        }
    ]
    for k, chunk in enumerate(chunks):
        choices = [{"value": token_id(i), "description": ""} for i in chunk]
        if len(chunks) > 1:
            choices.append(
                {"value": NONE, "description": "The span is not among these tokens"}
            )
        for boundary, position in (("start", "first"), ("end", "last")):
            questions.append(
                {
                    "type": "choice",
                    "name": f"{boundary}_{k}",
                    "instructions": f"Each token in `context` is enclosed between [id] and "
                    f"[/id] markers. Which is the {position} token of the minimal span of "
                    f'`context` that best answers the question: "{question}"?',
                    "choices": choices,
                }
            )
    return questions


def get_span(answers, chunks, max_answer_tokens):
    best = (0, None, None)
    for k, chunk in enumerate(chunks):
        start = answers.get(f"start_{k}", {})
        end = answers.get(f"end_{k}", {})
        for i in chunk:
            p_start = start.get(token_id(i), 0)
            if p_start <= best[0]:
                continue
            for j in range(i, min(i + max_answer_tokens, chunk[-1] + 1)):
                p = p_start * end.get(token_id(j), 0)
                if p > best[0]:
                    best = (p, i, j)
    return best


def get_candidates(answers, chunks, start, end, num_boundaries):
    # The most likely boundaries, together with those neighbouring the best span
    k = next(k for k, chunk in enumerate(chunks) if start in chunk)
    candidates = []
    for boundary, best in (("start", start), ("end", end)):
        probabilities = answers.get(f"{boundary}_{k}", {})
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
    # Choosing between literal excerpts is easier than pointing to tokens
    return {
        "type": "choice",
        "name": "span",
        "instructions": "Choose the excerpt of `context` that a SQuAD annotator would "
        "highlight as the answer to the question: the minimal span, word for word, "
        "that answers it.\n\nExamples of questions and answers:\n"
        + "\n".join(f"- {json.dumps(example)}" for example in examples)
        + f'\n\nQuestion: "{question}"',
        "choices": [{"value": span, "description": ""} for span in spans],
    }


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
    decision = await decide(
        client, semaphore, state, get_questions(sample["question"], chunks)
    )

    answers = get_answers(decision)
    answerable = answers.get("answerable", 0)
    _, start, end = get_span(answers, chunks, script_args.max_answer_tokens)
    if start is None:
        return answerable, NO_RESPONSE
    span = context[tokens[start].start() : tokens[end].end()]
    if script_args.num_boundaries == 0:
        return answerable, span

    starts, ends = get_candidates(
        answers, chunks, start, end, script_args.num_boundaries
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
    if len(spans) == 1:
        return answerable, spans[0]
    decision = await decide(
        client,
        semaphore,
        {"context": context},
        [get_refine_question(sample["question"], spans)],
    )
    # With relative_probability < 1, prefer a shorter span that is nearly as likely
    probabilities = get_answers(decision).get("span")
    if not probabilities:
        return answerable, span
    min_probability = script_args.relative_probability * max(probabilities.values())
    return answerable, next(
        span for span in spans if probabilities.get(span, 0) >= min_probability
    )


async def main():
    semaphore = asyncio.Semaphore(script_args.concurrency)
    async with AsyncOpenAI() as client:
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
