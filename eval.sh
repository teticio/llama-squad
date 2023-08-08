#!/usr/bin/env bash

# clone repo from https://github.com/EleutherAI/lm-evaluation-harness.git
# ./eval.sh [path to peft checkpoint]

# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
tasks=arc_challenge,hellaswag,truthfulqa_mc
few_shot_tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions

if [ $# -gt 0 ]; then
    python ../lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/mnt/scratch/Llama-2-7b-chat-hf,peft=$1 \
    --tasks $few_shot_tasks \
    --num_fewshot 5 \
    --device cuda:0
    python ../lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/mnt/scratch/Llama-2-7b-chat-hf,peft=$1 \
    --tasks $tasks \
    --device cuda:0
else
    python ../lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/mnt/scratch/Llama-2-7b-chat-hf \
    --tasks $few_shot_tasks \
    --num_fewshot 5 \
    --device cuda:0
    python ../lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/mnt/scratch/Llama-2-7b-chat-hf \
    --tasks $tasks \
    --device cuda:0
fi
