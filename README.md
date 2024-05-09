# LLaMA SQuAD

![LLaMA SQuAD](llama_squad.png)

## TL;DR

Encoder models based on BERT typically excel at "exact tasks" such as SQuAD, however there is currently much less investment in training large Open Source encoder models, likely because they are less widely applicable out of the box than foundation models. The purpose is of this repo is to explore whether it is possible to combine the best of both worlds: the "reasoning" abilities of large foundation models and the specialization capabilities of encoder models.

This repo uses the TRL (Transformer Reinforcement Library) to fine-tune Meta's Llama 2 & 3 models on the SQuAD v2 task. This is a particularly challenging task for generative (decoder) models like Llama, because it requires abstention when the answer is not in the context and exact extraction from the context when it is present. If we can fine-tune a decoder model so that it is more "honest" about what it cannot answer and to always give answers in a predictable format, then it should be possible to specialize these generalized foundation models on a wide range of tasks.

While a lot of progress has been made in the field of fine-tuning for more general tasks, we find that it is necessary to adapt the procedure in order to get good results.

**UPDATE:** Adpated to work with with Llama 3 and added results.

## Motivation

Before ChatGPT, we typically used encoder (discriminative) models to solve specific tasks such as classification and question answering. In fact, many of the SOTA results for these kind of tasks appear to have got stuck in time. Back then, decoder (generative) models like GPT seemed like they could be of little more use than to generate an alternative ending to a Harry Potter book. However, a first hint of their surprising utility was uncovered in the Open AI paper, ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in which they demonstrated the ability of GPT-2 (1.5B parameters!) to perform a variety of tasks such as translation, question answering, and summarization, all without the need for task-specific training.

ChatGPT works astonishingly well on a wide range of tasks, but it is not possible to specialize on a custom task beyond techniques such as "few shot learning" and "prompt engineering". In particular, there is no effective way to feed back examples that do not pass human validation, in order to improve model performance with active learning. Furthermore, there is no guarantee that OpenAI will maintain the performance of the model for your use case (see ["How is ChatGPT Behaviour Changing Over Time?"](https://arxiv.org/pdf/2307.09009.pdf)).

Thankfully, there have been great advances in Open Source, from models like Llama 2 (7B - 70B parameters) to techniques like QLoRA (Quantized Low-Rank Adaption) that make fine-tuning these models on a single "prosumer" GPU possible.

As an aside, encoders are not usually used for generation, but they can be persuaded to do so. As they were pre-trained on Masked Language Modelling, you can get them to "fill in the blanks". Unfortunately, they quickly become incoherent when several blanks appear together. Nevertheless, using MCMC (Markov Chain Monte Carlo) methods, it is possible to get [good results](https://github.com/teticio/inBERTolate). If you attempt this with an encoder that has been fine-tuned on a specific task, you will find it generates gibberish. The "fine-tuning" is causing catastrophic forgetting that may or may not be an issue for your particular use case. Whatever the case, if a model can be used for multiple purposes, then the cost of training it can be more easily justified. I suspect that this is why currently available Open Source decoder models are so much larger than encoder models.

It seems plausible that decoders may have some advantages over encoders for some tasks that require "reasoning". An encoder is specialized on a task by including a "head", which is often simply a dense layer. [Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A) compares encoders to System 1 (automatic) thinking and decoders to System 2 (logical) thinking - Posner's classification of thought processes that was popularized by Daniel Kahneman. Indeed, Prompt Engineers have discovered that adding a seemingly innocuous instruction like "Think step by step" can improve the accuracy of the results. Certainly, the process of generating a sequence of tokens (words) requires much more computation than sticking a dense layer on top of an encoder, as you have to generate each token auto-regressively by feeding the previous generations into the model in sequence. This, combined with the "chat" approach of alternating between generated tokens and human input, seems to give the model more opportunity to "think" about the answer. Of course, the fact that decoder models can also give an explanation as part of the answer makes them more suitable for human validation.

So how can we specialize these generalized decoder models? How can we put the Human back In The Loop?

## Approach

While ChatGPT has been fine-tuned with RLHF (Reinforcement Learning with Human Feedback) - a complex process still very much in research - a paper published by Meta called ["LIMA: Less Is More for Alignment"](https://arxiv.org/pdf/2305.11206.pdf) claims that SFT (Supervised Fine-Tuning) may be sufficient with relatively few examples, provided they are of very high quality.

The idea here is to use the excellent [TRL](https://github.com/lvwerra/trl) library from Hugging Face's Leandro von Werra to SFT a model on the SQuAD v2 task. The current SQuAD v2 [SOTA](https://paperswithcode.com/sota/question-answering-on-squad20) is an Exact Match for 90.9% of the test set. The dataset consists of contexts, questions and answers - which are verbatim extracts from the contexts. In some cases, there *is* no answer to the question in the context, and so the answer is an empty string. Foundation models like ChatGPT may excel in "reasoning", but it can be challenging to ensure that the answers are taken word-for-word from the context. In the case that there is no answer, there is an additional risk that they will provide an answer from memory or a "hallucination". There is, of course, also a risk that the SQuAD v2 test dataset was used directly or indirectly as part of the training set of the foundation model.

We create a dataset of chats, which can be converted easily into a prompt with `tokenizer.apply_chat_template`, for example:

```yaml
messages:
- role: "system"
  content: |
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
- role: "user"
  content: |
    Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
    ```json
    {
      "answer": ...
    }
    ```
    If the answer is not in the context, the answer should be "?".
    Context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
    Question: When did Beyonce start becoming popular?
- role: "assistant"
  content: |
    ```json
    {
      "answer": "in the late 1990s"
    }
    ```
```

### Masked Causal Language Modeling

We would like to retain the chat capacity of the model, while improving the accuracy of its responses. In order to this, we limit the cross entropy loss in the forward method of the model to only apply to the tokens in each of the assistant responses.

We can train the model in this way by creating a custom `DataCollator` (see [`llama_squad.py`](llama_squad.py)) which sets the target tokens to which we do not want to apply cross entropy loss to `-100` (a special value in Hugging Face's `transformers` package). To verify which parts of the prompt the cross entropy loss will be applied to, you can run `python test_data_collator.py` and these sections will be coloured green.

### Blah blah blah

Notice that we include
````
Think step by step and explain your reasoning.
````
in the prompt. It would no doubt be beneficial to include the reasoning in the training data. As we do not have a ground-truth for the reasoning, one approach would be to use ChatGPT to generate it (knowing the answer) in a similar way to how the Alpaca model was trained.

Alternatively we can replace the reasoning with `<blah><blah><blah>...` tokens. When I tried this experiment in July 2023, I didn't obtain good results, as the model appeared to just learn to spit out these tokens. I tried masking the attention, in the hope that it would learn to at least space out the answer due to the relative positional embeddings, but this made the results worse. In hindsight, and after reading the paper ["Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/pdf/2310.02226) first published in October 2023 (thanks Thom!), I realized that it was important to use a special token that was *learnable* (i.e., one that was not already in the vocabulary). By introducing these tokens, we multiply the number of operations (or amount of "thinking") between the question and the answer. At inference time, the output corresponding to these tokens is ignored and not fed back into the model auto-regressively, as it need not make any linguistic sense. I found it helped to also include start of the answer `\n```json` in the prompt, especially early on in the training. However, the results from the paper point to the importance of not just fine-tuning but pre-training the model with the `<blah>` tokens (or using their nomenclature, `<pause>` tokens); we are only doing LORA fine-tuning. During training, of course, we do not apply cross entropy loss to the outputs corresponding to these tokens.

### Multi-turn prompt

It can be interesting to break the problem into explicit steps, taking advantage of the fact that these models were pre-trained with a chat format. For the SQuAD task in particular, we can first of all try to determine whether the context provides an answer to the question before actually attempting to answer the question.

```yaml
messages:
- role: system
  content: ...
- role: user
- content: |
    Does the the following context contain enough information to be able to answer the question? Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
    ```json
    {
      "answer": ... # "yes" or "no"
    }
    ```
    Context: ...
    Question: ...
- role: assistant
  content: |
    ```json
    {
      "answer": ...
    }
    ```
- role: user
- content: |
    If the question has an answer in the context, extract the minimal span word for word from the context that best answers the question, otherwise the answer should be "?". Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
    ```json
    {  
      "answer": ...
    }
    ```
- role: assistant
  content: |
    ```json
    {
      "answer": ...
    }
    ```
  role: assistant
```

### Learning Rate

The first experiment was performed with a learning rate of `2e-4`. The model quickly learned to respond reasonably well to the task, but it just as quickly completely forgot how to speak English. In order to be able to train over the full training set without catastrophic forgetting, we set the learning rate to `2e-7`. This was selected, somewhat heuristically, such that the "elbow" in the convergence graph coincided approximately with one epoch. With Llama 3 we found it necessary to use a larger learning rate `1e-6` with a warmup, to avoid the loss getting stuck at a higher value.

### LORA rank

LORA works by inserting [low rank](https://web.stanford.edu/class/cs168/l/l9.pdf) trainable weight matrices between layers, while freezing all the other weights. Lower rank matrices can be compressed into fewer parameters but are less expressive. We tried ranks of 16, 64 and 512 but didn't notice any significant difference in performance, so we settled on 64.

## Results

On the test set, the models achieve the following [results](https://docs.google.com/spreadsheets/d/1N4XyrAyzKOHEmpAFvfRzEjZZis1T61_ekFeFbFW0lYM/edit?usp=sharing), where we have included Llama 2 70b chat, GPT 3.5 Turbo and DeBERTa (an encoder model) for reference:

| Model                           | % Valid JSON | % Exact Match | % EM for Valid JSON | % Correct No Answer | % Correct Has Answer |
| ----------------------------------- | ------------ | ------------- | ------------------- | ------------------- | -------------------- |
| Llama 2 7b chat (base model)        | 66.42%       | 18.76%        | 28.24%              | 3.72%               | 33.82%               |
| - [Fine-tuned single-turn 1.2 epochs](https://wandb.ai/teticio/huggingface/runs/p00jazs1) | 97.17%       | 47.22%        | 48.60%              | 39.44%              | 55.02%               |
| - 3.7 epochs                        | 98.85%       | 64.71%        | 65.46%              | 65.85%              | 63.56%               |
| - 8.0 epochs 1 beam                 | 98.83%       | 73.11%        | 73.97%              | 79.90%              | 66.30%               |
| - 8.0 epochs 10 beams               | 99.75%       | 74.99%        | 75.18%              | 82.02%              | 67.95%               |
| Llama 2 70b chat (quantized)        | 95.30%       | 35.80%        | 37.57%              | 17.69%              | 54.12%               |
| OpenAI GPT 3.5 Turbo*               | 83.80%       | 47.60%        | 56.80%              | 40.78%              | 54.10%               |
| OpenAI GPT 4*                       | 99.90%       | 63.50%        | 63.56%              | 77.08%              | 50.30%               |
| deepset/deberta-v3-large-squad2     | N/A          | 80.01%        | N/A                 | 94.67%              | 65.30%               |
| Llama 3 8b instruct (base model)    | 96.98%       | 51.85%        | 53.47%              | 37.21%              | 66.54%               |
| - Fine-tuned single-turn 1.2 epochs | 99.83%       | 70.03%        | 70.15%              | 69.92%              | 70.13%               |
| - 100 `<blah>`s 1.2 epochs          | 99.98%       | 66.82%        | 66.83%              | 59.19%              | 74.46%               |
| - 25 `<blah>`s 1.2 epochs           | 100.00%      | 72.45%        | 72.45%              | 77.21%              | 67.68%               |

\* In these cases, the test was run on a random subset of 1,000 examples, due to costs or long inference times.

The fine-tuned model has clearly learned to respect JSON format, has learned to abstain more often and has greatly improved the exact matches (although this is still far from SOTA!). In fact, it performs substantially better than its big brother Llama 70b chat and even beats OpenAI's GPT 4. Of course, DeBERTA is the clear winner - mainly thanks to its abstinence - but I suspect that it may have learned to abstain when the question is phrased in a particular way (e.g., "What is *not* a...".)

As the Llama 2 model is fine-tuned over more epochs, it continues to improve its accuracy on the SQuAD v2 task, up until about 8 epochs. It also tends to adhere more strictly to the output format, to the point of not returning an explanation in most cases (although it is still possible to ask it to produce its reasoning with a follow up prompt). To be sure that the model is not catastrophically forgetting, we compared its performance over the benchmark tests used in the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). We found that it performed as well as the base model in most cases, and even slightly better in others.

A qualitative analysis of the results reveals that the 7B parameter model is inherently limited by its reasoning capabilities. It is often tripped up by deliberately misleading questions, such as the following:

> Studies on income inequality and growth have sometimes found evidence confirming the Kuznets curve hypothesis, which states that with economic development, inequality first increases, then decreases. Economist Thomas Piketty challenges this notion, claiming that from 1914 to 1945 wars and "violent economic and political shocks" reduced inequality. Moreover, Piketty argues that the "magical" Kuznets curve hypothesis, with its emphasis on the balancing of economic growth in the long run, cannot account for the significant increase in economic inequality throughout the developed world since the 1970s.

> What isn't Thomas Piketty's job?

The table indicates that fine-tuning the 70B parameter model could yield interesting results. To give some idea of the cost, a A100 with 80Gb VRAM currently costs $1.69 an hour on [runpod.io](https://runpod.io). The maximum batch size that fits is 2 and 1 training step takes about 45 seconds, so 10,000 steps (1.2 epochs) would cost around $200.

## How to use

### Install requirements

```bash
pip install -r requirements.txt
```

### Create dataset

```bash
python create_squad_dataset.py
```

### Train model

The training script `train_llama_squad.py` is heavily based on [one](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da) provided by `younesbelkada`. You may need to adjust the `per_device_train_batch_size` in order to fit the model in your GPU memory. You should also set the `gradient_accumulation_steps` so that `per_device_train_batch_size * gradient_accumulation_steps` is preserved.

`config.yaml`:
```yaml
model_name: meta-llama/Meta-Llama-3-8B-Instruct
system_prompt: |
  You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
  If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
reasoning_tokens: 20
dataset_name: data/squad_v2
```

```bash
python train_llama_squad.py \
--bf16 \
--max_seq_length 4096 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--max_steps 10000 \
--merge_and_push \
--save_steps 1000 \
--learning_rate=2e-7
```

Alternatively, to ensure you are using a reproducible environment, you can train in Docker with

```bash
docker-compose run llama-squad python train_llama_squad.py \
...
```

### Evaluate model

If you run out of GPU memory, pass the parameter `--quantize` to the script.

```bash
python test_llama_squad.py --adapter_name=results/final_checkpoints
```

This generates a CSV file `results/results.csv` which you can summarize with

```bash
python summarize_results.py
```

To see how the model performs on the benchmarks that are tracked in the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) clone [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) in the same directory as `llama-squad` and run

```bash
./eval.sh <path_to_base_model> results/final_checkpoints
```

You can test out the model interactively with

```bash
python app.py --adapter_name=results/final_checkpoints
```
