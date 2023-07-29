# Llama Masked

![Llama masked](llama_masked.png)

## Motivation

Before ChatGPT, we typically used encoder (discriminative) models to solve specific tasks such as classification and question answering. In fact, many of the SOTA results for these kind of tasks appear to have got stuck in time. Back then, decoder (generative) models like GPT seemed like they could be of little more use than to generate an alternative ending to a Harry Potter book. However, a first hint of their surprising utility was uncovered in the Open AI paper, ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in which they demonstrated the ability of GPT-2 (1.5B parameters!) to perform a variety of tasks such as translation, question answering, and summarization, all without the need for task-specific training.

ChatGPT works astonishingly well on a wide range of tasks, but it is not possible to specialize on a custom task beyond techniques such as "few shot learning" and "prompt engineering". In particular, there is no effective way to feed back examples that do not pass human validation, in order to improve model performance with active learning. Furthermore, there is no guarantee that OpenAI will maintain the performance of the model for your use case (see ["How is ChatGPT Behaviour Changing Over Time?"](https://arxiv.org/pdf/2307.09009.pdf)).

Thankfully, there have been great advances in Open Source, from models like Llama 2 (7B - 70B parameters) to techniques like QLoRA (Quantized Low-Rank Adaption) that make fine-tuning these models on a single prosumer GPU possible.

As an aside, encoders are not usually used for generation, but they can be persuaded to do so. As they were pre-trained on Masked Language Modelling, you can get them to "fill in the blanks". Unfortunately, they quickly become incoherent when several blanks appear together. Nevertheless, using MCMC (Markov Chain Monte Carlo) methods, it is possible to get [good results](https://github.com/teticio/inBERTolate). If you attempt this with a encoder that has been fine-tuned on a specific task, you will find it generates gibberish. The "fine-tuning" is causing catastrophic forgetting that may or may not be an issue for your particular case. Whatever the case, if a model can be used for multiple purposes, then the cost of training it can be more easily justified. I suspect that this is why available decoder models are so much larger than encoder models.

It seems plausible that decoders may have some advantages over encoders for some tasks that require "reasoning". An encoder is specialized on a task by including a classification head, which is usually simply a dense layer. [Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A) compares encoders to System 1 (automatic) thinking and decoders to System 2 (logical) thinking - Posner's classification of thought processes that was popularized by Daniel Kahneman. Indeed, Prompt Engineers have discovered that adding a seemingly innocuous instruction like "Think step by step" can improve the accuracy of the results. Certainly, the process of generating a sequence of tokens (words) requires much more computation than sticking a dense layer on top of an encoder, as you have to generate each token auto-regressively by feeding the previous generations into the model in sequence. This, combined with the "chat" approach of alternating between generated tokens and human input, seems to give the model more opportunity to "think" about the answer.

So how can we specialize these generalized decoder models? How can we put the Human back In The Loop?

## Approach

While ChatGPT has been fine-tuned with RLHF (Reinforcement Learning with Human Feedback) - a complex process still very much in research - a paper published by Meta called ["LIMA: Less Is More for Alignment"](https://arxiv.org/pdf/2305.11206.pdf) claims that SFT (Supervised Fine-Tuning) may be sufficient with relatively few examples, provided they are of very high quality.

The idea here is to use the excellent [TRL]() library from Hugging Face to SFT a model on the SQuAD v2 task. The current SQuAD v2 [SOTA](https://paperswithcode.com/sota/question-answering-on-squad20) is an Exact Match for 90.9% of the test set. The dataset consists of contexts, questions and answers - which are verbatim extracts from the contexts. In some cases, there *is* no answer to the question in the context, and so the answer is an empty string. Foundation models like ChatGPT may excel in "reasoning", but it can be challenging to ensure that the answers are taken word-for-word from the context. In the case that there is no answer, there is an additional risk that they will provide an answer from memory or a "hallucination".

A baseline approach would be to generate training data of the form:

````
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Use the following context to answer the question and give the answer in JSON format as follows:
```json
{
  "answer": ...
}
```
Context: In April, during the Revolution of 1848 in Paris, he left for London, where he performed at several concerts and at numerous receptions in great houses. This tour was suggested to him by his Scottish pupil Jane Stirling and her elder sister. Stirling also made all the logistical arrangements and provided much of the necessary funding.
Question: Where did Chopin go in the spring of 1848? [/INST] ```json
{
  "answer": "London"
}
``` </s>
````

There is a system prompt enclosed by `<<SYS>>` and `<</SYS>>` tokens, followed by an instruction terminated by `[/INST]`, and finally the ground truth answer (as a continuation of the prompt). This is the template that was used to train the chat version of Llama 2.

With this approach, we will be fine-tuning a model with a number of parameters an order of magnitude greater than the largest Open Source encoder models currently available, but we will not be taking advantage of the "reasoning" capabilities of decoders.

## Masked Causal Language Modeling

Alternatively, we can provide a multi-turn prompt with the following instructions:
1) Use the following context to answer the question.
2) Extract the minimal span word for word from the context that best answers the question.
3) Now give the answer in JSON format as follows.

One complication with this idea is that, while we have the questions, the context and the answers, we do not have a ground-truth for the reasoning to get to that answer. For that, we could use ChatGPT, in a similar way to how the Alpaca model was trained. Or, we could simply replace the reasoning with "blah blah blah..." and adapt the training of the model accordingly.

Consider training data of the following form, where there are several instructions in the `[INST]` and `[/INST]` token blocks, and the `blah` tokens are repeated a number of times:

````
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Use the following context to answer the question.
Context: The origin of Tom Robinson is less clear, although many have speculated that his character was inspired by several models. When Lee was 10 years old, a white woman near Monroeville accused a black man named Walter Lett of raping her. The story and the trial were covered by her father's newspaper which reported that Lett was convicted and sentenced to death. After a series of letters appeared claiming Lett had been falsely accused, his sentence was commuted to life in prison. He died there of tuberculosis in 1937. Scholars believe that Robinson's difficulties reflect the notorious case of the Scottsboro Boys, in which nine black men were convicted of raping two white women on negligible evidence. However, in 2005, Lee stated that she had in mind something less sensational, although the Scottsboro case served "the same purpose" to display Southern prejudices. Emmett Till, a black teenager who was murdered for flirting with a white woman in Mississippi in 1955, and whose death is credited as a catalyst for the Civil Rights Movement, is also considered a model for Tom Robinson.
Question: Who's death was a catalyst for the Civil Rights Movement? [/INST] blah ... blah </s><s>[INST] Extract the minimal span word for word from the context that best answers the question. [/INST] Emmett Till </s><s>[INST] Now give the answer in JSON format as follows:
```json
{
  "answer": ...
}
``` [/INST] ```json
{
  "answer": "Emmett Till"
}
```
 </s>
 ````

Then we can train an adapted Llama model to not pay attention to the `blah`s nor to include this section of the tokens in the cross-entropy loss. This is achieved by simply adding the following lines to the `forward` method either side of the call to the model:

```python
# Don't attend "blah" tokens
attention_mask = labels != self.blah_token_id
```

```python
# Masked CE loss
shift_attention_mask = attention_mask[..., 1:].contiguous()
shift_attention_mask = shift_attention_mask.view(-1)
shift_attention_mask = shift_attention_mask.to(shift_logits.device)
masked_loss = torch.where(
    shift_attention_mask, loss, torch.zeros_like(loss)
)
loss = masked_loss.sum() / (
    shift_attention_mask.sum() - (shift_labels == -100).sum()
)
```

Remember that the model generates all the tokens in parallel in the forward pass, because we inject the ground-truth tokens in the input using [teacher forcing](https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c). If we were to pass the generation of the previous token as an input to the next token - as is done in generation at inference time - this would mean reverting to a sequential calculation, which would be infeasible. This means that the model is not guided during the `blah blah blah` section, and is likely to go off at a tangent, rather like the encoder models do when generating text as mentioned previously.

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

The training script `finetune_llama_v2.py` is heavily based on [one](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da) provided by `younesbelkada`.

```bash
./train.sh
```

### Evaluate model

```bash
python test_model.py
```
