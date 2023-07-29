from datasets import load_from_disk
import random
import torch
import transformers

pipeline = transformers.pipeline(
    "text-generation",
    # model="meta-llama/Llama-2-7b-chat-hf",
    model="results/final_merged_checkpoint",
    tokenizer="meta-llama/Llama-2-7b-chat-hf",
    trust_remote_code=True,
    device_map="auto",
    model_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
)

while True:
    dataset_dict = load_from_disk("data/squad_v2")
    text = random.choice(dataset_dict["test"])["text"]
    question = text[:text.find("[/INST] ") + len("[/INST] ")]
    answer = text[text.rfind("```") :]
    print(f"Question: {question}")
    print(f"Correct answer: {answer}")
    print()

    response = ""
    while True:
        prompt = response
        instruction = text.find("[/INST] ")
        if instruction == -1:
            break
        instruction += len("[/INST] ")
        prompt += text[: instruction] + "</s>"
        text = text[instruction :]
        text = text[text.find("<s>") :]
        response = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        )[0]["generated_text"].strip()
        print(f"Response: {response[len(prompt) :]}")
        print()

    input("Press enter to continue...")
