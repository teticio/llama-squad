from transformers import DataCollatorForLanguageModeling
import torch

class SquadDataCollator(DataCollatorForLanguageModeling):
    answer_start_token_id = 7521  # "_```"

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            answer_end = torch.where(label == -100)[0][0]
            answer_start = torch.where(label == self.answer_start_token_id)[0][-1]
            label[:answer_start] = -100
            label[answer_end] = 2
            batch["labels"][idx] = label

        return batch
