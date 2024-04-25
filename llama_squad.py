from transformers import DataCollatorForLanguageModeling
import torch


class SquadDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, answer_start_tokens: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            answer_end = torch.where(label == -100)[0]
            window = label.unfold(0, self.answer_start_tokens.shape[0], 1)
            answer_start = (
                (window == self.answer_start_tokens).all(dim=1).nonzero()[-1, 0]
            )
            label[:answer_start] = -100
            if len(answer_end) > 0:
                label[answer_end[0]] = self.tokenizer.eos_token_id
            batch["labels"][idx] = label

        return batch
