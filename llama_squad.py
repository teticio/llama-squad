from typing import Optional

import torch
from transformers import DataCollatorForLanguageModeling


class SquadDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        answer_start_tokens: torch.Tensor,
        answer_end_tokens: torch.Tensor,
        reasoning_tokens: Optional[torch.Tensor],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens
        self.answer_end_tokens = answer_end_tokens
        self.reasoning_tokens = reasoning_tokens

    def __call__(self, examples):
        batch = super().__call__(examples)

        for i, label in enumerate(batch["labels"]):
            # Only apply cross entropy loss to the answer part of the labels
            mask = torch.ones_like(label)
            window = label.unfold(0, self.answer_start_tokens.shape[0], 1)
            answer_starts = (window == self.answer_start_tokens).all(dim=1).nonzero()[
                :, 0
            ] + self.answer_start_tokens.shape[0]
            window = label.unfold(0, self.answer_end_tokens.shape[0], 1)
            answer_ends = (window == self.answer_end_tokens).all(dim=1).nonzero()[
                :, 0
            ] + self.answer_end_tokens.shape[0]
            for answer_start in answer_starts:
                mask[answer_start : answer_ends[answer_ends > answer_start][0]] = 0
            label = label.where(mask == 0, -100)

            # Mask out the reasoning tokens
            if self.reasoning_tokens is not None:
                mask = (label.unsqueeze(1) == self.reasoning_tokens).any(dim=1)
                label = torch.where(mask, torch.tensor(-100), label)

            batch["labels"][i] = label

        return batch
