from transformers import DataCollatorForLanguageModeling


class SquadDataCollator(DataCollatorForLanguageModeling):
    answer_start_token_id = 7521  # "_```"

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for _ in range(batch["labels"].size(0)):
            answer_start = (
                batch["labels"][_] == self.answer_start_token_id
            ).nonzero(as_tuple=True)[0][-1]
            batch["labels"][_][:answer_start] = -100

        return batch
