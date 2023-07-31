from typing import Tuple, Union

from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlamaForMaskedCausalLM(LlamaForCausalLM):
    blah_token_id = 29268
    answer_start_token_id = 7521

    def forward(self, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        # # Don't attend "blah" tokens
        kwargs["attention_mask"] = kwargs["labels"] != self.blah_token_id
        # Only calculate CE loss for the answer section of the labels
        for batch in range(kwargs["labels"].size(0)):
            answer_start = (
                kwargs["labels"][batch] == self.answer_start_token_id
            ).nonzero(as_tuple=True)[0][-1]
            kwargs["labels"][batch][:answer_start] = -100
        return super(LlamaForMaskedCausalLM, self).forward(**kwargs)
