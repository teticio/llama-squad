from typing import Tuple, Union

import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlamaForMaskedCausalLM(LlamaForCausalLM):
    blah_token_id = 29268

    def forward(self, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        blah = kwargs["labels"] == self.blah_token_id
        # # Don't attend "blah" tokens
        kwargs["attention_mask"] = ~blah
        # Don't calculate CE loss for "blah" tokens
        kwargs["labels"] = torch.where(blah, -100, kwargs["labels"])
        return super(LlamaForMaskedCausalLM, self).forward(**kwargs)
