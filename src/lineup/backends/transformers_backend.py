from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Generation, LanguageModel, Message, Scoring

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    dtype = _DTYPES[name]
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16        # e.g. a T4, which lacks native bfloat16
    return dtype


def _gather_response_logprobs(
    logprobs: torch.Tensor, prompt_len: int, response_ids: torch.Tensor
) -> list[float]:
    # The distribution that predicts the token at absolute position p sits at
    # position p - 1, so the i-th response token is scored from prompt_len + i - 1.
    scored = []
    for i, token_id in enumerate(response_ids.tolist()):
        scored.append(float(logprobs[0, prompt_len + i - 1, token_id]))
    return scored


class TransformersModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if load_in_4bit and not self.device.startswith("cuda"):
            raise ValueError("4-bit quantization requires a CUDA device")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=_resolve_dtype(dtype, self.device)
            )
            self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_config(cls, config) -> "TransformersModel":
        return cls(
            config.model_name,
            device=config.device,
            dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            load_in_4bit=getattr(config, "load_in_4bit", False),
        )

    def _encode_prompt(self, messages: Sequence[Message]) -> torch.Tensor:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(
                payload, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            # Plain transcript fallback for base models without a chat template.
            text = "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
            ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    @torch.no_grad()
    def generate(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> Generation:
        prompt_ids = self._encode_prompt(messages)
        output = self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,        # greedy decoding is the temperature-0 setting
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        generated = output.sequences[0, prompt_ids.shape[1]:]
        logprobs = []
        for step_scores, token_id in zip(output.scores, generated):
            step_logprobs = torch.log_softmax(step_scores[0].float(), dim=-1)
            logprobs.append(float(step_logprobs[token_id]))
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(text=text, token_ids=generated.tolist(), token_logprobs=logprobs)

    @torch.no_grad()
    def score(self, messages: Sequence[Message], response: str) -> Scoring:
        prompt_ids = self._encode_prompt(messages)
        response_ids = self.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        logits = self.model(input_ids).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        scored = _gather_response_logprobs(logprobs, prompt_ids.shape[1], response_ids[0])
        ids = response_ids[0].tolist()
        return Scoring(
            tokens=self.tokenizer.convert_ids_to_tokens(ids),
            token_ids=ids,
            logprobs=scored,
        )
