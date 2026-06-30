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


def _response_logprobs(logits: torch.Tensor, prompt_len: int, response_ids: list) -> list[float]:
    # The distribution that predicts the token at absolute position p sits at position
    # p - 1, so the response tokens are predicted by the window of positions
    # [prompt_len - 1, prompt_len - 1 + len(response)). Softmax is taken over that
    # window only, not the whole sequence, to keep memory bounded for long contexts.
    window = logits[0, prompt_len - 1 : prompt_len - 1 + len(response_ids), :].float()
    logprobs = torch.log_softmax(window, dim=-1)
    return [float(logprobs[i, token_id]) for i, token_id in enumerate(response_ids)]


class TransformersModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
        device_map: str | None = None,
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
            # Pin the 4-bit model to one GPU. A <=14B model in nf4 is ~8.5GB and fits a single
            # 16GB card; splitting it across two with device_map="auto" makes the loader
            # materialize shards in fp16 during dispatch and overflow a T4's memory.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization, device_map={"": 0}
            )
            self.device = self.model.get_input_embeddings().weight.device
        elif device_map is not None:
            # Shard an unquantized model across all visible GPUs (e.g. two T4s) so a 7B in
            # fp16 -- ~15GB, too big for one 16GB card with room to generate -- fits. The 4-bit
            # caveat above does not apply: no nf4 dispatch, accelerate just places fp16 shards,
            # and inputs go to whichever device holds the embeddings. Cap each card below its
            # full size so the loader balances the shards and leaves room for the KV cache,
            # rather than filling the first GPU and OOM-ing on the forward pass.
            n_gpus = torch.cuda.device_count()
            max_memory = {i: "11GiB" for i in range(n_gpus)} if n_gpus > 1 else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=_resolve_dtype(dtype, "cuda"),
                device_map=device_map,
                max_memory=max_memory,
            )
            self.device = self.model.get_input_embeddings().weight.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=_resolve_dtype(dtype, self.device)
            )
            self.model.to(self.device)
        self.model.eval()
        # Greedy decoding ignores the sampling parameters some models ship in their
        # generation config; clearing them silences a per-call warning without any change
        # in behaviour.
        for attribute in ("temperature", "top_p", "top_k"):
            if getattr(self.model.generation_config, attribute, None) is not None:
                setattr(self.model.generation_config, attribute, None)
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
        # apply_chat_template returns a bare tensor on older transformers and a
        # BatchEncoding on newer ones; reduce both to the input-id tensor.
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
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
        budget = max_new_tokens or self.max_new_tokens
        truncated = len(generated) >= budget and int(generated[-1]) != self.tokenizer.eos_token_id
        return Generation(
            text=text, token_ids=generated.tolist(), token_logprobs=logprobs, truncated=truncated
        )

    def _prompt_text(self, messages: Sequence[Message]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(payload, add_generation_prompt=True, tokenize=False)
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    @torch.no_grad()
    def score(self, messages: Sequence[Message], response: str) -> Scoring:
        # Tokenize the prompt and the prompt+response jointly, so the scored tokens are the
        # ones the model actually sees in context — the suffix past the shared prefix.
        # Tokenizing the response in isolation can split the BPE seam differently and score
        # a token string the model would never generate.
        prompt_text = self._prompt_text(messages)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(prompt_text + response, add_special_tokens=False).input_ids
        prompt_len = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            prompt_len += 1
        response_ids = full_ids[prompt_len:]
        if not response_ids:
            return Scoring(tokens=[], token_ids=[], logprobs=[])
        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids).logits
        scored = _response_logprobs(logits, prompt_len, response_ids)
        return Scoring(
            tokens=self.tokenizer.convert_ids_to_tokens(response_ids),
            token_ids=response_ids,
            logprobs=scored,
        )
