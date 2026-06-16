import torch

from lineup.backends.transformers_backend import _gather_response_logprobs


def test_gather_uses_the_predicting_position():
    logprobs = torch.full((1, 4, 3), -9.0)
    logprobs[0, 1, 1] = -0.5      # predicts response token 0 (absolute position 2)
    logprobs[0, 2, 2] = -0.2      # predicts response token 1 (absolute position 3)
    scored = _gather_response_logprobs(logprobs, prompt_len=2, response_ids=torch.tensor([1, 2]))
    assert scored == [-0.5, -0.2]
