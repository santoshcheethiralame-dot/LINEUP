import pytest

torch = pytest.importorskip("torch")

from lineup.backends.transformers_backend import _response_logprobs


def test_response_logprobs_reads_the_predicting_position():
    logits = torch.full((1, 4, 5), -1e9)
    logits[0, 1, 3] = 0.0   # predicts response token 0 (absolute position 2)
    logits[0, 2, 4] = 0.0   # predicts response token 1 (absolute position 3)
    out = _response_logprobs(logits, prompt_len=2, response_ids=[3, 4])
    assert abs(out[0]) < 1e-3
    assert abs(out[1]) < 1e-3
