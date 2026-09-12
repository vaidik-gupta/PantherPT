"""The from-scratch GPT-2 (`implemented`) and the reference GPT-2 (`pretrained`)
must expose identical state_dicts, so weights are interchangeable between them
(and with HuggingFace via `pretrained.GPT2.from_pretrained`).
"""

import pytest
import torch

from src.llm.config.gpt2 import GPT2Config
from src.llm.implemented.gpt2 import GPT2 as ImplementedGPT2
from src.llm.pretrained.gpt2 import GPT2 as PretrainedGPT2


@pytest.fixture
def config():
    # Tiny config: fast to build, still exercises every parameter tensor.
    return GPT2Config(vocab_size=16, n_ctx=8, n_embd=8, n_layer=2, n_head=2)


def test_state_dict_keys_and_order_match(config):
    impl = ImplementedGPT2(config).state_dict()
    pre = PretrainedGPT2(config).state_dict()
    assert list(impl.keys()) == list(pre.keys())


def test_state_dict_shapes_match(config):
    impl = ImplementedGPT2(config).state_dict()
    pre = PretrainedGPT2(config).state_dict()
    for key in impl:
        assert impl[key].shape == pre[key].shape, key


def test_weights_load_across_implementations(config):
    """A state_dict from one model must load cleanly into the other."""
    impl = ImplementedGPT2(config)
    pre = PretrainedGPT2(config)
    missing, unexpected = pre.load_state_dict(impl.state_dict(), strict=False)
    assert missing == []
    assert unexpected == []


def test_lm_head_tied_to_token_embedding(config):
    """Both models tie the LM head to the token embedding weight."""
    for model in (ImplementedGPT2(config), PretrainedGPT2(config)):
        sd = model.state_dict()
        assert torch.equal(sd["lm_head.weight"], sd["wte.weight"])
