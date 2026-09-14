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


def test_forward_predicts_same_token_per_position(config):
    """With shared weights and dropout disabled, both models produce the same
    logits -- and thus the same argmax token -- at every position.

    No shared RNG seed is required: the forward pass is deterministic in eval
    mode. `manual_seed` here only fixes the random weights/input so the test
    itself is reproducible, not to synchronise the two models.
    """
    torch.manual_seed(0)
    impl = ImplementedGPT2(config).eval()
    pre = PretrainedGPT2(config).eval()
    pre.load_state_dict(impl.state_dict())   # give both models identical weights

    idx = torch.randint(0, config.vocab_size, (2, config.n_ctx))
    impl_logits = impl(idx)
    pre_logits = pre(idx)

    # Tiny fp differences (e.g. `* (1/sqrt(d))` vs `/ sqrt(d)`) are allowed.
    torch.testing.assert_close(impl_logits, pre_logits, rtol=1e-4, atol=1e-5)
    # Token-by-token: the predicted token at each position must be identical.
    assert torch.equal(impl_logits.argmax(dim=-1), pre_logits.argmax(dim=-1))


def test_generate_emits_identical_tokens(config):
    """Greedy decoding (top_k=1) is deterministic, so both models emit the exact
    same token sequence -- WITHOUT needing a shared RNG seed.

    (Only temperature *sampling* via multinomial would require synchronising the
    global RNG before each `generate` call.)
    """
    torch.manual_seed(0)   # reproducible weights/prompt only
    impl = ImplementedGPT2(config).eval()
    pre = PretrainedGPT2(config).eval()
    pre.load_state_dict(impl.state_dict())

    prompt = torch.randint(0, config.vocab_size, (1, 2))
    impl_out = impl.generate(prompt.clone(), max_new_tokens=5, top_k=1)
    pre_out = pre.generate(prompt.clone(), max_new_tokens=5, top_k=1)

    assert torch.equal(impl_out, pre_out)
