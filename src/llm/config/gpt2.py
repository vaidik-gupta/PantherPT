from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024       # max sequence length / positional embedding rows
    n_embd: int = 768       # d_model
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1

    @property
    def d_head(self):
        return self.n_embd // self.n_head
