import torch


## Static for a whole batch of sequences
class KVState:
    """Key-Value state for past key/value tensors during autoregressive generation.

    Layout: k, v are [B, n_layers, n_head, seq, d_head]; the sequence dim (3) grows over
    time. All layers advance in lockstep -- the model collects k_new/v_new from every layer
    and appends them in one combined update per step.
    """

    def __init__(self, B, n_layers, n_head, n_context, d_head, device):
        self.n_context = n_context
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_head = d_head
        self.k = torch.zeros(B, n_layers, n_head, 0, d_head, device=device)
        self.v = torch.zeros(B, n_layers, n_head, 0, d_head, device=device)

    def update(self, k: torch.Tensor, v: torch.Tensor):
        # k, v: [B, n_layers, n_head, T, d_head] -- new K/V for ALL layers, appended together.
        self.k = torch.cat([self.k, k], dim=3)   # dim 3 = sequence
        self.v = torch.cat([self.v, v], dim=3)
        return self.k, self.v

    def read(self, layer: int):
        return self.k[:, layer], self.v[:, layer]

    def seq_len(self) -> int:
        return self.k.shape[3]

    def clear(self):
        B, n_layers, n_head, _, d_head = self.k.shape
        self.k = torch.zeros(B, n_layers, n_head, 0, d_head, device=self.k.device)
        self.v = torch.zeros_like(self.k)
