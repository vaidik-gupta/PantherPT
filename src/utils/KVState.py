import torch


## Static (preallocated) KV cache for a whole batch of sequences.
class KVState:
    """Preallocated key/value cache for autoregressive generation.

    Layout: k, v are [B, n_layers, n_head, n_context, d_head], allocated ONCE up front.
    New K/V are written in place into the [pos : pos+T] slice of the sequence dim as tokens
    are generated -- no reallocation and no per-step copy of the past. (Contrast the earlier
    torch.cat-growth approach, which rebuilt the whole cache every step: O(N^2) copies over a
    generation of N tokens, a transient ~2x peak, and allocator fragmentation.)
    """

    def __init__(self, B, n_layers, n_head, n_context, d_head, device):
        self.n_context = n_context
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_head = d_head
        self.k = torch.zeros(B, n_layers, n_head, n_context, d_head, device=device)
        self.v = torch.zeros(B, n_layers, n_head, n_context, d_head, device=device)
        self.pos = 0   # number of tokens currently cached (the write cursor)

    def write(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Write this layer's new K/V at the current cursor; return the valid prefix.

        k_new, v_new: [B, n_head, T, d_head]. The returned k, v are views onto the buffer
        covering positions [0 : pos+T] (past + just-written) -- no copy of the past.
        """
        T = k_new.shape[2]
        end = self.pos + T
        self.k[:, layer, :, self.pos:end] = k_new
        self.v[:, layer, :, self.pos:end] = v_new
        return self.k[:, layer, :, :end], self.v[:, layer, :, :end]

    def advance(self, T: int):
        # Move the cursor forward once per generation step, after all layers have written.
        self.pos += T

    def read(self, layer: int):
        return self.k[:, layer, :, :self.pos], self.v[:, layer, :, :self.pos]

    def seq_len(self) -> int:
        return self.pos

    def clear(self):
        self.k.zero_()
        self.v.zero_()
        self.pos = 0
