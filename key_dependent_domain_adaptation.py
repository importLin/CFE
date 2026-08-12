# -*- coding: utf-8 -*-
"""CFE-PPAR: Key-Dependent Domain Adaptation (KDDA) -- the model side.

Following the paper (Sec. 3.3), a VT trained on plain videos is adapted to the
encrypted domain by transforming its cube embedding layer with the SAME keys
used in CFE:
  B.1  subdivide the spatial components of the 3D conv kernel E into SBs,
  B.2  apply the five SB-level transformations of A.3 to E with key K_ST,
  B.3  undo the MB scrambling of A.4 with key K_MS.

Every SB-level transformation is a permutation or a sign flip, so B.2 reduces
to a column permutation + sign on E (E_hat = sign * E[:, perm]); B.3 is
realized as a token position remap inside the embedding layer, which pairs
each token with its original position and is therefore equivalent to
rearranging the positional embeddings E_pos. With mean = std = 0.5
normalization, NP inversion becomes a pure sign flip, so cube tokens from the
encrypted video equal plaintext tokens exactly -- no fine-tuning involved.

Index layout: ViViT's cube embedding is a Conv3d with kernel (t, Hp, Wp), so
flattening its weight gives columns ordered as (C, t, h, w). CFE instead works
on (H, W, C) frames. Translating between those two layouts is the only real
bookkeeping in this file, and it lives in
`cube_kernel_permutation_and_sign`.
"""

import numpy as np
import torch
from torch import nn

from compression_friendly_encryption import derive_secret_keys, main_block_key


def cube_kernel_permutation_and_sign(cipher, key, mb_size, in_channels,
                                     cube_temporal_len):
    """B.2 for one main-block: express its cipher as (perm, sign) over the
    kernel's input dimension D_in, so that E_hat = sign * E[:, perm].

    The cipher acts on a single (Hp, Wp, C) frame block, while a cube spans
    `cube_temporal_len` frames. CFE uses the same key for every frame of a
    video, so the same spatial map is simply replicated across the cube's
    temporal slots -- the loop below writes it once per slot, converting from
    the cipher's HWC layout to Conv3d's (C, t, h, w) flatten order.
    """
    D_in = in_channels * cube_temporal_len * mb_size * mb_size
    # conv_ids[c, t, y, x] = that element's column index in the flattened E
    conv_ids = np.arange(D_in).reshape(in_channels, cube_temporal_len,
                                       mb_size, mb_size)
    perm = np.empty(D_in, dtype=np.int64)
    sign = np.empty(D_in, dtype=np.int64)

    # The cipher as an element map on one frame block: encrypted position j
    # holds plaintext element src_hwc[j], multiplied by sgn_hwc[j].
    src_hwc, sgn_hwc = cipher.index_permutation_and_sign(
        key, mb_size, mb_size, in_channels)

    def to_conv_order(flat_hwc):
        """(Hp*Wp*C,) laid out as (H, W, C) -> (C*Hp*Wp,) as (C, h, w)."""
        return flat_hwc.reshape(mb_size, mb_size, in_channels) \
                       .transpose(2, 0, 1).reshape(-1)

    for ts in range(cube_temporal_len):
        slot_ids = conv_ids[:, ts]                     # (C, Hp, Wp)
        # gather in HWC order: for each encrypted position, the column index
        # of the plaintext element that CFE moved into it
        src_ids = slot_ids.transpose(1, 2, 0).reshape(-1)[src_hwc]
        # scatter both back into the kernel's own column ordering
        perm[slot_ids.reshape(-1)] = to_conv_order(src_ids)
        sign[slot_ids.reshape(-1)] = to_conv_order(sgn_hwc)
    return perm, sign


class KeyDependentCubeEmbedding(nn.Module):
    """KDDA-transformed cube embedding layer (drop-in for ViViT's Conv3d).
    B.2: per-MB transformed kernels E_hat from K_ST (one shared kernel under
    V1, per-MB kernels under V2). B.3: token remap undoing the K_MS
    scrambling.

    The layer is a matrix multiply rather than a convolution: ViViT's cube
    embedding is non-overlapping (stride = kernel size), so extracting the
    cubes and multiplying by E is exactly equivalent -- and it is the form in
    which a *different* kernel can be applied per MB position, which V2 needs.
    """

    def __init__(self, original_conv, cipher, main_block_size=16,
                 cube_temporal_len=2):
        super().__init__()
        self.cipher = cipher
        self.mb_size = main_block_size
        self.cube_len = cube_temporal_len
        w = original_conv.weight.detach()   # (D, C, t, Hp, Wp)
        self.embed_dim, self.in_channels = w.shape[0], w.shape[1]
        self.register_buffer("E", w.flatten(1).clone())     # (D, D_in)
        self.register_buffer("bias", original_conv.bias.detach().clone())
        self._seed, self._variant = None, "V2"
        # Built lazily on the first forward (the MB count depends on the input
        # resolution) and reused afterwards -- the one-time key setup cost.
        self._transformed = None            # (E_hat, mb_scrambling_order)

    def set_encryption_keys(self, seed, variant="V2"):
        """Point the layer at one video's keys; invalidates the cached E_hat
        so the next forward rebuilds it. Call once per video (one-time key
        policy)."""
        self._seed, self._variant = seed, variant
        self._transformed = None

    def _build_transformed_parameters(self, num_mb, device):
        """B.2 + B.3: derive the transformed kernels and the MB scrambling
        order for a frame holding `num_mb` main-blocks.

        Under V1 a single kernel serves every position; under V2 there is one
        kernel per MB, keyed by its pre-scrambling position."""
        K_ST, K_MS = derive_secret_keys(self._seed)
        # same permutation the encryption side drew from K_MS (A.4)
        mb_scrambling_order = torch.as_tensor(
            np.random.default_rng(K_MS).permutation(num_mb), device=device)

        D_in = self.E.shape[1]
        n_kernels = 1 if self._variant == "V1" else num_mb
        perms = np.empty((n_kernels, D_in), np.int64)
        signs = np.empty((n_kernels, D_in), np.int64)
        for s in range(n_kernels):
            perms[s], signs[s] = cube_kernel_permutation_and_sign(
                self.cipher, main_block_key(K_ST, s, self._variant),
                self.mb_size, self.in_channels, self.cube_len)

        # E[:, perms] -> (D, n_kernels, D_in); scale by sign, then put the
        # kernel axis first so E_hat[s] is the kernel for MB position s
        E_hat = (self.E[:, torch.as_tensor(perms, device=device)]
                 * torch.as_tensor(signs, dtype=self.E.dtype,
                                   device=device).unsqueeze(0)
                 ).permute(1, 0, 2)                    # (n_kernels, D, D_in)
        self._transformed = (E_hat, mb_scrambling_order)
        return self._transformed

    def forward(self, x):
        # accept (B,C,T,H,W) or (B,T,C,H,W)
        assert x.shape[0] == 1, "batch=1"
        if x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1, 3, 4)
        C, T, H, W = x.shape[1:]
        bs, t = self.mb_size, self.cube_len
        num_cubes, grid_h, grid_w = T // t, H // bs, W // bs
        num_mb = grid_h * grid_w

        # cut the clip into cubes and flatten each one into a D_in vector,
        # ordered (C, t, h, w) to match the columns of E
        cube_vectors = x[0].reshape(C, num_cubes, t, grid_h, bs, grid_w, bs) \
                           .permute(1, 3, 5, 0, 2, 4, 6) \
                           .reshape(num_cubes, num_mb, -1)
        E_hat, mb_scrambling_order = (self._transformed or
                                      self._build_transformed_parameters(
                                          num_mb, x.device))
        # The MB at source position s was A.3-encrypted with
        # main_block_key(s) and then A.4-moved to mb_scrambling_order[s];
        # recover its token by applying kernel s at position
        # mb_scrambling_order[s].
        x_src = cube_vectors[:, mb_scrambling_order]        # (G, N, D_in)
        if self._variant == "V1":
            out = x_src @ E_hat[0].T + self.bias            # one shared kernel
        else:
            out = torch.einsum("gni,ndi->gnd", x_src, E_hat) + self.bias

        # back to Conv3d's output layout: (B, D, G, grid_h, grid_w). Tokens now
        # sit at their PLAINTEXT positions, so ViViT's positional embeddings
        # apply unchanged -- this is B.3.
        return out.reshape(1, num_cubes, grid_h, grid_w, self.embed_dim) \
                  .permute(0, 4, 1, 2, 3)


class DomainAdaptationViViT(nn.Module):
    """Domain-adaptation VT (paper Sec. 3.3): a plain-trained ViViT whose cube
    embedding layer is replaced by the KDDA-transformed
    KeyDependentCubeEmbedding, so it recognizes CFE-encrypted videos directly.
    The adaptation is key-driven and exact -- no fine-tuning. Call
    set_encryption_keys(seed, variant) per video (one-time key policy).

    NOTE: the replacement mutates `hf_vivit` IN PLACE. Pass a copy if the same
    checkpoint is also needed for plaintext inference."""

    def __init__(self, hf_vivit, cipher, main_block_size=16,
                 cube_temporal_len=2):
        super().__init__()
        self.model = hf_vivit
        emb = self.model.vivit.embeddings.patch_embeddings
        self.kdda_embedding = KeyDependentCubeEmbedding(
            emb.projection, cipher, main_block_size, cube_temporal_len)
        emb.projection = self.kdda_embedding
        assert emb.projection is self.kdda_embedding, \
            "projection replacement failed"

    def set_encryption_keys(self, seed, variant="V2"):
        """Forward one video's keys to the embedding layer."""
        self.kdda_embedding.set_encryption_keys(seed, variant)

    def forward(self, x):
        return self.model(x)
