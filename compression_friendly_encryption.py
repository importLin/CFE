# -*- coding: utf-8 -*-
"""CFE-PPAR: Compression-Friendly Encryption (CFE) -- the client side.

Following the paper (Sec. 3.2), a resized clip is encrypted per frame:
  A.1  divide the frame into main-blocks (MBs) of Hp x Wp (= the VT patch),
  A.2  subdivide each MB into sub-blocks (SBs) of Hs x Ws,
  A.3  apply five SB-level transformations with key K_ST: rotation, flipping,
       negative-positive inversion, RGB-channel shuffling, SB scrambling,
  A.4  scramble MB positions within the frame with key K_MS.

Every transformation stays *inside* one main-block (A.3) or moves whole blocks
without touching their pixels (A.4). Local pixel correlation therefore survives
encryption, which is what makes the cipher compression-friendly.

Key assignment (Sec. 4.1): V1 transforms all MBs with the SAME key; V2 gives
each MB a DIFFERENT key. One-time key policy: keys are constant within a
video and unique across videos. Both K_ST and K_MS are derived from --seed.

Every step above is a permutation or a sign flip, so the cipher is exactly
invertible: `decrypt_frame` / `decrypt_video_with_cfe` recover the plaintext
for whoever holds the keys. Recognition itself never uses them -- the adapted
video transformer reads the ciphertext directly (see
key_dependent_domain_adaptation.py).

Layout convention: video frames are handled as (H, W, C) uint8 RGB, but every
block-level routine below works in (C, H, W) so that channel shuffling is a
plain row selection. `encrypt_main_block` is where the two meet.
"""

import argparse
from typing import NamedTuple

import numpy as np

import video_io


# ---------------------------------------------------------------------------
# Secret keys
# ---------------------------------------------------------------------------

def _mix(*vals):
    """Fold `vals` into one deterministic, non-negative 31-bit integer.

    A Boost-`hash_combine`-style avalanche around the golden-ratio constant
    0x9E3779B97F4A7C15: each value is xor-folded into the running state
    together with shifted copies of that state, so changing any input bit
    changes the output unpredictably. The result is masked to 31 bits because
    it is consumed as a seed by `np.random.default_rng`.

    This is a fast, reproducible mixer, NOT a cryptographic key-derivation
    function -- see "Security Notes" in the README.
    """
    h = np.uint64(0x9E3779B97F4A7C15)
    for v in vals:
        h ^= np.uint64(v) + np.uint64(0x9E3779B9) + (h << np.uint64(6)) \
             + (h >> np.uint64(2))
    return int(h & np.uint64(0x7FFFFFFF))


def derive_secret_keys(seed):
    """One master seed -> the two secret keys of the paper: (K_ST, K_MS).
    K_ST drives the SB-level transformations (A.3); K_MS drives the
    main-block scrambling (A.4). The two salts (0x57, 0x35) only ensure the
    keys differ; they carry no other meaning."""
    return _mix(seed, 0x57), _mix(seed, 0x35)


def main_block_key(K_ST, mb_position, variant):
    """A.3 key for one main-block. V1: all MBs share K_ST; V2: per-MB keys.

    `mb_position` is the MB's index in raster order *before* A.4 scrambling,
    which is exactly what KDDA re-derives on the model side -- so encryption
    and adaptation always agree on which key belongs to which block."""
    return K_ST if variant == "V1" else _mix(K_ST, mb_position)


SUB_BLOCK_OPS = ("rot", "flip", "npt", "cs", "sbs")   # SB-level (A.3)
ALL_ENCRYPTION_OPS = SUB_BLOCK_OPS + ("mbs",)         # + MB scrambling (A.4)


# ---------------------------------------------------------------------------
# A.3 -- sub-block level transformations
# ---------------------------------------------------------------------------

class SubBlockTransformParams(NamedTuple):
    """One draw of the five A.3 parameters, for all SBs of a single MB."""
    rotation: np.ndarray             # (num_sb,)     quarter turns, 0..3
    flipping: np.ndarray             # (num_sb,)     flip code, 0..3
    np_inversion: np.ndarray         # (num_sb,)     bool, apply 255 - x
    channel_shuffle: np.ndarray      # (num_sb, C)   new channel order
    sb_scrambling_order: list        # (num_sb,)     output SB i <- input [i]


def _draw_sub_block_transform_params(rng, num_sb, channels, enabled):
    """Draw all SB-level params in a FIXED order (rng is consumed even for
    disabled ops, so the key stream stays aligned), then neutralize the ones
    turned off. `enabled` maps op name -> bool. For VISUALIZATION only:
    disabling ops lets you isolate the visual effect of each transform; it is
    NOT meant to be paired with KDDA recovery."""
    rotation = rng.integers(0, 4, size=num_sb)
    flipping = rng.integers(0, 4, size=num_sb)
    np_inversion = rng.integers(0, 2, size=num_sb).astype(bool)
    channel_shuffle = np.stack([rng.permutation(channels)
                                for _ in range(num_sb)])
    sb_scrambling_order = list(rng.permutation(num_sb))

    # Neutral value per op = the identity transform, so a disabled op is a
    # no-op while the draws above have already advanced the key stream.
    if not enabled["rot"]:
        rotation = np.zeros(num_sb, dtype=rotation.dtype)
    if not enabled["flip"]:
        flipping = np.zeros(num_sb, dtype=flipping.dtype)
    if not enabled["npt"]:
        np_inversion = np.zeros(num_sb, dtype=bool)
    if not enabled["cs"]:
        channel_shuffle = np.tile(np.arange(channels), (num_sb, 1))
    if not enabled["sbs"]:
        sb_scrambling_order = list(range(num_sb))
    return SubBlockTransformParams(rotation, flipping, np_inversion,
                                   channel_shuffle, sb_scrambling_order)


def _split_into_sub_blocks(mb_chw, sb_size):
    """A.2: cut one main-block into its sub-blocks.

    (C, Hp, Wp) -> ((num_sb, C, Hs, Ws), (ny, nx)), the SBs in raster order.
    `ny`/`nx` are the SB grid dimensions, which `_merge_sub_blocks` needs to
    reassemble the block.
    """
    channels, h, w = mb_chw.shape
    ny, nx = h // sb_size, w // sb_size
    # split each spatial axis into (grid, within-block), then bring the two
    # grid axes to the front so they can be flattened into one SB index
    grid = mb_chw.reshape(channels, ny, sb_size, nx, sb_size) \
                 .transpose(1, 3, 0, 2, 4)                # (ny, nx, C, Hs, Ws)
    return grid.reshape(ny * nx, channels, sb_size, sb_size), (ny, nx)


def _merge_sub_blocks(flat, ny, nx):
    """Inverse of `_split_into_sub_blocks`: (num_sb, C, Hs, Ws) -> (C, Hp, Wp),
    reading the SBs back in raster order over the `ny` x `nx` grid."""
    n, channels, sh, sw = flat.shape
    return flat.reshape(ny, nx, channels, sh, sw).transpose(2, 0, 3, 1, 4) \
               .reshape(channels, ny * sh, nx * sw)


class SubBlockLevelCipher:
    """The five SB-level transformations of the paper (A.3), applied inside
    one main-block: rotation (rot), flipping (flip), negative-positive
    inversion (npt), RGB-channel shuffling (cs), and sub-block scrambling
    (sbs).

    All five are either a permutation of pixel positions or a sign flip of
    pixel values. That algebraic property is what KDDA exploits: the same
    cipher can be replayed on the transformer's conv kernel instead of on the
    pixels (see `index_permutation_and_sign`).

    `enabled` selectively turns ops off FOR VISUALIZATION (e.g. inspect the
    effect of a single transform). Disabled ops still consume their RNG draw,
    so the remaining ops behave identically to full encryption. Main-block
    scrambling (mbs, A.4) is handled in encrypt_video_with_cfe, not here."""

    def __init__(self, sub_block_size=8, enabled=None):
        self.sub_block_size = sub_block_size
        self.enabled = {op: True for op in SUB_BLOCK_OPS}
        if enabled:
            self.enabled.update(enabled)

    def _apply(self, volumes, key, invert_fns):
        """Run the A.3 op sequence on one or more parallel volumes.

        All volumes are driven by a SINGLE key stream, so they undergo exactly
        the same geometric shuffle. That is what keeps the pixel cipher
        (`encrypt_main_block`) and its symbolic twin
        (`index_permutation_and_sign`) in lockstep -- the latter pushes an
        index volume and a sign volume through this very code path.

        volumes:     list of (C, Hp, Wp) arrays, all of the same shape.
        invert_fns:  one callable per volume, applied wherever NP inversion is
                     drawn (255 - x for pixels, identity for indices, negation
                     for signs).
        Returns the transformed volumes, in the input order.
        """
        splits = [_split_into_sub_blocks(v, self.sub_block_size)
                  for v in volumes]
        sub_blocks = [flat.copy() for flat, _ in splits]     # writable views
        ny, nx = splits[0][1]
        num_sb, channels = sub_blocks[0].shape[:2]
        params = _draw_sub_block_transform_params(
            np.random.default_rng(key), num_sb, channels, self.enabled)

        # ops 1-4 act within a sub-block, identically across all volumes
        for i in range(num_sb):
            for v, flat in enumerate(sub_blocks):
                sb = flat[i]                                 # (C, Hs, Ws)
                if params.rotation[i]:                                # rot
                    sb = np.rot90(sb, k=int(params.rotation[i]), axes=(1, 2))
                flip_code = int(params.flipping[i])                   # flip
                if flip_code == 1:
                    sb = sb[:, :, ::-1]                      # horizontal
                elif flip_code == 2:
                    sb = sb[:, ::-1, :]                      # vertical
                elif flip_code == 3:
                    sb = sb[:, ::-1, ::-1]                   # both
                if params.np_inversion[i]:                            # npt
                    sb = invert_fns[v](sb)
                flat[i] = sb[list(params.channel_shuffle[i]), :, :]   # cs

        # op 5 (sbs) moves whole sub-blocks to new positions in the MB
        order = list(params.sb_scrambling_order)
        return [_merge_sub_blocks(flat[order], ny, nx) for flat in sub_blocks]

    def encrypt_main_block(self, main_block, key):
        """A.3 on real pixels. main_block: (Hp, Wp, C) uint8 -> encrypted
        (Hp, Wp, C) uint8. Computed in int64 so that `255 - x` cannot wrap
        around before the result is cast back to uint8."""
        mb = np.ascontiguousarray(
            main_block.transpose(2, 0, 1)).astype(np.int64)   # -> (C, Hp, Wp)
        out, = self._apply([mb], key, [lambda s: 255 - s])
        return out.transpose(1, 2, 0).astype(np.uint8)

    def index_permutation_and_sign(self, key, H, W, C):
        """The cipher expressed as a per-element map on the flattened HWC
        volume: y[j] = sign[j] * x[perm[j]] (+255 where sign = -1). This is
        the bridge to KDDA (B.2): the same map is applied to the conv kernel.

        It is obtained by running two bookkeeping volumes through the very
        same `_apply` path as the pixels: a volume of position indices (which
        records where each element travelled) and a volume of +1s (which
        records how many times NP inversion negated it). Because normalization
        uses mean = std = 0.5, the `+255` offset of `255 - x` vanishes in the
        normalized domain and only `sign` remains -- so the map is exactly
        linear for KDDA's purposes.
        """
        idx = np.arange(H * W * C).reshape(H, W, C).transpose(2, 0, 1)
        sgn = np.ones((C, H, W), dtype=np.int64)
        out_idx, out_sgn = self._apply(
            [np.ascontiguousarray(idx), sgn], key,
            [lambda s: s, lambda s: -s])   # indices ride along; signs negate
        return (out_idx.transpose(1, 2, 0).reshape(-1),
                out_sgn.transpose(1, 2, 0).reshape(-1))

    def decrypt_main_block(self, main_block, key):
        """Inverse of `encrypt_main_block` -- for the holder of the keys.

        Reuses the element map above. Encryption is
        `y[j] = x[perm[j]]` where `sign[j] = +1` and `y[j] = 255 - x[perm[j]]`
        where `sign[j] = -1`, so undoing the inversion and scattering the
        result back through `perm` restores the plaintext exactly. Each of the
        five A.3 transforms is a permutation or a sign flip, which is what
        makes the cipher invertible in the first place."""
        H, W, C = main_block.shape
        perm, sign = self.index_permutation_and_sign(key, H, W, C)
        ciphertext = main_block.reshape(-1).astype(np.int64)
        plaintext = np.empty_like(ciphertext)
        plaintext[perm] = np.where(sign == 1, ciphertext, 255 - ciphertext)
        return plaintext.reshape(H, W, C).astype(np.uint8)


# ---------------------------------------------------------------------------
# A.1-A.4 -- the complete cipher
# ---------------------------------------------------------------------------

def _frame_cipher_setup(frame, seed, variant, main_block_size, enabled):
    """Shared A.1/A.2/A.4 setup for `encrypt_frame` and `decrypt_frame`:
    check the geometry, build the SB cipher, derive the keys and lay out the
    MB scrambling. Returns (cipher, K_ST, mb_scrambling_order, grid_w)."""
    assert variant in ("V1", "V2")
    enabled_ops = {op: True for op in ALL_ENCRYPTION_OPS}
    if enabled:
        enabled_ops.update(enabled)

    H, W, _ = frame.shape
    assert H % main_block_size == 0 and W % main_block_size == 0, \
        "frame size must be a multiple of the main-block size"

    # A.1: the MB grid covering the frame
    grid_h, grid_w = H // main_block_size, W // main_block_size
    num_mb = grid_h * grid_w

    cipher = SubBlockLevelCipher(                     # A.2 + A.3
        enabled={op: enabled_ops[op] for op in SUB_BLOCK_OPS})
    K_ST, K_MS = derive_secret_keys(seed)

    # A.4: a permutation of MB positions; identity when mbs is disabled.
    # Read as: the block at source position s moves to mb_scrambling_order[s].
    mb_scrambling_order = (np.random.default_rng(K_MS).permutation(num_mb)
                           if enabled_ops["mbs"] else np.arange(num_mb))
    return cipher, K_ST, mb_scrambling_order, grid_w


def encrypt_frame(frame, seed, variant="V2", main_block_size=16,
                  enabled=None):
    """CFE (A.1-A.4) on ONE frame: (H, W, C) uint8 RGB -> encrypted frame.

    This is the whole cipher; `encrypt_video_with_cfe` simply applies it to
    every frame of a clip with the same keys. It is public because the
    visualization notebook walks through the method frame by frame.

    `main_block_size` = Hp (= Wp). `enabled` maps op name
    (rot/flip/npt/cs/sbs/mbs) -> bool; default all on = full encryption.
    Turning ops off is a VISUALIZATION aid to isolate each transform's effect
    and is not meant for KDDA recovery."""
    cipher, K_ST, mb_scrambling_order, grid_w = _frame_cipher_setup(
        frame, seed, variant, main_block_size, enabled)

    out = frame.copy()
    bs = main_block_size
    for src, dst in enumerate(mb_scrambling_order):
        src_y, src_x = divmod(src, grid_w)
        dst_y, dst_x = divmod(int(dst), grid_w)
        block = frame[src_y * bs:(src_y + 1) * bs,
                      src_x * bs:(src_x + 1) * bs]
        encrypted = cipher.encrypt_main_block(                         # A.3
            block, main_block_key(K_ST, src, variant))
        out[dst_y * bs:(dst_y + 1) * bs,                               # A.4
            dst_x * bs:(dst_x + 1) * bs] = encrypted
    return out


def decrypt_frame(frame, seed, variant="V2", main_block_size=16,
                  enabled=None):
    """Inverse of `encrypt_frame`: recover the plaintext frame from a
    ciphertext frame, given the same keys.

    Note that CFE-PPAR never needs this to recognize a video: the server runs
    the KDDA-adapted transformer directly on the ciphertext and never sees the
    plaintext (Sec. 3.3). Decryption is for the party holding the keys, and it
    doubles as a check that the cipher really is lossless.

    Exact only while the ciphertext is intact. After lossy compression
    (Motion-JPEG / H.264, Sec. 4) it returns an approximation, because the
    codec has already perturbed the pixels this function un-permutes.

    Same body as encryption with the roles of the two block positions swapped:
    encryption reads position `src` and writes `dst`, decryption reads `dst`
    and writes `src`."""
    cipher, K_ST, mb_scrambling_order, grid_w = _frame_cipher_setup(
        frame, seed, variant, main_block_size, enabled)

    out = frame.copy()
    bs = main_block_size
    for src, dst in enumerate(mb_scrambling_order):
        src_y, src_x = divmod(src, grid_w)
        dst_y, dst_x = divmod(int(dst), grid_w)
        block = frame[dst_y * bs:(dst_y + 1) * bs,               # undo A.4
                      dst_x * bs:(dst_x + 1) * bs]
        decrypted = cipher.decrypt_main_block(                   # undo A.3
            block, main_block_key(K_ST, src, variant))
        out[src_y * bs:(src_y + 1) * bs,
            src_x * bs:(src_x + 1) * bs] = decrypted
    return out


def encrypt_video_with_cfe(video_in, video_out, seed, variant="V2",
                           main_block_size=16, num_frames=32, frame_size=224,
                           enabled=None):
    """CFE on a whole clip: read and preprocess to `num_frames` x
    `frame_size`, encrypt every frame with `encrypt_frame`, write losslessly.

    All frames share the same keys -- that is the one-time key policy (keys
    constant within a video, unique across videos), and it is also what lets
    KDDA use a single set of transformed parameters for the entire clip."""
    _map_clip(video_in, video_out, "CFE", variant, num_frames, frame_size,
              lambda frame: encrypt_frame(frame, seed, variant,
                                          main_block_size, enabled))


def decrypt_video_with_cfe(video_in, video_out, seed, variant="V2",
                           main_block_size=16, num_frames=32, frame_size=224,
                           enabled=None):
    """Inverse of `encrypt_video_with_cfe`: turn a CFE-encrypted clip back
    into its plaintext with the same keys. Round-trips exactly on a losslessly
    stored ciphertext; see `decrypt_frame` for the caveats."""
    _map_clip(video_in, video_out, "CFE inverse", variant, num_frames,
              frame_size,
              lambda frame: decrypt_frame(frame, seed, variant,
                                          main_block_size, enabled))


def _map_clip(video_in, video_out, tag, variant, num_frames, frame_size,
              transform):
    """Read and preprocess a clip, apply `transform` to every frame, write the
    result back losslessly. Shared by the encrypt and decrypt entry points."""
    frames = video_io.read_clip(video_in, num_frames, frame_size)
    out = np.stack([transform(frame) for frame in frames])

    fps = video_io.probe(video_in)["fps"]
    video_io.write_video(video_out, out, fps=fps)
    print(f"[{tag}] {video_in} -> {video_out}  "
          f"({variant}, {len(out)} frames)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Encrypt a video with CFE (paper Sec. 3.2), or invert the "
                    "cipher with --decrypt. The same --seed and --variant "
                    "must later be given to verify_consistency.py.")
    p.add_argument("--decrypt", action="store_true",
                   help="invert CFE instead of applying it: recover the "
                        "plaintext clip using the same --seed/--variant. Not "
                        "needed for recognition -- the adapted VT reads the "
                        "ciphertext directly.")
    p.add_argument("--input", default=None,
                   help="default: samples/plain_1.avi, or "
                        "samples/encrypted_<variant>.avi with --decrypt")
    p.add_argument("--output", default=None,
                   help="default: samples/encrypted_<variant>.avi, or "
                        "samples/decrypted_<variant>.avi with --decrypt")
    p.add_argument("--seed", type=int, default=1,
                   help="per-video secret; K_ST/K_MS are derived from it")
    p.add_argument("--variant", default="V2", choices=["V1", "V2"],
                   help="V1: all MBs share one key; V2: per-MB keys")
    p.add_argument("--main_block_size", type=int, default=16,
                   help="Hp = Wp, must match the VT patch size")
    p.add_argument("--num_frames", type=int, default=32,
                   help="frames uniformly sampled from the whole video")
    p.add_argument("--frame_size", type=int, default=224,
                   help="output resolution; must be a multiple of "
                        "--main_block_size")
    p.add_argument("--disable", nargs="*", default=[],
                   choices=list(ALL_ENCRYPTION_OPS),
                   help="ops to turn OFF for visualization "
                        "(rot/flip/npt/cs/sbs/mbs); default: none (full enc)")
    a = p.parse_args()
    enabled = {op: op not in a.disable for op in ALL_ENCRYPTION_OPS}
    run = decrypt_video_with_cfe if a.decrypt else encrypt_video_with_cfe
    if a.decrypt:
        video_in = a.input or f"samples/encrypted_{a.variant}.avi"
        video_out = a.output or f"samples/decrypted_{a.variant}.avi"
    else:
        video_in = a.input or "samples/plain_1.avi"
        video_out = a.output or f"samples/encrypted_{a.variant}.avi"
    run(video_in, video_out, a.seed, a.variant, a.main_block_size,
        a.num_frames, a.frame_size, enabled=enabled)
