# -*- coding: utf-8 -*-
"""CFE-PPAR demo (paper Sec. 3): run a plain-trained ViViT on a PLAINTEXT
clip and the domain-adaptation VT (KDDA) on its CFE-ENCRYPTED counterpart,
and check that they produce the SAME prediction.

  plaintext.avi  --[VivitForVideoClassification]------> logits_plain
  encrypted.avi  --[DomainAdaptationViViT]------------> logits_enc
  compare predictions (top-1 / top-5) and logit difference

Uses the pretrained Kinetics-400 checkpoint (no fine-tuning needed for the
equivalence check). Normalization is mean=std=0.5.

Precision: the KDDA kernel transform cancels the CFE pixel transform, so in
float64 the two logit vectors are bit-for-bit equal (max diff ~1e-6). The
deployed system recognizes encrypted (perturbed) videos, where matching the
PREDICTION is what matters, not bit-exact logits -- so this script defaults
to float32 (faster, GPU-friendly), under which predictions still agree while
the logit gap widens to ~1e-3. Use --dtype float64 to see the exact
equivalence. GPU is used automatically when available (float64 is often slow
on consumer GPUs; float32 is the realistic deployment setting).
"""

import argparse
import copy
import time

import numpy as np
import torch
from transformers import VivitForVideoClassification

import video_io
from compression_friendly_encryption import SubBlockLevelCipher
from key_dependent_domain_adaptation import DomainAdaptationViViT


HF_NAME = "google/vivit-b-16x2-kinetics400"

# Secondary sanity check on the raw logits. float64 cancels exactly (~1e-6);
# float32 rounding widens the gap to ~1e-3 without changing the prediction.
LOGIT_TOLERANCE = {torch.float64: 1e-4, torch.float32: 5e-2}


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def clip_to_normalized_tensor(frames):
    """(T,H,W,C) uint8 RGB -> (1,T,C,H,W) with mean=std=0.5 norm.
    HF ViViT's pixel_values are (batch, num_frames, channels, H, W).

    The mean=std=0.5 choice is what makes KDDA exact: it maps `255 - x` to
    exactly `-x`, turning negative-positive inversion into a pure sign flip."""
    x = (frames.astype(np.float64) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0)


def load_clip_tensor(path, num_frames, frame_size, device, dtype):
    """Read, preprocess and normalize one clip onto the target device."""
    frames = video_io.read_clip(path, num_frames, frame_size)
    return clip_to_normalized_tensor(frames).to(device=device, dtype=dtype)


def class_name_of(model, idx):
    """Class name from the model's own config.id2label (authoritative).
    This checkpoint ships without a Kinetics-400 label map, so names usually
    read as LABEL_<id>; the ids are what the comparison actually uses."""
    return model.config.id2label.get(int(idx), f"id {int(idx)}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def sync(device):
    """Block until queued GPU work is done, so wall-clock timing measures
    actual compute. CUDA kernels are launched asynchronously; without this
    the timer would only capture launch overhead. No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(fn, runs, device):
    """Mean/std wall-clock seconds over `runs` calls. Assumes steady state:
    the caller must have run `fn` (or its model) once already, so that lazy
    caches -- notably KDDA's E_hat -- are populated. sync() before and after
    each call so GPU timing is accurate."""
    ts = []
    for _ in range(runs):
        sync(device)
        t = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t)
    return float(np.mean(ts)), float(np.std(ts))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_prediction_section(model, plain_top5, enc_top5):
    """Section 1: do the two paths agree on the predicted action?"""
    same_top1 = plain_top5[0] == enc_top5[0]
    same_top5 = plain_top5 == enc_top5
    print("\n1) PREDICTION  (does the encrypted-domain VT predict the same "
          "action?)")
    print(f"   plaintext  top-1: {class_name_of(model, plain_top5[0])}  "
          f"(id {plain_top5[0]})")
    print(f"   encrypted  top-1: {class_name_of(model, enc_top5[0])}  "
          f"(id {enc_top5[0]})")
    print("   top-5 ranking (plaintext  vs  encrypted):")
    for r, (pt, et) in enumerate(zip(plain_top5, enc_top5)):
        mark = "==" if pt == et else "!="
        print(f"      {r + 1}. {class_name_of(model, pt):26s} {mark} "
              f"{class_name_of(model, et)}")
    print(f"   -> top-1 match: {same_top1}   top-5 order match: {same_top5}")


def print_numerical_section(diff, dtype, dtype_name, logit_tol):
    """Section 2: how far apart are the raw logits, and is that expected?"""
    print("\n2) NUMERICAL AGREEMENT  (how close are the raw logits?)")
    print(f"   max |logit difference|: {diff:.2e}   "
          f"(tolerance for {dtype_name}: {logit_tol:.0e})")
    if dtype == torch.float64:
        print("   float64: KDDA cancels CFE exactly -> bit-for-bit equal.")
    else:
        print("   float32: rounding only; the prediction is unaffected. "
              "Run --dtype float64 for exact equivalence.")


def print_timing_section(timing_runs, plain, enc, first_encrypted_s):
    """Section 3: what running on ciphertext costs. `plain`/`enc` are
    (mean, std) in seconds; `first_encrypted_s` is the first encrypted run,
    which also paid for building E_hat."""
    plain_mean, plain_std = plain
    enc_mean, enc_std = enc
    overhead_ms = (enc_mean - plain_mean) * 1e3
    ratio = enc_mean / plain_mean if plain_mean > 0 else float("nan")
    print("\n3) INFERENCE COST  (wall-clock, averaged over "
          f"{timing_runs} steady-state runs)")
    print(f"   plaintext ViViT                 : "
          f"{plain_mean * 1e3:8.1f} ms  +/- {plain_std * 1e3:.1f}")
    print(f"   encrypted-domain DA-ViViT       : "
          f"{enc_mean * 1e3:8.1f} ms  +/- {enc_std * 1e3:.1f}")
    print(f"   extra cost of running on cipher : "
          f"{overhead_ms:+8.1f} ms  "
          f"({(ratio - 1) * 100:+.1f}%, {ratio:.2f}x)")
    print(f"   one-time key setup (E_hat build): "
          f"{first_encrypted_s * 1e3:8.1f} ms  "
          f"(first run only, once per video under the one-time key policy)")
    print("   note: the per-inference overhead is the KDDA embedding "
          "(kernel\n"
          "         permutation + token remap); the encoder cost is "
          "unchanged.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Check that encrypted-domain inference (KDDA) matches "
                    "plaintext inference. --seed and --variant must match "
                    "those used by compression_friendly_encryption.py.")
    p.add_argument("--plain_video", default="samples/plain_1.avi",
                   help="plaintext clip")
    p.add_argument("--encrypted_video", default=None,
                   help="default: samples/encrypted_<variant>.avi "
                        "(produced by compression_friendly_encryption.py)")
    p.add_argument("--seed", type=int, default=1,
                   help="encryption seed (must match the one used by "
                        "compression_friendly_encryption.py)")
    p.add_argument("--variant", default="V2", choices=["V1", "V2"],
                   help="key scheme used for encryption (must match); V2 is "
                        "the default everywhere in this repo")
    p.add_argument("--main_block_size", type=int, default=16)
    p.add_argument("--cube_temporal_len", type=int, default=2,
                   help="frames per cube token (ViViT-B/16x2 uses 2)")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--timing_runs", type=int, default=3,
                   help="steady-state timing runs per model (after warm-up)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float64"],
                   help="float32 (default): realistic deployment precision, "
                        "GPU-friendly, predictions agree. float64: exact "
                        "bit-for-bit logit equivalence, often slow on GPU.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="auto uses CUDA when available, else CPU")
    return p


def resolve_device(name):
    """'auto' -> CUDA when available, else CPU; otherwise the named device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    a = build_arg_parser().parse_args()

    device = resolve_device(a.device)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    encrypted_video = (a.encrypted_video
                       or f"samples/encrypted_{a.variant}.avi")

    base = VivitForVideoClassification.from_pretrained(HF_NAME) \
        .eval().to(device=device, dtype=dtype)

    plain = load_clip_tensor(a.plain_video, a.num_frames, a.frame_size,
                             device, dtype)
    enc = load_clip_tensor(encrypted_video, a.num_frames, a.frame_size,
                           device, dtype)

    # DomainAdaptationViViT replaces the projection IN PLACE, so give it its
    # own copy; otherwise base's plaintext path would also use the
    # key-dependent embedding.
    keyed = DomainAdaptationViViT(copy.deepcopy(base), SubBlockLevelCipher(),
                                  a.main_block_size,
                                  a.cube_temporal_len).eval().to(device)
    keyed.set_encryption_keys(a.seed, a.variant)

    def run_plaintext_inference():
        with torch.no_grad():
            return base(plain).logits

    def run_encrypted_inference():
        with torch.no_grad():
            return keyed(enc).logits

    # First encrypted run is timed separately: KeyDependentCubeEmbedding
    # builds E_hat lazily on the first forward, and under the one-time key
    # policy (paper Sec. 4.5) this setup is paid once per video.
    logits_plain = run_plaintext_inference()          # warms up the plain path
    sync(device)
    t0 = time.perf_counter()
    logits_enc = run_encrypted_inference()            # includes E_hat build
    sync(device)
    first_encrypted_s = time.perf_counter() - t0

    plain_timing = benchmark(run_plaintext_inference, a.timing_runs, device)
    enc_timing = benchmark(run_encrypted_inference, a.timing_runs, device)

    diff = (logits_plain - logits_enc).abs().max().item()
    p_top = logits_plain[0].topk(5).indices.tolist()
    e_top = logits_enc[0].topk(5).indices.tolist()

    # The verdict is driven by prediction agreement, with the dtype-appropriate
    # logit tolerance as a secondary sanity check.
    logit_tol = LOGIT_TOLERANCE[dtype]
    consistent = p_top == e_top and diff < logit_tol

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"CFE-PPAR consistency check   "
          f"[device={device.type}  dtype={a.dtype}  variant={a.variant}]")
    print(bar)

    print_prediction_section(base, p_top, e_top)
    print_numerical_section(diff, dtype, a.dtype, logit_tol)
    print_timing_section(a.timing_runs, plain_timing, enc_timing,
                         first_encrypted_s)

    print(f"\n{bar}")
    print(f"VERDICT: {'CONSISTENT' if consistent else 'INCONSISTENT'}  "
          f"(encrypted-domain inference "
          f"{'matches' if consistent else 'does NOT match'} plaintext)")
    print(bar)


if __name__ == "__main__":
    main()
