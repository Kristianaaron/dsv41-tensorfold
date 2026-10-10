# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando). Added in this fork; see NOTICE, "Further changes in this fork".
"""Token embeddings in the display carve-out (cebu, 2026-10-07; TF_DS_DISPLAY_EMBED=1).

A headless GB10 can register a 1.75 GiB DRM dumb buffer as device memory outside the kernel's MemAvailable
(cebu-profiler overlay ``dsv41_display.py`` over coolbho3k's display_kv.c; needs /dev/dri/card0 and the display
driver capabilities in the container). The embedding table (vocab x dim bf16, 1.23 GiB, held whole on every rank) is
only ever gathered a few rows a step, so the slower span costs nothing measurable and the host gains its bytes back.
Same bits: the table is copied, never converted. If the pool is unavailable the table stays where it is.
"""

from __future__ import annotations

import os
import sys

import torch


def relocate_embed(w) -> None:
    if os.environ.get("TF_DS_DISPLAY_EMBED", "0") in ("0", ""):
        return
    sys.path.insert(0, os.environ.get("DSV41_DISPLAY_DIR", "/opt/dsv41"))
    try:
        import dsv41_display as disp
    except ImportError as exc:
        print(f"[tensorfold] display embed: module unavailable ({exc}); embeddings stay in ordinary memory", flush=True)
        return
    old = w.embed
    nbytes = old.numel() * old.element_size()
    view = disp.carve(nbytes)
    if view is None:
        print("[tensorfold] display embed: no pool space; embeddings stay in ordinary memory", flush=True)
        return
    new = view.view(old.dtype).view(old.shape)
    new.copy_(old)
    torch.cuda.synchronize()
    if not torch.equal(new, old):
        raise RuntimeError("display embed: copy mismatch")
    w.embed = new
    del old
    torch.cuda.empty_cache()
    print(f"[tensorfold] display embed: {tuple(new.shape)} {new.dtype} ({nbytes / 2**30:.2f} GiB) in the display pool; "
          f"MemAvailable {disp._memavail():.2f} GiB", flush=True)
