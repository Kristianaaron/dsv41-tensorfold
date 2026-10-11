# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando). Added in this fork; see NOTICE, "Further changes in this fork".
"""Closeness to the full model at an agent's decision points, from the same served-KLD dumps as the headline KLD.

The headline KLD averages every position of the agent-session holdout. Most of those tokens are easy; a run is
derailed by a few: whether to call a tool, which one, whether to make another call, and when to stop. This scores
only those positions, teacher-forced, against the full model's stored top-20 distribution:

  act_or_answer   the tokens after </think> up to the first one that commits: the tool-call marker or the answer's
                  first word
  tool_name       every token of the tool name in <｜DSML｜ invoke name="...">; tool_name_first is its first token
  another_call    the token after </｜DSML｜ invoke>: open another invoke, or close the calls
  stop            the end-of-sentence token that ends an assistant turn

Per class: positions, mean / p90 KLD, and how often the quant's most likely token is the full model's (the decision
it would make greedily). Confidence intervals resample whole windows (cluster bootstrap), since positions in one
session are not independent.

KLD is computed exactly as dsv41_kld_compare.py does (teacher top-20 plus a tail bucket; a student logprob missing
from its own top-20 takes its tail mass spread over the unseen tokens, which can only overstate the student's KLD),
so the "all" row reproduces the headline numbers. Windows are rebuilt from the holdout manifest with the teacher's
tokenizer and must match the dumps' window_hash.

  python3 decision_kld.py --manifest holdout-manifest.json --tokenizer tokenizer.json \
      --teacher teacher_ref_b000-009.pt teacher_ref_b009-018.pt --student ours.pt mia.pt --out decisions.json
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
from pathlib import Path

VOCAB = 129_280
CLASSES = ("all", "decisions", "act_or_answer", "tool_name", "tool_name_first", "another_call", "stop")
THINK_END = "</think>"
EOS = "<｜end▁of▁sentence｜>"
INVOKE_NAME = re.compile(r'<｜DSML｜ invoke name="([^"]*)"')
INVOKE_END = re.compile(r"</｜DSML｜ invoke>\n")
TURN = re.compile(r"<｜(Assistant|User|System)｜>")


def windows_hash(windows: list[list[int]]) -> str:
    h = hashlib.sha256()
    for w in windows:
        h.update(len(w).to_bytes(4, "little"))
        for t in w:
            h.update(int(t).to_bytes(4, "little"))
    return h.hexdigest()


def decision_targets(text: str, starts: list[int]) -> dict[str, set[int]]:
    """Token indices (into the row's ids) whose prediction is a decision, by class. ``starts`` are token start chars."""

    def at(char: int) -> int | None:
        i = bisect.bisect_left(starts, char)
        return i if i < len(starts) else None

    turns = [(m.start(), m.group(1)) for m in TURN.finditer(text)]
    turn_starts = [t[0] for t in turns]

    def assistant(char: int) -> bool:
        """Only the model's own turns hold decisions; quoted markers in prompts or tool results do not."""
        k = bisect.bisect_right(turn_starts, char) - 1
        return k >= 0 and turns[k][1] == "Assistant"

    found: dict[str, set[int]] = {c: set() for c in CLASSES[2:]}
    for m in re.finditer(re.escape(THINK_END), text):
        if not assistant(m.start()):
            continue
        i = at(m.end())
        while i is not None and i < len(starts):
            found["act_or_answer"].add(i)
            piece = text[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(text)]
            if piece.strip() and piece.strip() != "<":
                break
            i += 1
    for m in INVOKE_NAME.finditer(text):
        if not assistant(m.start()):
            continue
        first = at(m.start(1))
        last = at(m.end(1))
        if first is None:
            continue
        found["tool_name_first"].add(first)
        found["tool_name"].update(range(first, max(first + 1, last if last is not None else first + 1)))
    for m in INVOKE_END.finditer(text):
        if not assistant(m.start()):
            continue
        i = at(m.end())
        if i is not None:
            found["another_call"].add(i)
    for m in re.finditer(re.escape(EOS), text):
        i = at(m.start())
        if i is not None and assistant(m.start()):
            found["stop"].add(i)
    return found


def build(manifest: dict, tokenizer_path: Path, seq_len: int):
    """Windows (as dsv41_layer_hessian.build_windows) and, per class, the (window, position) keys of decisions."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    rows = [r for r in manifest["rows"] if r.get("text")]
    lens = [int(r.get("seq_len") or 0) for r in rows]
    per_row = None if (not any(lens) or len(set(lens)) == 1) else [n or seq_len for n in lens]
    windows: list[list[int]] = []
    keys: dict[str, set[tuple[int, int]]] = {c: set() for c in CLASSES[2:]}
    names: dict[tuple[int, int], str] = {}
    for r_i, row in enumerate(rows):
        n = per_row[r_i] if per_row else seq_len
        enc = tok.encode(row["text"], add_special_tokens=False)
        ids = enc.ids
        starts = [o[0] for o in enc.offsets]
        targets = decision_targets(row["text"], starts)
        for s in range(0, len(ids) - 1, n):
            window = ids[s: s + n]
            if len(window) < n // 4:
                continue
            w = len(windows)
            windows.append(window)
            for cls, idx in targets.items():
                for g in idx:
                    p = g - s - 1  # row (w, p) predicts window[p + 1]
                    if 0 <= p < len(window) - 1:
                        keys[cls].add((w, p))
            for m in INVOKE_NAME.finditer(row["text"]):
                g = bisect.bisect_left(starts, m.start(1))
                if g in targets["tool_name_first"] and 0 <= g - s - 1 < len(window) - 1:
                    names[(w, g - s - 1)] = m.group(1)
    return windows, keys, names


def load(paths: list[Path]):
    import torch

    parts = [torch.load(p, weights_only=False) for p in paths]
    if len({(d["corpus_sha256"], d["window_hash"]) for d in parts}) != 1:
        raise SystemExit("dump parts disagree on corpus/window hash")
    cat = lambda k: torch.cat([d[k] for d in parts])  # noqa: E731
    return {"corpus_sha256": parts[0]["corpus_sha256"], "window_hash": parts[0]["window_hash"],
            "label": parts[0].get("label"), "keys": cat("keys"), "top_ids": cat("top_ids"),
            "top_logprobs": cat("top_logprobs")}


def per_position(teacher: dict, student: dict):
    """(keys, kld, top1_agree) on the joined positions, same maths as dsv41_kld_compare.compare."""
    import torch

    s_index = {tuple(k): i for i, k in enumerate(student["keys"].tolist())}
    t_rows, s_rows, keys = [], [], []
    for i, k in enumerate(teacher["keys"].tolist()):
        j = s_index.get(tuple(k))
        if j is not None:
            t_rows.append(i)
            s_rows.append(j)
            keys.append(tuple(k))
    ti = teacher["top_ids"][t_rows].long()
    tl = teacher["top_logprobs"][t_rows].double()
    si = student["top_ids"][s_rows].long()
    sl = student["top_logprobs"][s_rows].double()
    p = tl.exp()
    p_tail = (1 - p.sum(-1)).clamp_min(0)
    match = ti.unsqueeze(-1) == si.unsqueeze(-2)
    found = match.any(-1)
    s_lp = torch.where(match, sl.unsqueeze(-2), torch.full_like(sl.unsqueeze(-2), -math.inf)).amax(-1)
    s_known = sl.exp().masked_fill(si < 0, 0).sum(-1)
    s_rest = (1 - s_known).clamp_min(1e-12)
    unseen = VOCAB - (si >= 0).sum(-1)
    s_lp = torch.where(found, s_lp, (s_rest / unseen).log().unsqueeze(-1).expand_as(s_lp))
    q = s_lp.exp()
    q_tail = (1 - q.sum(-1)).clamp_min(1e-12)
    kld = (p * (tl - s_lp)).sum(-1) + p_tail * (p_tail.clamp_min(1e-12).log() - q_tail.log())
    return keys, kld.clamp_min(0), (ti[:, 0] == si[:, 0]), ti[:, 0], si[:, 0]


def summarize(kld, agree, windows_of, boot: int, seed: int) -> dict:
    import torch

    n = len(kld)
    if n == 0:
        return {"positions": 0}
    out = {"positions": n, "kld_mean": float(kld.mean()), "kld_p90": float(torch.quantile(kld.float(), 0.9)),
           "top1_agree": float(agree.double().mean()), "flips": int((~agree).sum())}
    groups: dict[int, list[int]] = {}
    for i, w in enumerate(windows_of):
        groups.setdefault(w, []).append(i)
    ids = list(groups)
    g = torch.Generator().manual_seed(seed)
    means, agrees = [], []
    for _ in range(boot):
        pick = torch.randint(len(ids), (len(ids),), generator=g).tolist()
        rows = [i for j in pick for i in groups[ids[j]]]
        if rows:
            means.append(float(kld[rows].mean()))
            agrees.append(float(agree[rows].double().mean()))
    q = lambda xs: [float(torch.tensor(xs).quantile(0.025)), float(torch.tensor(xs).quantile(0.975))]  # noqa: E731
    out["kld_mean_ci95"] = q(means)
    out["top1_agree_ci95"] = q(agrees)
    out["windows"] = len(ids)
    return out


def paired_ratio(keys: list[tuple[int, int]], a: dict, b: dict, boot: int, seed: int) -> dict:
    """Mean KLD of ``a`` over mean KLD of ``b`` on the same positions; windows resampled together."""
    import torch

    if not keys:
        return {"positions": 0}
    x = torch.tensor([a[k] for k in keys], dtype=torch.float64)
    y = torch.tensor([b[k] for k in keys], dtype=torch.float64)
    out = {"positions": len(keys), "ratio": float(x.mean() / y.mean()), "lower": int((x < y).sum())}
    groups: dict[int, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k[0], []).append(i)
    ids = list(groups)
    g = torch.Generator().manual_seed(seed)
    ratios = []
    for _ in range(boot):
        rows = [i for j in torch.randint(len(ids), (len(ids),), generator=g).tolist() for i in groups[ids[j]]]
        if rows and float(y[rows].mean()) > 0:
            ratios.append(float(x[rows].mean() / y[rows].mean()))
    if ratios:
        t = torch.tensor(ratios)
        out["ratio_ci95"] = [float(t.quantile(0.025)), float(t.quantile(0.975))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="holdout manifest (cebu.dsv41.calib_corpus)")
    ap.add_argument("--tokenizer", type=Path, required=True, help="the teacher's tokenizer.json")
    ap.add_argument("--teacher", type=Path, nargs="+", required=True, help="teacher_ref_*.pt parts")
    ap.add_argument("--student", type=Path, nargs="+", required=True, help="student .pt dumps (one per quant)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--boot", type=int, default=2000, help="cluster-bootstrap resamples (0 = none)")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    windows, keys, names = build(manifest, args.tokenizer, args.seq_len)
    teacher = load(args.teacher)
    if windows_hash(windows) != teacher["window_hash"]:
        raise SystemExit("rebuilt windows do not match the teacher's window_hash (tokenizer or manifest differs)")
    keys["decisions"] = set().union(*(keys[c] for c in ("act_or_answer", "tool_name", "another_call", "stop")))

    report = {"schema": "dsv41.decision_kld/v1", "corpus_sha256": teacher["corpus_sha256"],
              "window_hash": teacher["window_hash"], "classes": {}, "results": [], "paired": []}
    per_key: list[dict] = []
    for path in args.student:
        student = load([path])
        if student["window_hash"] != teacher["window_hash"]:
            raise SystemExit(f"{path}: window_hash differs from the teacher's")
        pos, kld, agree, t_top, s_top = per_position(teacher, student)
        index = {k: i for i, k in enumerate(pos)}
        per_key.append({k: float(kld[i]) for k, i in index.items()})
        result = {"label": student["label"] or path.stem, "classes": {}}
        for cls in CLASSES:
            rows = list(range(len(pos))) if cls == "all" else sorted(index[k] for k in keys[cls] if k in index)
            result["classes"][cls] = summarize(kld[rows], agree[rows], [pos[i][0] for i in rows], args.boot, args.seed)
        result["tool_name_first_flips"] = [
            {"key": list(pos[i]), "tool": names.get(pos[i]), "teacher_top1": int(t_top[i]), "student_top1": int(s_top[i])}
            for i in sorted(index[k] for k in keys["tool_name_first"] if k in index) if not bool(agree[i])]
        report["results"].append(result)
    report["classes"] = {c: len(keys[c]) for c in CLASSES[1:]}
    for j in range(1, len(per_key)):
        a, b = per_key[0], per_key[j]
        row = {"student": report["results"][0]["label"], "against": report["results"][j]["label"], "classes": {}}
        for cls in CLASSES[1:]:
            row["classes"][cls] = paired_ratio(sorted(k for k in keys[cls] if k in a and k in b), a, b,
                                               args.boot, args.seed)
        report["paired"].append(row)

    for r in report["results"]:
        print(r["label"])
        for cls in CLASSES:
            c = r["classes"][cls]
            if not c.get("positions"):
                print(f"  {cls:>16}  n=0")
                continue
            ci = c.get("kld_mean_ci95", [math.nan, math.nan])
            print(f"  {cls:>16}  n={c['positions']:>6}  KLD {c['kld_mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"p90 {c['kld_p90']:.4f}  same top-1 {c['top1_agree'] * 100:.1f}%  flips {c['flips']}")
    for row in report["paired"]:
        print(f"{row['student']} / {row['against']}, same positions (mean KLD ratio, cluster-bootstrap 95% CI):")
        for cls, c in row["classes"].items():
            if c.get("positions"):
                ci = c.get("ratio_ci95", [math.nan, math.nan])
                print(f"  {cls:>16}  {c['ratio']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  lower at {c['lower']}/{c['positions']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
