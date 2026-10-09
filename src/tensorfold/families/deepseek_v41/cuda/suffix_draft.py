"""Context suffix drafts beside DSpark (cebu mini-project 5, 2026-10-07; TF_DS_SUFFIX_DRAFT=1).

Code edits, JSON, tool output and quoted text repeat runs that already sit in the stream's own context. When the
tokens just before the next position (the committed tail plus the pending token) occurred earlier in the context, the
tokens that followed them then are a draft with a high acceptance rate, and runs of repeated text also reuse the same
routed experts, which keeps a verify window's expert union small. Verification is unchanged, so the output is the same
bits with or without this drafter; only the drafts proposed for verification change.

Per stream: an index from each N-token window of the committed context to the position just after its most recent
occurrence, extended incrementally as tokens commit (constant work a round, also at 262k context). A proposal needs the
exact match to extend at least ``SUFFIX_MIN`` tokens back; it replaces DSpark's drafts for that round at the same
length (so verify windows keep the shapes the decode graphs were captured for), and only when it is at least as long.
"""

from __future__ import annotations

import os

N = int(os.environ.get("TF_DS_SUFFIX_N", "4"))                 # tokens in the index key
# exact match length a proposal needs. 2026-10-07: at 6, same-length replacement of DSpark's drafts cost 5-6% on code
# edits (87.1 -> 81.6-83.1 tok/s; DSpark already lands ~5.8 of 6 there) and 12% on JSON: only long repeats now, and
# they go deeper than DSpark's block instead of replacing it (the verify window's free rows, TF_DS_DECODE_ROWS).
SUFFIX_MIN = int(os.environ.get("TF_DS_SUFFIX_MIN", "16"))
MAX_BACK = 64                                                  # how far back the match length is counted
ON = os.environ.get("TF_DS_SUFFIX_DRAFT", "0") == "1"
# TF_DS_SUFFIX_DEEP=1 (default): drafts past DSpark's block into the window's free rows. Off for a day (2026-10-08): on
# the feature-edit prompt its reply differed from serial decoding (974 vs 943 tokens). Cause: the bf16 projections
# (head, compressor wkv / wgate) ran on cuBLAS, whose kernel depends on the row count; with native_fp8.rowinv_linear
# serial, DSpark and deep-suffix replies are byte-identical (eval/tf/det-check-rowinv), so it is back on.
DEEP = os.environ.get("TF_DS_SUFFIX_DEEP", "1") == "1"


class SuffixIndex:
    """The context index of one stream: N-token window -> position just after its latest occurrence."""

    def __init__(self) -> None:
        self.table: dict[tuple, int] = {}
        self.built = 0                                         # windows ending before this position are indexed

    def update(self, host: list, committed: int) -> None:
        if committed < self.built:                             # the stream was rewound (should not happen): rebuild
            self.table.clear()
            self.built = 0
        for end in range(max(self.built, N), committed + 1):
            self.table[tuple(host[end - N:end])] = end
        self.built = committed + 1

    def propose(self, host: list, committed: int, pending: int, k: int) -> tuple[list, int]:
        """(draft tokens, exact match length) for the position after ``pending``; ([], 0) when nothing matches."""

        if k <= 0 or committed < N:
            return [], 0
        tail = tuple(host[committed - N + 1:committed]) + (pending,)
        pos = self.table.get(tail)
        if pos is None or pos >= committed:
            return [], 0
        # the match length: how far the earlier occurrence agrees with the context before the tail
        m, a, b = N, pos - N - 1, committed - N
        while m < MAX_BACK and a >= 0 and b >= 0 and host[a] == host[b]:
            m, a, b = m + 1, a - 1, b - 1
        return [int(t) for t in host[pos:min(pos + k, committed)]], m


def apply(live, proposed: dict, want: dict, room: int = 0) -> int:
    """Replace a drafting stream's DSpark drafts with a context suffix draft where the match is long enough, extended
    past DSpark's length by up to ``room`` rows (the verify window's rows no stream uses this round). Returns the number
    of streams that took a suffix draft this round."""

    if not ON:
        return 0
    took = 0
    for s in live:
        k = want.get(s.sid, 0)
        if k <= 0:
            continue
        sc = s.st.sc
        idx = s.__dict__.get("_sfx")
        if idx is None:
            idx = s._sfx = SuffixIndex()
        idx.update(sc.host, sc.length)
        dsp = proposed.get(s.sid, [])
        base = len(dsp) or k
        sd, m = idx.propose(sc.host, sc.length, s.pending, base + (max(0, room) if DEEP else 0))
        if m >= SUFFIX_MIN and sd and len(sd) >= len(dsp):
            room -= max(0, len(sd) - len(dsp))
            proposed[s.sid] = sd
            took += 1
    return took
