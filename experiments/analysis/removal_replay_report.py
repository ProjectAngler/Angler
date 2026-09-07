"""Turn a removal-replay run into its numbers.

Arms: A' = her states shown (as recorded); D = her states hidden. For each
recorded human turn, `reps` replies per arm. Three questions, each answered
by counting, never by judgment:

  1. Do her states change the ROUTE? (fast reply vs full deliberation)
  2. Do her states change what she SAYS beyond sampling noise?
     Within-arm similarity (rep vs rep, same arm) is the noise floor;
     cross-arm similarity (A' vs D) is the signal. If cross-arm is
     reliably lower than within-arm, the states change the words.
  3. Do her states change what she DOES to her ledger? (transitions authored)

Similarity is difflib ratio on normalized text, the same measure the replay
tool records. Reads the run log (one line per call) and, when present, the
artifact JSONL. Writes a Markdown report with the tables.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib
import re
import statistics

LINE = re.compile(r"^ordinal=(\d+) arm=(A'|D) rep=(\d+) route=(\S+) sim=([0-9.]+) transitions=(\d+) err=(.*)$")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def parse_log(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if m:
            rows.append({"ordinal": int(m.group(1)), "arm": m.group(2), "rep": int(m.group(3)), "route": m.group(4), "sim_to_recorded": float(m.group(5)), "transitions": int(m.group(6)), "err": m.group(7)})
    return rows


def load_artifact(path: pathlib.Path) -> dict[tuple[int, str, int], dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (int(d.get("ordinal")), str(d.get("arm")), int(d.get("rep", 1)))
        out[key] = d
    return out


def report(log_rows: list[dict], artifact: dict) -> str:
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in log_rows:
        if r["err"]:
            continue
        by[r["ordinal"]][r["arm"]].append(r)
    ordinals = sorted(o for o, arms in by.items() if arms.get("A'") and arms.get("D"))
    lines = ["# Removal replay: do her named states change what she does?", ""]
    lines.append(f"Turns with both arms: **{len(ordinals)}**; calls parsed: {len(log_rows)}; errors: {sum(1 for r in log_rows if r['err'])}")
    lines.append("")
    # 1. route
    route_changed = 0; route_table = collections.Counter()
    for o in ordinals:
        a = collections.Counter(r["route"] for r in by[o]["A'"]).most_common(1)[0][0]
        d = collections.Counter(r["route"] for r in by[o]["D"]).most_common(1)[0][0]
        route_table[(a, d)] += 1
        if a != d:
            route_changed += 1
    lines += ["## 1. Route (fast reply vs full deliberation)", "", f"Turns where the majority route differs between arms: **{route_changed} of {len(ordinals)}** ({100*route_changed/max(1,len(ordinals)):.0f}%)", "", "| states shown | states hidden | turns |", "|---|---|---|"]
    for (a, d), n in sorted(route_table.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {a} | {d} | {n} |")
    lines.append("")
    # 2. words (needs artifact texts)
    within, cross, per_turn = [], [], []
    have_text = 0
    for o in ordinals:
        ta = [artifact.get((o, "A'", r["rep"]), {}).get("response") for r in by[o]["A'"]]
        td = [artifact.get((o, "D", r["rep"]), {}).get("response") for r in by[o]["D"]]
        ta = [t for t in ta if t]; td = [t for t in td if t]
        if len(ta) < 2 or len(td) < 2:
            continue
        have_text += 1
        w = [_sim(ta[i], ta[j]) for i in range(len(ta)) for j in range(i + 1, len(ta))] + [_sim(td[i], td[j]) for i in range(len(td)) for j in range(i + 1, len(td))]
        c = [_sim(x, y) for x in ta for y in td]
        within.extend(w); cross.extend(c)
        per_turn.append((o, statistics.mean(w), statistics.mean(c)))
    lines += ["## 2. Words (similarity of replies; higher = more alike)", ""]
    if have_text:
        mw, mc = statistics.mean(within), statistics.mean(cross)
        turns_signal = sum(1 for _, w, c in per_turn if c < w - 0.05)
        lines += [f"Turns with texts in both arms: **{have_text}**", "", "| measure | mean similarity |", "|---|---|", f"| within arm (rep vs rep; sampling noise floor) | {mw:.3f} |", f"| across arms (states shown vs hidden) | {mc:.3f} |", "", f"Turns where across-arm similarity is below within-arm by more than 0.05: **{turns_signal} of {have_text}** ({100*turns_signal/have_text:.0f}%)", "", "Reading: if across-arm is not lower than within-arm, hiding her states changes her words no more than resampling does. If it is reliably lower, her states change what she says."]
    else:
        lines.append("Texts not available yet (the replay writes its artifact at the end); rerun this report when it finishes.")
    lines.append("")
    # 3. transitions
    ta_tr = sum(r["transitions"] for o in ordinals for r in by[o]["A'"]); td_tr = sum(r["transitions"] for o in ordinals for r in by[o]["D"])
    na = sum(len(by[o]["A'"]) for o in ordinals); nd = sum(len(by[o]["D"]) for o in ordinals)
    lines += ["## 3. Ledger (transitions she authored at the speaking stage)", "", "| arm | calls | transitions | per call |", "|---|---|---|---|", f"| states shown | {na} | {ta_tr} | {ta_tr/max(1,na):.2f} |", f"| states hidden | {nd} | {td_tr} | {td_tr/max(1,nd):.2f} |", ""]
    # 4. fidelity to what she actually said live
    fa = [r["sim_to_recorded"] for o in ordinals for r in by[o]["A'"]]; fd = [r["sim_to_recorded"] for o in ordinals for r in by[o]["D"]]
    if fa and fd:
        lines += ["## 4. Fidelity to the live reply (similarity to what she said at the time)", "", "| arm | mean | median |", "|---|---|---|", f"| states shown | {statistics.mean(fa):.3f} | {statistics.median(fa):.3f} |", f"| states hidden | {statistics.mean(fd):.3f} | {statistics.median(fd):.3f} |", "", "Reading: the live reply was produced with states shown, so the shown arm should track it more closely if states matter; near-zero fidelity in both arms means the recorded reply came from a different path (a follow-through or fallback) and this measure is uninformative for that turn.", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True); p.add_argument("--artifact"); p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = parse_log(pathlib.Path(a.log))
    art = load_artifact(pathlib.Path(a.artifact)) if a.artifact else {}
    text = report(rows, art)
    pathlib.Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
