# Removal replay: do her named states change what she does?

*Partial run: stopped at 449 of 708 calls at Becca's request. Word-level comparison needs the run to complete; the route and ledger numbers below stand as measured.*

Turns with both arms: **75**; calls parsed: 449; errors: 0

## 1. Route (fast reply vs full deliberation)

Turns where the majority route differs between arms: **12 of 75** (16%)

| states shown | states hidden | turns |
|---|---|---|
| FAST_RESPONSE | FAST_RESPONSE | 34 |
| FULL_DELIBERATION | FULL_DELIBERATION | 29 |
| FULL_DELIBERATION | FAST_RESPONSE | 7 |
| FAST_RESPONSE | FULL_DELIBERATION | 5 |

## 2. Words (similarity of replies; higher = more alike)

Texts not available yet (the replay writes its artifact at the end); rerun this report when it finishes.

## 3. Ledger (transitions she authored at the speaking stage)

| arm | calls | transitions | per call |
|---|---|---|---|
| states shown | 225 | 21 | 0.09 |
| states hidden | 224 | 3 | 0.01 |

## 4. Fidelity to the live reply (similarity to what she said at the time)

| arm | mean | median |
|---|---|---|
| states shown | 0.173 | 0.020 |
| states hidden | 0.176 | 0.028 |

Reading: the live reply was produced with states shown, so the shown arm should track it more closely if states matter; near-zero fidelity in both arms means the recorded reply came from a different path (a follow-through or fallback) and this measure is uninformative for that turn.

