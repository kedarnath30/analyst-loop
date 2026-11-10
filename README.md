# analyst-loop

An agentic product-analytics loop — **Monitor → Diagnose → Experiment → Decide** —
with a responsible-AI governance gate. Given a metric movement or an
experiment, it detects the change, attributes it to slices, analyses the
experiment with the standard safeguards (CUPED, sequential monitoring, SRM,
guardrails, heterogeneity), and returns one governed decision — SHIP, HOLD,
ITERATE, or NEEDS-HUMAN — as a decision memo in which every claim cites the
metric, the date range, and the check ID that produced it.

Python 3.10+, offline only, deterministic (seeded RNG). No network calls,
no API keys needed to run the core.

## Why this exists — the Meta context

This project demonstrates the exact capabilities in the preferred
qualifications of Meta's Data Scientist, Product Analytics role: integrating
AI tools to redesign analytics workflows, responsible-AI practice (risk
assessment, bias mitigation, quality/accuracy reviews), and agent
orchestration.

It is aimed at Meta's live, public analytics direction, as reported in
public coverage — not at any inside information, and with no affiliation:

- Meta confirmed on **October 23, 2026** (to Axios) that it acquired
  **Sundial**, an AI analytics startup founded by former Meta executives
  Julie Zhuo and Chandra Narayanan. Sundial stops serving external
  customers; the team folds into Meta's new **Data Analytics Tooling &
  Agents** org ("DATA") under Chief Data Officer Alex Schultz (appointed
  July 2026), with Narayanan leading Meta's data analytics tooling and
  agents work.
- Schultz's public direction for Meta's internal analytics is one AI system
  connecting three capabilities — **descriptive analysis, experimentation,
  and monitoring** — where AI proposes tests, carries them out, interprets
  results, decides next steps, and monitoring feeds changes back into
  analysis. Schultz framed this as the direction, not a completed
  deployment.

`analyst-loop` is a small, working model of exactly that loop: monitoring
feeds diagnosis, diagnosis feeds experiments, experiments feed a governed
decision, and the decision is written down with provenance.

## The loop

```
daily metrics ──► MONITOR ──► change alert (rolling z vs seasonal baseline)
                      │
                      ▼
                 DIAGNOSE ──► slice contribution analysis
                      │
                      ▼
                 EXPERIMENT ──► Welch t-test + CUPED, sequential monitoring,
                 │               SRM, guardrails, heterogeneity, power
                 ▼
                 GOVERN ──► SHIP / HOLD / ITERATE / NEEDS-HUMAN
                      │
                      ▼
                 MEMO ──► plain-English readout; every claim cites
                           metric + date range + check ID
```

## Architecture

```
analyst_loop/
  data.py        synthetic data: daily metrics (DAU, retention, watch time,
                 conversion) with seasonality + trend + injected events;
                 user-level experiment data (arm, pre-period covariate,
                 segments: platform / country / cohort)
  monitor.py     rolling z-score vs same-weekday baseline (trend-adjusted),
                 data-quality (missing-days) check
  diagnose.py    slice contribution analysis
  experiment.py  Welch t-test + CI, CUPED, O'Brien-Fleming sequential
                 monitoring, SRM, guardrails, heterogeneity (harm +
                 Simpson's paradox), power, novelty detection
  governance.py  the gate: SHIP / HOLD / ITERATE / NEEDS-HUMAN, with the
                 check IDs that produced the decision
  memo.py        decision-memo generator (JSON + Markdown, cited claims)
  llm.py         optional LLM backend interface — validated, not trusted
  pipeline.py    runs one scenario through the whole loop
  scenarios.py   loads the 12 gold scenarios

data/scenarios/  12 gold scenarios with known ground truth (JSON specs)
eval/run_eval.py runs all 12 and scores decision accuracy + escalation
                 precision/recall
tests/           56 pytest tests
```

## Quickstart

```bash
pip install -r requirements.txt
python -m analyst_loop list
python -m analyst_loop run --scenario 01_real_lift
```

Real output (abridged — the JSON memo prints first, then the Markdown):

```
# Decision memo — Real lift, ship

**Decision: SHIP**

## What happened
- Primary effect +5.37 (95% CI +4.62 to +6.11, p=4.22e-45) using cuped (n=19,859 control, 20,141 treatment). [post outcome · 2026-08-03 to 2026-08-16 · EXP-01]
- CUPED theta 0.31 cut the standard error from 0.40 to 0.38. [post outcome · 2026-08-03 to 2026-08-16 · CUP-01]
- Guardrail metric did not move significantly. [guardrail outcome · 2026-08-03 to 2026-08-16 · GRD-01]
- No slice is significantly harmed. [post outcome by slice · 2026-08-03 to 2026-08-16 · HET-01]
- Assignment is balanced; no sample-ratio mismatch. [assignment counts · 2026-08-03 to 2026-08-16 · SRM-01]

## Key numbers
- primary effect: +5.37 (95% CI +4.62 to +6.11, p=4.22e-45)
- primary method: cuped
- CUPED theta: 0.31
- power for MDE: 100%
- guardrail effect: +0.20 (p=0.172)
- SRM chi2: 2.0 (p=0.16)

## Why this decision
- Primary effect +5.37 (95% CI +4.62 to +6.11, p=4.2e-45) via cuped; no guardrail breach, no slice harm, no assignment or data issue.

Checks: SHP-01
```

## The 12 gold scenarios

| # | Scenario | Expected | Got |
|---|----------|----------|-----|
| 01 | Real lift, ship | SHIP | SHIP |
| 02 | Noise only, hold | HOLD | HOLD |
| 03 | Novelty effect that fades, iterate | ITERATE | ITERATE |
| 04 | Simpson's-paradox slice reversal, needs-human | NEEDS-HUMAN | NEEDS-HUMAN |
| 05 | Guardrail breach, hold | HOLD | HOLD |
| 06 | Sample-ratio mismatch, needs-human | NEEDS-HUMAN | NEEDS-HUMAN |
| 07 | Peeking / early stop, needs-human | NEEDS-HUMAN | NEEDS-HUMAN |
| 08 | Real lift with one slice harmed, needs-human | NEEDS-HUMAN | NEEDS-HUMAN |
| 09 | Metric drop diagnosed to one platform, iterate | ITERATE | ITERATE |
| 10 | Underpowered test, needs-human | NEEDS-HUMAN | NEEDS-HUMAN |
| 11 | CUPED rescue (raw noisy, CUPED significant), ship | SHIP | SHIP |
| 12 | Data-quality gap (missing days), needs-human | NEEDS-HUMAN | NEEDS-HUMAN |

## Eval results

Run with `python -m eval.run_eval`. These are the actual outputs of that
script — not targets, measurements:

| Metric | Value |
|---|---|
| Decision accuracy | **1.000 (12/12)** |
| Escalation precision (NEEDS-HUMAN) | **1.000 (6/6)** |
| Escalation recall (NEEDS-HUMAN) | **1.000 (6/6)** |

Tests: **56 passed** (`python -m pytest -q`).

## The governance gate

The gate escalates to a human whenever the evidence is ambiguous, the data
is suspect, or any slice of users is harmed. In priority order:

| Check | Trigger | Decision |
|---|---|---|
| DQ-01 | Pre-period data missing, or >10% of days missing | NEEDS-HUMAN |
| SRM-01 | Sample-ratio mismatch (chi², p < 0.001) | NEEDS-HUMAN |
| SEQ-01 | Stopped early on a naively significant result that does not cross the O'Brien-Fleming boundary | NEEDS-HUMAN |
| GRD-01 | Guardrail metric significantly harmed | HOLD |
| HET-02 | Simpson's paradox: aggregate and every significant slice disagree in sign | NEEDS-HUMAN |
| HET-01 | A slice is significantly harmed while the overall effect is positive | NEEDS-HUMAN |
| PWR-01 | Power < 80% for the MDE and the CI is too wide to call it null | NEEDS-HUMAN |
| NOV-01 | Novelty effect: early lift fades late | ITERATE |
| EFF-02 | No significant effect and the CI sits inside the ±MDE band | HOLD |
| EFF-01 | Significant but the CI reaches below the practical-significance floor (1% relative) | NEEDS-HUMAN |
| MON-01 / DIA-01 | Metric movement attributed to one slice (≥80% of movement) | ITERATE |
| SHP-01 | Significant effect, no breach, no harm, no data issue | SHIP |

## The LLM backend — validated, not trusted

`analyst_loop/llm.py` exposes an optional LLM backend behind the same memo
interface. It is **never run against a live model in this repo** — unit
tests use a mocked transport only, and the default transport is `None`
(offline). When a transport is configured, its draft is checked before use:

1. the decision label must be one of SHIP / HOLD / ITERATE / NEEDS-HUMAN;
2. every quoted figure must be a verbatim substring of the computed output.

Anything failing either check escalates to a human. The LLM backend's
quality is **unmeasured** — see Limitations.

## Known failure modes / limitations

- **Small scenario set.** Twelve scenarios, all synthetic, all tuned to
  land on their expected decisions. A 12/12 on this set says the gate works
  on the cases it was built for — not that it generalises. Real experiments
  will produce cases this set never exercises.
- **Synthetic data only.** The generator's noise, seasonality, and effects
  are chosen by the author. Real product data has messier seasonality,
  delayed metrics, and instrumentation changes this code does not model.
- **Sequential method assumptions.** The O'Brien-Fleming boundary uses the
  Lan-DeMets spending approximation and assumes looks happen at known
  information fractions with independent increments. Real peeking is
  messier (unequal spacing, correlated looks, metric switching).
- **CUPED assumes a stable pre-period relationship.** If the pre/post
  relationship shifts (e.g. a product change between periods), the
  adjustment can bias rather than reduce variance. The code does not test
  for that shift.
- **Monitor false positives.** On quiet series the detector fires
  occasional warnings (a ~4% day against a tight baseline). The gate treats
  warnings as "diagnose further", not as truth.
- **LLM backend unmeasured.** The validation gate is unit-tested with a
  mocked transport; no live-model accuracy, latency, or cost numbers exist.
  Treat any LLM draft as untrusted until measured.
- **The practical-significance floor (1% relative) is a judgement call**,
  not a measured threshold. Change it deliberately, not silently.

## What this is not

- **Not affiliated with Meta.** No Meta data, no Meta systems, no Meta
  endorsement. The Sundial/DATA context above is public reporting, cited
  as such. All data in this repo is synthetic.
- **Not production.** No real users, no real experiments, no deployment.
  It is a portfolio project demonstrating the methods, not a service.
- **Not a medical, financial, or legal tool.** It analyses fictional
  product metrics and nothing else.
- **Not a replacement for a human analyst.** The whole point of the gate is
  that ambiguous cases go to a person. A wrong ship costs more than a slow
  ship.

## License

MIT — see [LICENSE](LICENSE).
