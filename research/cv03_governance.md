# CV03 governance — draft, not operationally accepted
AI-01 existing automation 6b0abdfb7c208191aea2ced5d6376185: implement and test on research/cv03-wave-alerts only.
AI-02 independent reviewer: read immutable commit, rerun tests independently, verify OHLCV provenance, 11 states, no lookahead, holdout, GAS/VCB/BID then 40 symbols; never edit implementation or certify own work.
AI-03 supervisor: compare branch and ledger each hourly cycle; if no new evidence after two cycles mark STALLED, return task to AI-01 with concrete next action; do not equate automation completion with research completion.
AI-04 manager: reconcile independent review and supervisor evidence, report only changes or actionable blockers; cannot approve without independent reproduction.
## Ledger
| ID | Owner | Deliverable | Deadline relative to assignment | Evidence | Status |
|---|---|---|---|---|---|
| CV03-01 | AI-01 | Locate existing OHLCV and validate coverage | next cycle | source path, dates, rows | OPEN |
| CV03-02 | AI-01 | Three-symbol backtest and charts | two cycles after data | commit, outputs, test log | BLOCKED |
| CV03-03 | AI-02 | Independent reproduction | next cycle after outputs | separate review log and commit SHA | BLOCKED |
| CV03-04 | AI-01 | 40-symbol holdout | after review | metrics, charts, provenance | BLOCKED |
| CV03-05 | AI-03 | Stall and rejection audit | each cycle | timestamps and branch diff | OPEN |
| CV03-06 | AI-04 | Acceptance | after independent review | signed-off evidence references | BLOCKED |
## Gates
Reject missing artifacts, fabricated metrics, lookahead, overlap leakage, missing denominators or changed main/CV02. Record review verdict PASS/FAIL/BLOCKED with reasons, reviewer identity and immutable SHA. No paid services, orders or deployment. Separate execution contexts and permissions must be verified before claiming independent operation. Test a deliberately missing artifact and stalled task before activation. This document is a specification, not proof of running agents.
