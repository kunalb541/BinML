"""Write docs/VERIFICATION_2026-09-11.md from the findings JSON and dispositions.py."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dispositions import ISSUES, MAP, main as mapped  # noqa: E402

lines, R = mapped()
F = R["findings"]
n_sev = {s: sum(1 for f in F if f["severity"] == s) for s in ("critical", "major", "minor")}
n_ver = {v: sum(1 for f in F if f["verdict"] == v) for v in ("real", "partly", "refuted")}
doc = f"""# Verification of the revision work, 2026-09-11

**What was checked.** Everything that will go into the A&C revision from the post-submission work: the
draft manuscript text (`paper/draft_gulls_section.tex`), the experiment ledger and section 1 of
`paper/REVISION.md`, the README / usage / CHANGELOG / audit entries, the two example notebooks, and every
code path changed since the audit fixes (27 files: `binml/gulls.py`, `pipeline/*`, `validation/gulls/*`,
the runners, the tests).

**How.** A workflow of eight independent verifiers, each re-deriving one slice from the raw data without
trusting any stated number (numbers in the draft; numbers in the ledger and docs; dataset facts from the
pinned RMDC26 tables; a fresh re-implementation of the transfer tables; the science of the relabelling;
the training experiments; code review with the full test suite in a separate worktree; a referee-style
logic read), then two adversarial checkers that tried to refute every finding and listed what nobody
had checked. Raw output: `docs/verification/2026-09-11_findings.json`.

**Result.** {len(F)} findings ({n_sev['critical']} critical, {n_sev['major']} major, {n_sev['minor']} minor);
{n_ver['real']} upheld as stated, {n_ver['partly']} upheld in part, {n_ver['refuted']} refuted. They reduce to
{len(ISSUES)} distinct issues, listed below with what was done. Several changed conclusions, not just wording:

* **RMDC26's pause schedule differs between its six high-cadence seasons.** The "exact schedule" mask used in
  the schedule experiment and in the threshold calibration matched only the first season (16% of scored
  events). Experiment 2 was rerun with the measured per-season schedule and single-factor arms, and every
  calibration was redone under the measured seasons on one common held-out pool.
* **Partial bin occupancy.** RMDC26's colour visits leave one of eight F146 epochs empty in about a third of
  the model's bins, a pattern training never contains. It cannot be imposed faithfully on binned data (doing
  so produced a spurious collapse, caught before use); on real RMDC26 inputs its effect was measured directly.
* **The colour-band gain is not anomaly signal.** It comes from planetary events without a detectable F146
  anomaly; its mechanism is not identified.
* **"The schedule hides half the planets" was wrong.** The unobserved gap costs 12 (1S2L) and 20 (2S2L) points of claimable anomalies.
* **Recall against RMDC26's generator labels is not bounded by the detectable fraction**, and "no anomaly a
  survey could claim" was an overstatement of what our 0.02 mag, single-season policy says.
* **The onset fix had two off-by-one errors and changed a released default**; the relabel rule that consumes
  the onset is not truth-based even with an exact onset (audit finding 9 re-opened).
* **The finite-source magnification function had artificial 4-8 mmag steps**; replaced, and the recommended
  checkpoint's recipe rerun with the smooth function.
* **The finite-source gain had no continued-training control**; one was run.
* **RMDC26 served as a development set** (diagnosis, a prior bound, checkpoint choice among seven); disclosed.
* Plus the count error (385,004 was an event id; the release has 377,999 events), an impossible argmax
  statement, uncommitted notebooks and weights, numbers with no artifact, double rounding, and stale text.

Two statements made in conversation during the work were also wrong and are corrected here: the quantised
onset DOES enter the shipped model's training labels (through the truncation augmentation, as the paper
says), and the between-seasons result does not show the schedule hiding half the planets.

A label-policy fragility older than this work was found along the way (a borderline single-start refit that
flips under 1e-14 mag perturbations, about one event per thousand); it is recorded in
`validation/gulls/PROVENANCE.md` and not changed, because changing it would change the released labels.

## Issues and dispositions

| Issue | Severity | What was wrong | What was done | Findings |
|---|---|---|---|---|
""" + "\n".join(lines) + """

## What no verifier checked (the adversarial pass's completeness list) and what happened to it

""" + "\n".join(f"* {g}" for g in R["gaps"]) + """

Dispositions of these gaps: development-set use, seed variance, the cascade model mix, the calibration
mask, out-of-support attribution, held-out disjointness (held-outs use shards 100-103, outside the shipped
model's training shards 0-89 and every fine-tune's shards 0-11), the occupancy mismatch, the notebook/CI
break and the missing bibliography entry are all addressed above or in paper/REVISION.md section 1.5. Not
done: a clean-archive paper build (deferred to the macro integration), and establishing whether RMDC26's
flux_err is computed at the true or the measured flux, whether GULLS includes limb darkening, and what
ObsGroup_0_chi2 is exactly (questions for the data providers; disclosed as unknowns).
"""
open(os.path.join(os.path.dirname(HERE), "VERIFICATION_2026-09-11.md"), "w").write(doc)
print("wrote docs/VERIFICATION_2026-09-11.md")
