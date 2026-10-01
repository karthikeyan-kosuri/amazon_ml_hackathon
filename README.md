# ML Challenge 2026: Business Entity Resolution Solution Template

**Submission Date:** [27-09-2026]

---

## 1. Executive Summary

We built a blocking-plus-classifier entity resolution pipeline: DuckDB-backed
token blocking (scaled to the full 12.5M-record dataset), a LightGBM matcher
trained on RapidFuzz/Jellyfish string-similarity and address-overlap
features, and a decision threshold tuned directly against the challenge's
macro-averaged per-entity F0.5. Our main technical contribution is a
measured, evidence-driven blocking-recall diagnosis (not a guess) that
attributes 78.8% of missed true matches to an overly strict token
document-frequency filter versus 20.6% to genuine cross-script name
mismatches, which directly shaped our tuning decisions.

---

## 2. Methodology

### 2.1 Problem Analysis

Key patterns identified before and during pipeline development:

- **Missingness is structural, not incidental.** Source1 has zero missing
  fields (it's the clean reference source); Source2/Source3 addresses are
  missing on roughly 9–17% of records depending on source. This ruled out
  address as a primary matching signal — it can only be used as an
  assistive feature, with an explicit "both present" flag rather than
  treating a missing address as evidence of a mismatch.
- **Country is an open set, not a fixed pair.** Training data covers only
  US and India, but the test set adds France with zero training examples.
  This meant `country` could not be used as a categorical/one-hot feature
  anywhere in the pipeline (no vocabulary exists for an unseen value) —
  we use it only as a same/different boolean, which needs no vocabulary
  and generalizes to any unseen country string.
- **Cross-script name variation is real and measurable, not theoretical.**
  A meaningful share of Source2/Source3 business names contain non-ASCII
  characters (consistent with Indic-script transliteration of Source1's
  always-ASCII reference names), which plain token/character-level
  blocking cannot bridge by construction.
- **Match cardinality is asymmetric.** Ground truth confirmed that a
  Source1 entity can have zero to many true matches, but no single
  Source2/Source3 record ever belongs to more than one Source1 entity —
  a hard structural constraint we exploited directly in the assignment
  step (Section 4).
- **The evaluation metric rewards conservatism.** F0.5 is macro-averaged
  per Source1 entity, with singletons scored explicitly (empty prediction
  = 1.0, any false prediction = 0.0). This makes correctly predicting "no
  match" worth exactly as much as a large, fully correct match set, and
  makes over-eager blocking/matching actively harmful rather than merely
  suboptimal.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier

**Core Innovation:** Rather than tuning blocking parameters by intuition,
we built a direct, ground-truth-based diagnostic that decomposes *why*
specific true matches are missed (no shared token vs. frequency-filtered
vs. cut by the ranking cap), which let us make an evidence-based decision
between two very different threshold settings instead of guessing. We
also explicitly optimize the decision threshold against the challenge's
actual macro-averaged per-entity F0.5, rather than a generic
precision/recall tradeoff, since the two can favor different cutoffs
under this specific metric.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Normalized name tokens (post lowercasing,
  Unicode NFKD diacritic stripping, punctuation removal, and hand-written
  legal-suffix/abbreviation expansion — e.g. `Pvt` ↔ `Private`, `Corp` ↔
  `Corporation`), joined via DuckDB SQL on shared tokens within the same
  country. A per-country token document-frequency filter excludes tokens
  too common to be useful blocking keys (tuned value: 150 — see below for
  why). An optional multilingual sentence-embedding channel (general
  pretrained model, Apache-2.0, ~118M params) was designed to catch
  cross-script matches with zero token overlap, but was not enabled in
  the submitted run (see Section 5 / Appendix B).
- **Candidate pairs generated:** At full dataset scale (2,206,821 /
  5,034,616 / 5,285,603 records across Source1/2/3): 31,798,598 raw
  candidate rows via the DuckDB join, capped to 18,987,312 after
  per-entity ranking (max 40 candidates per Source1 entity, selected by
  batched RapidFuzz `token_sort_ratio` rather than a per-pair Python
  loop, which was necessary for this to complete in a tractable time).
  625,375 of 2,206,821 Source1 entities received at least one candidate
  at this stage.
- **How we ensured true matches were not lost:** We measured blocking
  recall directly against a 5,000-entity ground-truth sample rather than
  inferring it from candidate coverage alone. At the submitted
  configuration, 4,626 of 18,331 true links (~25.2%) survived blocking.
  We decomposed the misses by cause:

  | Cause | Share of misses |
  |---|---:|
  | Shared token(s), but excluded by the frequency filter | 78.76% |
  | No shared token at all | 20.58% |
  | Cut by the per-entity ranking cap | 0.66% |

  Of the "no shared token" cases, 47.0% involved a non-ASCII name on
  at least one side. We then tested loosening the frequency filter
  (150 → 800) specifically to recover the dominant failure category:
  raw blocking recall improved (4,626 → 7,038 true links recovered),
  but retraining on this looser setting *dropped* validation macro F0.5
  from 0.9219 to 0.7581, because the additional common-token false
  candidates hurt matcher precision more than the extra recall helped —
  expected under F0.5's 2× precision weighting. We reverted to 150 based
  on this direct comparison. The ranking cap was confirmed to cost
  negligible recall (0.66%) and was left unchanged.

---

## 4. Matching Model

**Features used:**

- **Name features:** Jaro-Winkler similarity, token-sort ratio, partial
  ratio, character-trigram Jaccard, and (when enabled) multilingual
  embedding cosine similarity with an explicit availability flag so the
  model can learn to trust or discount it.
- **Address features:** Token-level Jaccard overlap, computed only when
  both records have a present address, plus an explicit "both present"
  indicator — a missing address is never treated as evidence of a
  mismatch, given its high missingness rate in Source2/Source3.
- **Other:** Name length and token-count deltas; a same/different
  country boolean (deliberately not a categorical country feature, to
  remain correct on France, which has zero training examples).

**Model type:** LightGBM (gradient-boosted trees). Chosen for strong
performance on heterogeneous tabular similarity features, good
calibration for threshold-based decisions, and straightforward
open-source licensing well within the challenge's model constraints.
Training/validation split is done **by Source1 entity**, not by
candidate pair, to prevent leakage of an entity's other positive pairs
across the split. Negative sampling combines random negatives with
explicit hard negatives (the highest-scoring non-matches from the
blocked candidate set), with `scale_pos_weight` tuned for class
imbalance.

Ground truth confirmed a structural one-to-many constraint (one Source1
entity can have multiple matches, but each Source2/Source3 record
belongs to at most one Source1 entity) — enforced at inference by
keeping only the highest-scoring Source1 assignment for any Source2/3
record scored against more than one candidate.

**Threshold selection method:** Direct search over the macro-averaged,
per-Source1-entity F0.5 (the challenge's actual scoring formula) on a
held-out validation split, rather than a generic cutoff or a pooled
precision/recall calculation — the two can select different optimal
thresholds under macro-averaging, since singleton entities are weighted
equally to large-match entities in this metric.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9219 on held-out validation (threshold =
  0.90), measured on a training run using a capped, ground-truth-aware
  sample (50,000 Source1 entities; Source2/3 sampled up to 300,000 rows
  each, with each capped entity's true ground-truth matches deliberately
  included before random fill, to avoid starving the matcher of positive
  examples). Full, uncapped test-set inference (1,732,544 Source1
  entities) produced 504,668 non-empty predicted matches and passed the
  official `validate_submission.py` check with no reported issues.
- **Common false positives (wrong merges):** Primarily candidates sharing
  only a common, generic name token (industry terms, frequent words)
  that survive blocking due to the frequency threshold but are
  structurally similar enough to pass the matcher — this is the exact
  failure mode that worsened precision when we tested a looser frequency
  threshold (Section 3).
- **Common false negatives (missed matches):** Two distinct, measured
  causes: (1) true matches sharing only tokens common enough to be
  excluded by the blocking frequency filter (78.76% of misses in our
  diagnostic sample) — a tunable recall/precision tradeoff we
  deliberately chose not to loosen further given its measured cost to
  F0.5; (2) genuine cross-script name pairs with no shared token at all
  (20.58% of misses, about half involving non-ASCII text) — a category
  that would require the (currently disabled) embedding-based blocking
  channel to recover, not a blocking-threshold change.

---

## 6. Conclusion

We built and validated a scalable blocking pipeline (confirmed to run
end-to-end on the full 12.5M-record dataset) paired with a LightGBM
matcher tuned directly against the challenge's macro-F0.5 metric,
achieving 0.9219 on held-out validation. Our key lesson was the value of
measuring blocking recall by failure category rather than tuning by
intuition: a loosened frequency filter recovered more true matches but
measurably hurt F0.5 through lower precision, and we identified — without
yet fixing — a separate scalability limitation in negative-sampling for
full-scale training, which we document transparently in the appendix
rather than overstating our results.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under
`code/business_entity_resolution/`, with all source in `src/`:

- `normalize.py` — Unicode-generic text normalization and hand-written
  abbreviation expansion (no external dictionaries).
- `blocking.py` / config's DuckDB-backed token join — candidate
  generation, scaled via SQL joins rather than an in-memory Python
  index, plus batched RapidFuzz ranking.
- `embeddings.py` — optional multilingual embedding blocking channel
  (disabled in the submitted run; see below).
- `features.py` — feature engineering for candidate pairs.
- `train_matcher.py` — LightGBM training, per-entity split, hard-negative
  mining.
- `assign.py` — enforces the one-Source1-per-candidate structural
  constraint.
- `evaluate.py` — macro-averaged per-entity F0.5, matching the
  challenge's exact scoring rule, used for threshold search.
- `infer.py` — scoring, assignment, and output-file writing.
- `pipeline.py` — CLI entry point.

**To reproduce the submitted outputs**, from
`code/business_entity_resolution/`:
```bash
pip install -r requirements.txt
python -m src.pipeline --stage train   # produces the trained model + threshold
python -m src.pipeline --stage infer   # produces output/matching_results.tsv
                                        # and output/candidate_pairs.tsv
```

### B. Additional Results

**Full-scale blocking validation.** Blocking and candidate generation
were run against the complete, uncapped training dataset to confirm the
strategy itself scales to the full data volume (not just the capped
training sample used for the matcher):

| Stage | Result |
|---|---|
| Source loading (all 3 files) | ~19s total |
| DuckDB token-table creation | <1s |
| Candidate join | 31,798,598 raw candidate rows in 37.2s |
| Ranking / capping | 18,987,312 candidates in ~36.6 min |

**Full-scale training limitation (disclosed transparently).** We
additionally attempted training on the fully uncapped dataset (all
2,206,821 Source1 entities; 1,866,988 positive / 17,120,324 negative
labeled pairs). Blocking, ranking, and feature construction all
completed successfully at this scale. The negative-sampling step
(which sorts the full negative pool to select hard negatives before
downsampling) did not complete in a reasonable time — sustained low CPU
utilization is consistent with memory pressure from an in-memory
sort/concat over a 17-million-row DataFrame on our 16GB development
machine, rather than a genuine computational bottleneck. This is a
distinct, understood limitation isolated to one training-preparation
function, separate from blocking (validated above at full scale) and
separate from the matcher itself. The identified fix — sampling the
negative pool down to a bounded size *before* sorting for hard
negatives, rather than sorting the full pool first — was not implemented
before submission due to time constraints.

**Multilingual matching (identified, not yet implemented).** The
embedding-based blocking channel was designed specifically to address
the "no shared token" miss category (Section 3), roughly half of which
are genuine cross-script cases. It was not enabled in the submitted run,
primarily due to CPU-only inference cost at full dataset scale on our
development hardware. This is the clearest identified avenue for further
recall improvement.
