import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Week 2 — ML Task Framing

**Author:** Zeyad
**Date:** 2026-07-16
**Lane:** 2 — Refresh / Content Opportunity Scoring (Lane 4 still open as a pivot through Week 4,
per `w01_research_question.ipynb`)
"""))

# 1. Task type
cells.append(nbf.v4.new_markdown_cell(
"""## 1. My lane as an ML task (type)

Lane 2 isn't one clean textbook category — it's a **binary classification model whose output
(a probability) is repurposed as a score to produce a ranking.** Being precise about that
distinction matters, so here's the breakdown:

- **Underlying model: binary classification.** The thing actually being learned is
  `is_declining_label` (page is declining vs not) from `scripts/03_train_model.py` — a
  standard supervised classification problem with a 0/1 target.
- **Consumed as: scoring.** The reviewer doesn't want a hard yes/no. `predict_proba()` gives a
  continuous number per page — that's the score, and it's what makes "how urgent is this page,
  relative to the other 29,999" answerable at all.
- **Delivered as: a ranking.** The score only matters in order — the pipeline sorts by it and
  hands editors a **queue** (`outputs/refresh_queue.csv`), not a table of probabilities. The
  decision this supports (see Week 1) is "which pages first," which is inherently a ranking
  question, not a classification one.
- **Not clustering.** There's a real target (`is_declining_label`) to predict, not an unlabeled
  structure to discover — clustering would be Lane 3's job (segmenting pages into groups),
  not mine.

So: classification model → probability score → ranked queue. I'll call it a **scoring/ranking
task built on a binary classifier**, because saying just "classification" would hide the part
that actually matters to the reviewer (relative order, not a category label), and saying just
"ranking" would hide how the score is produced (a trained probability, not a hand-built
formula)."""))

# 2. Target or proxy
cells.append(nbf.v4.new_markdown_cell(
"""## 2. Target or proxy

**Target used:** `is_declining_label` = 1 when `trend_direction == "down"`, i.e. impressions
in the last 30 days dropped more than 20% versus the prior 30 days (`docs/data-dictionary.md`).

This is explicitly a **proxy**, not the thing I actually care about:

- What I actually want to predict is *"will this page keep losing visibility if nobody touches
  it, and is it worth an editor's time to intervene."* That's a forward-looking, action-relevant
  outcome.
- What I have is *"did impressions already drop in a same-window comparison."* It's
  retrospective (the drop already happened by the time the row is scored) and it says nothing
  about whether a refresh would actually help — recovery isn't guaranteed even for a correctly
  flagged page.
- It's also intentionally **not** a leak: `trend_direction` and `trend_pct` are excluded from
  the model features precisely because the label is derived from them (`GUIDE.md`, rule 2 in
  the data dictionary).

The gap between proxy and target is exactly what I flagged as the open risk in Week 1: on the
full warehouse release (weeks 3+), I plan to swap this same-window proxy for a genuinely
future-looking label — prior-90-days features predicting a next-30-days decline — which is a
stronger match to the real decision than the starter slice's same-window definition."""))

# 3. Success metric
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Success metric

**Primary: Precision@50.** A reviewer works through a queue of finite size each cycle — Week
1's pipeline used the top 50 as that cycle's capacity. Precision@K asks the one question that
matters operationally: *of the pages I actually put in front of an editor, how many were worth
their time?* It's directly interpretable as reviewer-hours saved, which ROC AUC or accuracy
are not.

**Why not accuracy:** `is_declining_label` is close to balanced in this slice (~54% positive,
see the data dictionary), so accuracy wouldn't be *misleading* the way it would under heavy
imbalance — but it still isn't the right metric, because it scores the whole 30,000-row
population equally, when only the top of the ranking is ever acted on.

**Secondary, and non-negotiable given Week 1's cost analysis: Recall.** I already flagged in
Week 1 that false negatives (a genuinely declining, high-demand page that never gets flagged)
are more expensive than false positives (a healthy page that wastes a few minutes of review
time) — the lost exposure compounds silently until the next cycle. Optimizing Precision@50
alone could quietly let a model get very good at the *easy*, obvious top-50 cases while missing
a large tail of real decliners further down the ranking. So I'll report Recall (and Average
Precision, which summarizes the whole ranking rather than just the top slice) alongside
Precision@50, not instead of it — see the numbers pulled live below."""))

cells.append(nbf.v4.new_code_cell(
"""import json
import pandas as pd

res = json.load(open("../../outputs/model_results.json"))
rows = []
for name, m in res["models"].items():
    rows.append({
        "model": name,
        "precision_at_50": m.get("precision_at_50"),
        "recall": m.get("recall"),
        "average_precision": m.get("average_precision"),
        "roc_auc": m.get("roc_auc"),
    })
metrics_df = pd.DataFrame(rows).sort_values("precision_at_50", ascending=False)
metrics_df"""))

cells.append(nbf.v4.new_markdown_cell(
"""The best model on Precision@50 (random forest, per Week 1) isn't automatically the best model
overall once Recall is in the picture — worth watching as I move past the starter slice, not
just picking whichever row tops one column."""))

# 4. Unit of analysis
cells.append(nbf.v4.new_markdown_cell(
"""## 4. The unit of analysis, as a real dataframe

**One row = one content page (`content_id`), at one snapshot in time** — its trailing 90-day
metrics plus the derived trend bucket. Not a client, not a day, not a keyword. Below is an
actual slice of the starter data showing exactly that grain: identifiers for grouping, the
signals a model would see, and the target/proxy alongside them (shown here for inspection only
— never as a feature)."""))

cells.append(nbf.v4.new_code_cell(
"""df = pd.read_csv("../../data/raw/content_refresh_anonymized.csv")
print(f"{len(df):,} rows (pages) x {df.shape[1]} columns, across {df['client_id'].nunique()} pseudonymized clients")
print(f"One row per content_id -> {df['content_id'].nunique():,} unique ids for {len(df):,} rows: "
      f"{'confirmed 1 row per page' if df['content_id'].nunique() == len(df) else 'MISMATCH - investigate'}")

unit_of_analysis_cols = [
    "content_id", "client_id", "content_type", "content_age_days",
    "days_since_last_update", "impressions_90d", "clicks_90d", "ctr",
    "avg_position", "word_count", "trend_direction",
]
df[unit_of_analysis_cols].sample(8, random_state=7)"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4b. What the target column would look like

`is_declining_label` doesn't ship in the raw CSV — it's derived from `trend_direction` in
`scripts/01_prepare_features.py`. Sketching it here directly (not as a model feature, purely to
see its shape) confirms the class balance I cited above and shows it's a real, present-in-every-
row column, not a rare-event label."""))

cells.append(nbf.v4.new_code_cell(
"""sketch = df.copy()
sketch["is_declining_label"] = (sketch["trend_direction"] == "down").astype(int)

counts = sketch["is_declining_label"].value_counts().sort_index()
print("is_declining_label value counts:")
print(counts.rename({0: "0 (not declining)", 1: "1 (declining)"}))
print(f"\\nPositive rate: {sketch['is_declining_label'].mean():.1%}")

sketch[["content_id", "client_id", "trend_direction", "is_declining_label"]].sample(8, random_state=7)"""))

# 5. Why ML beats a fixed rule
cells.append(nbf.v4.new_markdown_cell(
"""## 5. Why ML beats a fixed rule here

This is the same evidence I built the Week 1 lane choice on, restated as the direct answer to
"why not just write an if-statement":

1. **A reasonable hand rule catches almost nothing.** `days_since_last_update >= 180 AND
   impressions_90d >= 500` — a rule that sounds sensible on paper — flags a tiny sliver of the
   30,000 pages (recomputed below). That's not noise; it's a sign the real pattern isn't a
   single threshold.
2. **The real candidate pool is large.** "Declining with real demand" (`trend_direction ==
   "down"` and `impressions_90d >= 100`) covers roughly 44% of pages — a workable, common
   pattern the fixed rule is almost entirely failing to surface.
3. **Decline shows up as a weighted combination of weak signals, not one strong one.** Position,
   impression trend, content age, word count, and CTR each carry a little information; the
   `outputs/model_report.md` feature-importance table shows the top signal
   (`days_with_impressions`) explains well under a fifth of the model's decisions on its own —
   no single column dominates enough to hand-code a rule around it.
4. **It's already been tested honestly, not just assumed.** `scripts/03_train_model.py` uses a
   **client-holdout split** (no client's pages appear in both train and test), and under that
   split the learned model still beats the hand rule by a wide margin on Precision@50 —
   confirmed live below, not asserted."""))

cells.append(nbf.v4.new_code_cell(
"""stale_visible = ((df["days_since_last_update"] >= 180) & (df["impressions_90d"] >= 500)).sum()
declining_with_demand = ((df["trend_direction"] == "down") & (df["impressions_90d"] >= 100)).sum()

print(f"Fixed-rule pool ('stale AND visible'):   {stale_visible:>6,} pages ({stale_visible/len(df):.2%})")
print(f"Real candidate pool ('declining+demand'): {declining_with_demand:>6,} pages ({declining_with_demand/len(df):.1%})")

base = res["baseline"]["baseline_precision_at_50"]
rf = res["models"]["random_forest"]["precision_at_50"]
print(f"\\nBaseline hand-rule score  Precision@50: {base:.3f}")
print(f"Random forest             Precision@50: {rf:.3f}  (~{rf/base:.1f}x the hand rule)")
print(f"Validated under a {res['split_strategy']} split, per scripts/03_train_model.py")"""))

# 6. Self-check
cells.append(nbf.v4.new_markdown_cell(
"""## 6. Self-check

Writing this out, the framing I keep needing to defend isn't the model choice — it's the label.
`is_declining_label` is a clean, present-in-every-row column that's easy to point at and say
"that's my target," but it's a same-window proxy for something that already happened, not a
forward prediction of what I actually want (whether intervention helps). I don't think that
disqualifies the task — Precision@50 and Recall are still meaningful on this proxy, and the
client-holdout result is real evidence, not an assumption — but I shouldn't let the proxy's
cleanliness make me forget it's a stand-in.

The unit-of-analysis check (`content_id.nunique() == len(df)`) was worth doing explicitly rather
than assuming it — it's the kind of thing that's obvious once confirmed and silently wrong if
it isn't, and a model built on the wrong grain (say, accidentally duplicating a page across
snapshot dates) would look fine in every metric until it hit new data.

Going into Week 3, the concrete thing I want to test on the full warehouse release is whether a
genuinely future-looking label (prior-90-days features predicting a next-30-days decline, using
the `fact_content_daily_performance` time series) holds up anywhere near as well as the
same-window proxy does here — that's the real test of whether this task survives contact with
a stricter definition of "declining.\""""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

with open("work/notebooks/w02_ml_task_framing.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote work/notebooks/w02_ml_task_framing.ipynb")