import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Week 1 — Research Question

**Author:** Zeyad
**Date:** 2026-07-10
**Status:** Provisional — confirmable/changeable through end of Week 4 per the lane guide.
"""))

# 1. My lane and why
cells.append(nbf.v4.new_markdown_cell(
"""## 1. My lane (or freestyle) and why

**Lane: Refresh / Content Opportunity Scoring** (Core lane, from `docs/ml-intern-dataset-and-lane-guide.md`).

Why this one over the other three core lanes:

- It's the only lane where I already have an *honestly validated* result in this repo, not just an idea.
  `notebooks/01_first_look_and_discovery.ipynb` ran the full pipeline (`scripts/run_all.py`) with a
  **client-holdout** split (no client's pages appear in both train and test) and a learned model beat a
  transparent hand-written rule by a wide margin on Precision@50 — see the numbers below.
- The candidate pool is large enough to matter: pages that are both "declining" and still getting real
  search demand are ~44% of the starter slice, not a rare-event problem.
- A single hand-tuned threshold rule (stale *and* highly visible) turns out to flag almost nothing — which
  is itself evidence that this problem needs a scored/learned approach rather than one if-statement.
- It produces the artifact FlyRank's reviewers actually want: a ranked queue with reason codes, not just a
  correlation table (Lane 1) or a cluster diagram (Lane 3).

I considered **Lane 4 (CTR / Engagement Scoring)** as a close second — the starter data has a comparable
volume of low-CTR visible pages (see the code cell below) — and I may pivot there or fold it in as a
secondary reason-code set inside the refresh queue (the starter pipeline already tags rows
`ctr_review_candidate` / `engagement_review_candidate`). I'm leaving that door open through Week 4.

I ruled out **Lane 1 (Ranking Signal Analysis)** as the *primary* lane because it doesn't require picking
a decision/action — I'll still use its EDA techniques inside Lane 2. I ruled out the mentor-gated
**AI Referral lane** for now: only ~6% of rows have any AI-referred sessions in this slice, too sparse to
be my main lane (the guide explicitly warns against training a classifier on AI sessions alone)."""))

# 2. The question
cells.append(nbf.v4.new_markdown_cell(
"""## 2. The question: decision, action, cost of a wrong call

**Research question:** Which content pages should a FlyRank reviewer look at *first* when deciding what to
refresh, expand, protect, prune, or monitor?

**Unit of analysis:** one row = one content page (`content_id`) at one snapshot in time — a page's
90-day trailing metrics plus its trend bucket. (Not a client, not a day.)

**Decision this supports:** given limited reviewer time, which pages go to the top of the review queue
this cycle.

**Output:** a ranked score per page (0–100) plus human-readable reason codes (e.g.
`declining_with_demand`, `low_ctr_visible_page`) and a suggested action (`refresh`,
`refresh_and_review_ctr`, `expand_and_refresh`, `monitor`).

**Action a human takes:** a content editor opens the top-ranked pages, checks them manually against
editorial context, and decides whether to rewrite/expand the content, fix title/meta for CTR, or leave it
alone. The model does not publish anything on its own.

**Cost of a wrong call — and it's asymmetric:**
- *False positive* (flagged, but page was actually fine): wastes an editor's review time on a healthy page.
  Annoying, but cheap — it costs a few minutes.
- *False negative* (missed, but page was genuinely declining with real demand): a page that was worth
  saving keeps losing visibility/traffic silently until the next review cycle. More expensive, especially
  for high-impression pages, because the lost exposure compounds.
- Because false negatives are costlier here, I should not evaluate only on Precision@K — Recall and the
  shape of what gets missed also matter, and I'll report both, not just whichever number looks best.

**Why data/ML helps here at all (not just "train a model"):** a single hand-written rule
(`days_since_last_update >= 180 AND impressions_90d >= 500`) only matches a tiny sliver of pages in this
slice (see below) — real decline shows up as a *combination* of weaker signals (position, impression
trend, content age, word count, CTR) that no single threshold captures well. That's exactly the case where
a model that can weigh several weak, noisy signals together is expected to outperform one brittle
if-statement — and the starter pipeline already shows that happening under an honest, client-grouped
validation split. The open question for the next 7 weeks is whether that finding holds up on the full
79M-row warehouse, with a properly future-looking label (prior-90-days -> next-30-days decline) instead of
the starter's same-window `trend_direction` proxy."""))

# 3. Quick look at the data
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Quick look at the data (real numbers from the starter dataset)

Loaded live from `data/raw/content_refresh_anonymized.csv` and the already-committed pipeline outputs
(`outputs/model_results.json`, `outputs/model_report.md`) — nothing below is hand-typed."""))

cells.append(nbf.v4.new_code_cell(
"""import json
import pandas as pd

df = pd.read_csv("../../data/raw/content_refresh_anonymized.csv")
print(f"{len(df):,} rows (pages) across {df['client_id'].nunique()} pseudonymized clients")

# Number 1 — the candidate pool size for this lane
declining_with_demand = ((df["trend_direction"] == "down") & (df["impressions_90d"] >= 100)).sum()
print(f"\\n'declining_with_demand' pages: {declining_with_demand:,} "
      f"({declining_with_demand/len(df):.1%} of all pages) -> a real, workable review pool, not a rare event")

# Number 2 — why a single hand rule is not enough on its own
stale_visible = ((df["days_since_last_update"] >= 180) & (df["impressions_90d"] >= 500)).sum()
print(f"'stale AND visible' hand rule (>=180 days stale, >=500 impressions): {stale_visible} pages "
      f"({stale_visible/len(df):.2%}) -> one brittle threshold catches almost nothing")

# Number 3 — the already-validated model-vs-baseline result (client-holdout split)
res = json.load(open("../../outputs/model_results.json"))
base = res["baseline"]["baseline_precision_at_50"]
rf = res["models"]["random_forest"]["precision_at_50"]
print(f"\\nBaseline rule   Precision@50: {base:.3f}  (~{round(base*50)} of the top 50 right)")
print(f"Random forest   Precision@50: {rf:.3f}  (~{round(rf*50)} of the top 50 right)")
print(f"Learned model is ~{rf/base:.1f}x the hand rule, under a {res['split_strategy']} split "
      "(no client's pages in both train and test)")"""))

cells.append(nbf.v4.new_markdown_cell(
"""Secondary check — how Lane 2 compares in *volume* to my closest alternative, Lane 4 (CTR/Engagement):"""))

cells.append(nbf.v4.new_code_cell(
"""visible = df[(df["impressions_90d"] >= 500) & (df["avg_position"] > 0) & (df["avg_position"] <= 20)]
low_ctr = visible[visible["ctr"] < 0.5]
print(f"CTR-lane candidate pool (visible, position 1-20, low CTR): {len(low_ctr):,} "
      f"({len(low_ctr)/len(df):.1%} of all pages) -> comparable volume to Lane 2, a real fallback option")"""))

# 4. Careful words
cells.append(nbf.v4.new_markdown_cell(
"""## 4. Careful words: what I can and can't claim

**Safe to say:**
- These are *observed / directional* patterns in a 30,000-row anonymized starter slice covering 32
  clients, not a proof about how any search engine ranks pages.
- "Declining" means `trend_direction == "down"` — a same-window derived proxy (last-30-days vs
  prior-30-days impressions, drop > 20%) — not a certified fact about business impact.
- The 2.8x Precision@50 lift came from an honest client-holdout split *on this starter slice*; it is
  evidence the approach is promising, not a benchmark on the full warehouse, and it needs to be re-earned
  there with a proper future-looking label.
- `content_id` / `client_id` are pseudonyms used only for joins and grouped splits — never treated as
  meaning or as identity.

**Not safe to say:**
- I cannot claim refreshing a flagged page *will cause* recovery — that needs an experiment or causal
  design I don't have.
- I cannot claim anything about Google's (or any AI platform's) actual ranking algorithm.
- I won't reproduce or lean on a FlyRank product decision flag (`health_score`, `priority_score`,
  `action_type`) as a feature — the internship data is observable-only by design, and my baseline/model
  should explain itself from raw signals, not from the product's own prior decision."""))

# 5. Self-check
cells.append(nbf.v4.new_markdown_cell(
"""## 5. Self-check

Lane 2 it is, for now — the client-holdout result and the 44% pool are the main reasons, not just
that it "sounded interesting." The hand-rule number (0.06%) is doing the real work of the argument:
it's the evidence that this actually needs a model, not just a decision I'm asserting.

Lane 4 is still a live option, not a formality — the pool size is close enough that I'd feel bad
ignoring it. I'll decide by Week 4 whether it's a secondary reason-code set or a real pivot.

The biggest thing I have to keep being disciplined about: the label right now is `trend_direction`,
a same-window proxy, not a real future outcome. Nothing here proves the model works — it proves the
starter slice responds to this approach under one specific split."""))
nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

with open("work/notebooks/w01_research_question.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote work/notebooks/w01_research_question.ipynb")
