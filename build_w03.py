import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""# Week 3 — Data Contract on the Full Warehouse Release

**Author:** Zeyad
**Date:** 2026-07-19
**Lane:** 2 — Refresh / Content Opportunity Scoring (per `w01_research_question.ipynb`,
`w02_ml_task_framing.ipynb`)

This is the first notebook that runs against the real warehouse (`FlyRank/internship-warehouse`
on Hugging Face) instead of the 30k-row starter CSV. The job is small on purpose: write the
contract in plain words, prove three facts about my slice with real queries, build five features,
then reproduce the leakage lesson from notebook 02 on real warehouse data instead of the starter
slice.

Per the assignment's own warning: `fact_content_daily_performance_sample` is the sealed final
month (June 2026) — test-only, never for label logic. Everything below iterates on a mid-panel
month, `month=2026-03`, and treats the final month as untouched."""))

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 0. Setup

Same pattern as `notebooks/03_working_with_the_full_release.ipynb`: DuckDB talks to the
Hugging Face release directly, no local download of the full table. I'm pointing at the
`month=2026-03` partition specifically rather than the whole 79M-row fact table — that's the
whole point of the partition, and it keeps this notebook fast and rate-limit-friendly."""))

cells.append(nbf.v4.new_code_cell(
"""%pip install -q duckdb huggingface_hub"""))

cells.append(nbf.v4.new_code_cell(
"""import os, getpass

# CI and power users set HF_TOKEN in the environment; everyone else gets the safe prompt.
# Never hardcode the token in a cell -- this repo is public.
HF_TOKEN = os.environ.get("HF_TOKEN") or getpass.getpass("Paste your Hugging Face READ token (hf_...): ")"""))

cells.append(nbf.v4.new_code_cell(
"""import duckdb

con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")

REL = "hf://datasets/FlyRank/internship-warehouse"

TABLES = {
    "dim_clients":     f"read_parquet('{REL}/dim_clients.parquet')",
    # Partition path, not the full fact table -- this is the "iterate on a mid-panel month"
    # instruction taken literally: only March's parquet files get pulled over the network.
    "fact_march":       f"read_parquet('{REL}/fact_content_daily_performance/month=2026-03/*.parquet')",
}

for name, src in TABLES.items():
    n = con.sql(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
    print(f"{name:12} {n:>12,} rows")"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Why I'm not touching `dim_content` or `fact_content_query_90d` here, before I even get to the
contract:** the lane guide documents `dim_clients` and `fact_content_daily_performance` column by
column (`gsc_data_start`, `ga4_data_start`, `ga4_data_available`, `gsc_impressions`,
`gsc_clicks`, `gsc_avg_position`, the join keys). It never gives me the full `dim_content` column
list, and I don't have a way to browse the dataset's own schema page from here. Rather than
guessing column names and having the notebook break on a typo, I'm confirming the columns I
actually use with a schema peek below, and I'm leaving `dim_content` out of this week's contract
entirely — that's the limitation I name in section 4, not something to paper over.

`fact_content_query_90d` is a harder no, not just an unconfirmed-schema maybe: the data
dictionary's leakage warning says that table is a **fixed** 90-day window anchored to the export
date (`2026-06-23`), not a per-month table. For a March decision point that window doesn't even
line up with March — it sits close to the sealed final month. Joining it here would silently
staple future information onto a March row. That's my one deliberate exclusion in section 2."""))

cells.append(nbf.v4.new_code_cell(
"""# Confirm the columns I'm about to rely on actually exist under these names, instead of assuming.
con.sql(f"SELECT * FROM {TABLES['fact_march']} LIMIT 3").df()"""))

# ---------------------------------------------------------------------------
# 1. Contract, part A
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 1. The contract, part A — row, tables, time window

**What one row means for my lane.** One row is one pseudonymized content page
(`content_hash_id`, within one `client_hash_id`), described by a decision-moment snapshot built
by aggregating that page's daily search performance up to a cutoff date. It is **not** one row
per day — the raw fact table's grain is daily, and I aggregate on top of it. That distinction is
exactly what fact A in section 3 exists to prove: if I skipped verifying the raw grain first, an
aggregation bug (e.g. an accidental fan-out on a bad join) could quietly duplicate a page's rows
and I'd never notice from the aggregate alone.

**Which table(s).** `dim_clients` (for per-client history coverage — `gsc_data_start` /
`ga4_data_start` — and for client-holdout grouping later) and `fact_content_daily_performance`,
restricted to the `month=2026-03` partition. Both have documented, confirmed columns. See the
setup section above for why `dim_content` and `fact_content_query_90d` are out for this week.

**Which time window.** For this notebook I'm using March 15, 2026 as a mid-month decision point,
splitting the `month=2026-03` partition into a **feature window** (days 1–15, "what I'd know by
the decision moment") and a **forward window** (days 16–31, "what happens next"). This is a
compressed, single-month stand-in for the real design I flagged at the end of `w02`: trailing
90 days of features predicting a next-30-days decline. I'm not building that full version yet —
the assignment is scoped to a single mid-panel month to test query mechanics honestly before I
spend real HF bandwidth on the full multi-month version."""))

# ---------------------------------------------------------------------------
# 2. Contract, part B
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 2. The contract, part B — label/proxy and exclusion

**What I'd predict or rank.** Same lane target as `w02`: an urgency score used to rank pages for
refresh review. For this notebook's teaching-scale exercise, the proxy label is
`is_declining = 1` when a page's forward-window (days 16–31) impressions drop more than 20%
versus its feature-window (days 1–15) impressions, mirroring the starter CSV's
`trend_direction == "down"` rule but computed fresh from the raw daily warehouse table instead of
inherited from a pre-built column. This is still a same-window-adjacent proxy, not the real
future-looking capstone label — see the self-check for what's still open.

**One thing I deliberately exclude.** `fact_content_query_90d` — reasoned through in the setup
section above. It's a fixed window anchored near the export date, not a per-month table, so it
cannot honestly describe March without leaking information from months that, relative to a March
decision point, haven't happened yet."""))

# ---------------------------------------------------------------------------
# 3. Prove it: three queries, five features, the trap
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Prove it — three queries, five features, the trap

### 3a. Fact one — the grain

If the raw fact table really is `report_date × client_hash_id × content_hash_id`, grouping by
that triple and counting rows should equal the row count of the whole slice. No mismatch, no
silent duplication."""))

cells.append(nbf.v4.new_code_cell(
"""grain_check = con.sql(f\"\"\"
    SELECT
        COUNT(*)                                                      AS raw_rows,
        COUNT(*) FILTER (WHERE key_count = 1)                         AS rows_with_unique_key
    FROM (
        SELECT report_date, client_hash_id, content_hash_id, COUNT(*) AS key_count
        FROM {TABLES['fact_march']}
        GROUP BY 1, 2, 3
    )
\"\"\").df()

print(grain_check)
assert grain_check.loc[0, "raw_rows"] == grain_check.loc[0, "rows_with_unique_key"], \\
    "grain assumption broken -- (report_date, client_hash_id, content_hash_id) is not unique"
print("Confirmed: one row per (report_date, client_hash_id, content_hash_id) in the March slice.")"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3b. Fact two — slice size and date span

How big is my working slice, and does it actually cover the calendar month I asked for?"""))

cells.append(nbf.v4.new_code_cell(
"""slice_shape = con.sql(f\"\"\"
    SELECT
        COUNT(*)                       AS row_count,
        MIN(report_date)               AS min_date,
        MAX(report_date)               AS max_date,
        COUNT(DISTINCT client_hash_id) AS n_clients,
        COUNT(DISTINCT content_hash_id) AS n_content_items
    FROM {TABLES['fact_march']}
\"\"\").df()

slice_shape"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3c. Fact three — availability, filtered with `IS TRUE`

`ga4_data_available` is `FALSE` for rows before a client's GA4 start — zero-filled GA4 columns on
those rows mean "not measured," not "measured as zero." How many of March's rows actually carry
usable GA4 coverage?"""))

cells.append(nbf.v4.new_code_cell(
"""availability = con.sql(f\"\"\"
    SELECT
        COUNT(*)                                      AS total_rows,
        COUNT(*) FILTER (WHERE ga4_data_available IS TRUE)  AS ga4_available_rows,
        ROUND(100.0 * COUNT(*) FILTER (WHERE ga4_data_available IS TRUE) / COUNT(*), 1) AS pct_available
    FROM {TABLES['fact_march']}
\"\"\").df()

availability"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3d. Five features, decision point = March 15

Every feature below is built **only** from `report_date BETWEEN '2026-03-01' AND '2026-03-15'` —
the feature window, strictly before the forward window the label comes from in 3e.

- `imp_h1` (SUM of `gsc_impressions`, days 1–15) — knowable at the decision moment because it's
  the running impression total already logged by March 15; nothing about it depends on what
  happens March 16 onward.
- `clicks_h1` (SUM of `gsc_clicks`, days 1–15) — same reasoning, observed clicks through the
  cutoff.
- `avg_position_h1` (AVG of `gsc_avg_position`, days 1–15, excluding no-data rows) — the page's
  average observed ranking position through the cutoff; a snapshot of where it already stood.
- `active_days_h1` (COUNT of distinct days with `gsc_impressions > 0`, days 1–15) — how
  consistently the page showed up in search through the cutoff, not just its total volume.
- `ga4_covered_days_h1` (COUNT of distinct days with `ga4_data_available IS TRUE`, days 1–15) —
  an ingestion-time coverage flag, not an outcome; it's set by when the client's GA4 connection
  started, which is already known by the cutoff regardless of what the page does later."""))

cells.append(nbf.v4.new_code_cell(
"""features_h1 = con.sql(f\"\"\"
    SELECT
        client_hash_id,
        content_hash_id,
        SUM(gsc_impressions)                                              AS imp_h1,
        SUM(gsc_clicks)                                                   AS clicks_h1,
        AVG(gsc_avg_position) FILTER (WHERE gsc_avg_position > 0)         AS avg_position_h1,
        COUNT(DISTINCT report_date) FILTER (WHERE gsc_impressions > 0)    AS active_days_h1,
        COUNT(DISTINCT report_date) FILTER (WHERE ga4_data_available IS TRUE) AS ga4_covered_days_h1
    FROM {TABLES['fact_march']}
    WHERE report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
    GROUP BY 1, 2
\"\"\").df()

print(f"{len(features_h1):,} (client, content) rows in the feature frame")
features_h1.head()"""))

cells.append(nbf.v4.new_markdown_cell(
"""### 3e. The trap — one label-derived column, on purpose

The label: did the page's forward-window (days 16–31) impressions drop more than 20% versus its
feature-window (days 1–15) impressions? This is the exact same rule the starter CSV's
`trend_direction == "down"` uses (`docs/data-dictionary.md`), just computed fresh here instead of
inherited pre-built.

I'll build the honest features (`features_h1`, already strictly before the label window), then
deliberately add `imp_h2` — forward-window impressions, the literal quantity the label is
computed from — as an extra "feature," exactly the leak notebook 02 warned about."""))

cells.append(nbf.v4.new_code_cell(
"""forward = con.sql(f\"\"\"
    SELECT
        client_hash_id,
        content_hash_id,
        SUM(gsc_impressions) AS imp_h2
    FROM {TABLES['fact_march']}
    WHERE report_date BETWEEN DATE '2026-03-16' AND DATE '2026-03-31'
    GROUP BY 1, 2
\"\"\").df()

import pandas as pd

trap = features_h1.merge(forward, on=["client_hash_id", "content_hash_id"], how="inner")
trap = trap[trap["imp_h1"] > 0].copy()  # ratio undefined at imp_h1 == 0
trap["is_declining"] = (trap["imp_h2"] < 0.8 * trap["imp_h1"]).astype(int)

print(f"{len(trap):,} rows with a defined label")
print(trap["is_declining"].value_counts(normalize=True).rename("share"))"""))

cells.append(nbf.v4.new_code_cell(
"""from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

honest_features = ["imp_h1", "clicks_h1", "avg_position_h1", "active_days_h1", "ga4_covered_days_h1"]
X_honest = trap[honest_features].fillna(0)
y = trap["is_declining"]

honest_model = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_honest, y)
honest_acc = accuracy_score(y, honest_model.predict(X_honest))
print(f"Honest quick score (5 features, all from before the decision cutoff): {honest_acc:.3f}")

# Now the trap: add imp_h2 -- the label's own generating column -- as a 6th "feature".
leaky_features = honest_features + ["imp_h2"]
X_leaky = trap[leaky_features].fillna(0)

leaky_model = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_leaky, y)
leaky_acc = accuracy_score(y, leaky_model.predict(X_leaky))
print(f"'Leaky' quick score (same 5 features + imp_h2): {leaky_acc:.3f}  <- jumps toward perfect")
print()
print(export_text(leaky_model, feature_names=leaky_features))"""))

cells.append(nbf.v4.new_markdown_cell(
"""The tree splits almost entirely on `imp_h2`, because `imp_h2` *is* the quantity the label is a
threshold rule on top of — the model isn't finding a pattern, it's reading the answer key. That's
the same lesson notebook 02 taught with `trend_pct`, reproduced here on real warehouse data by
me, not inherited from the starter pipeline.

Deleting the leak and keeping the honest number:"""))

cells.append(nbf.v4.new_code_cell(
"""# Delete the leak, keep the honest number.
del X_leaky, leaky_model
print(f"Honest quick score, kept: {honest_acc:.3f} (5 features, all knowable by March 15)")
print(f"Leaky quick score, discarded: {leaky_acc:.3f} (included the label's own generating column)")"""))

# ---------------------------------------------------------------------------
# 4. Limitation
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 4. One named limitation of this slice

I built this contract without `dim_content`. The lane guide names `dim_content` as a real table
(519,606 rows, one per content item) and tells me it holds content metadata and join context, but
it doesn't give me a confirmed column list the way it does for `dim_clients` and
`fact_content_daily_performance` — and I don't have a way to browse the dataset's schema directly
from here. Rather than guess column names, I left it out entirely. That means this contract has
zero content-type, word-count, or query-context signal — every feature above is pure time-series
behavior. That's a real gap, not a stylistic choice: two pages with identical `imp_h1` /
`avg_position_h1` could be completely different kinds of content, and I currently can't tell them
apart. Before I build the real feature set for modeling weeks, I need to actually inspect
`dim_content`'s schema (a `DESCRIBE` / `LIMIT 5` against it, the same way I confirmed
`fact_march`'s columns in section 0) rather than carry this gap forward silently."""))

# ---------------------------------------------------------------------------
# 5. Self-check
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"""## 5. Self-check

The part of this notebook I trust least is the label. Splitting March into two 15/16-day halves
and calling a >20% drop "declining" is a direct, deliberate compression of the real design I keep
pointing at (prior 90 days → next 30 days) down to something that fits inside one partition and
one sitting — useful for proving the query mechanics and the leak honestly, but the 16-day
forward window is short enough that ordinary week-to-week noise could easily produce a false
"decline" that has nothing to do with the kind of sustained drop `docs/ml-intern-dataset-and-lane-guide.md`
distinguishes from real decline (section 7: consolidation, seasonality, noise). I don't think
that disqualifies this notebook — the job this week was proving I can state and verify a contract,
not ship a label I'd defend at the capstone — but I shouldn't let a clean-looking accuracy number
here convince me the label itself is ready.

The grain check in 3a was worth doing explicitly rather than assuming the guide's documented key
was actually enforced in this partition — a partition-write bug or a bad upstream join could
silently duplicate rows and I'd have built every feature on top of that without ever seeing it in
the feature frame itself.

Leaving `dim_content` out was the right call for this week rather than guessing at column names
and shipping a notebook that might silently reference something that doesn't exist, but it's a
gap I'm carrying forward on purpose, named in section 4, not one I'm pretending isn't there.

Going into the modeling weeks, the concrete thing I want to test is whether the real design (prior
90 days of features, a next-30-days label, evaluated under a client-holdout split like
`scripts/03_train_model.py` uses) holds up anywhere near as well as this compressed mid-month
version suggests — and whether `dim_content`, once I've actually looked at its schema, adds
enough over pure time-series signal to be worth the join complexity."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

import os
os.makedirs("work/notebooks", exist_ok=True)
with open("work/notebooks/w03_data_contract.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote work/notebooks/w03_data_contract.ipynb")