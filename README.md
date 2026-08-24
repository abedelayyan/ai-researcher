# Signal Zero

An AI research scout that reads new arXiv papers the day they appear, scores them for
commercial potential using only information that existed at publication time, and logs
every prediction so it can be graded weeks later against what actually happened.

The value is speed of detection. A paper already trending on Hugging Face or sitting on
2,000 GitHub stars has been seen by thousands of people, so surfacing it adds nothing.
Crowd signals here are never inputs to the score. They are outcome labels, collected
after the fact, used to measure whether the scorer was right.

Phase 1 is built: ingest, day-zero scoring, the two-bucket ranking, the prediction log,
outcome collection and the digest. Phase 2 starts with the backtest gate and is not
built yet. See "What is next" below.

## How it works

```
arXiv new listings  ->  day-zero features  ->  two rankings  ->  digest + prediction log
      (t=0)                (t=0 only)          (t=0 only)              |
                                                                       v
                     HF, GitHub, OpenAlex, S2, HN  ->  outcomes at t+7, t+30, t+90
                                                              |
                                                              v
                                                   composite label, precision@k
```

**Ingest** polls the arXiv Atom API directly across cs.AI, cs.LG, cs.CL, cs.CV and
stat.ML, one request per three seconds, with backoff on 429 and a category RSS fallback.
Hugging Face Daily Papers curation lands a day after the arXiv announcement, so reading
arXiv first is the whole latency advantage.

**Day-zero features** are everything knowable on the morning a paper appears:

| Feature | Source |
| --- | --- |
| Capability delta, four axes 0 to 3 | cheap model against `prompts/capability_delta.md` |
| Benchmark jump size | parsed from the abstract, points and multiples separately |
| Day-zero code release | repository link in abstract, comments or related links |
| Compute band | consumer, single node, small cluster or frontier |
| Author prior | OpenAlex citation velocity, date-bounded, null when unknown |
| Cheap structural signals | cross-listing, system claim, named domain, first-seen terms |

**Two rankings, never merged.** The confidence bucket weights the author prior in. The
high-variance bucket takes papers with a null or low prior and a large claim, ranked on
claim size alone. A merged ranking would be dominated by known labs every time, and the
second bucket is where anything worth finding early would show up.

**The prediction log** records every paper scored, including the ones scored low. Without
the negatives the dataset cannot be calibrated. Outcomes are collected at t+7, t+30 and
t+90 and collapsed into one composite label at t+30, which gives precision-at-k per run
against that day's base rate.

## The anti-leakage rule

If a feature could not have been computed at publication time, it cannot be used to rank.
This is enforced in three places rather than by convention:

1. **Separate tables.** `paper_features` and `paper_scores` on one side,
   `paper_outcomes` and `outcome_labels` on the other.
2. **A locked connection.** `db.open_feature_scoped()` installs a SQLite authoriser that
   denies every read and write touching an outcome table. Feature extraction and scoring
   are handed that connection and nothing else, in tests and in the daily run alike.
3. **A test that reads the code.** `tests/test_no_leakage.py` walks the AST of every
   module under `src/features/`, `src/score/` and `src/ingest/`, and fails if one imports
   outcome code or so much as names an outcome column.

The moment a backtest harness exists it becomes easy to accidentally train on the future,
which is why the enforcement went in before the scorer did.

One deviation from the spec's layout: calibration lives in `src/outcomes/calibration.py`
rather than under `src/score/`. It reads predictions and outcomes together, which is
exactly the pairing the scorer may not see, so it belongs on the outcome side of the wall.

## Running it

```bash
pip install -r requirements-dev.txt
python -m src.cli init-db          # create data/signal.db and apply migrations
python -m src.cli daily            # ingest, score, log, render today's digest
python -m src.cli outcomes         # collect any due t+7 / t+30 / t+90 observations
python -m src.cli calibration      # precision at k over the prediction log so far
python -m src.cli render --date 2026-08-21   # re-render a digest from stored data
python -m src.cli spend            # model spend per day
pytest                             # the leakage test is the one that must never go red
ruff check .                       # lint, same command CI runs
```

Useful flags on `daily`: `--date`, `--lookback-hours`, `--limit` for a quick run, and
`--offline` to work from the response cache with no outbound calls.

Everything is tunable from `config.yaml`: categories, lookback, scoring weights, bucket
sizes and thresholds, model tiers, outcome horizons and the composite label.

## Models and cost

Work is tiered. A cheap model does capability scoring and shortlist summaries for 30 to
50 papers a day. The strong tier is reserved for the weekly cluster and critic passes in
Phase 2.

Providers are tried in the order set by `llm.fallback_order`, and the first with
credentials wins:

- `github_models` needs only `GITHUB_TOKEN` and is free within tight limits, which is
  enough for Phase 1 volumes.
- `anthropic` sends the shared system block with `cache_control` set, since it is
  identical across every paper in a run.
- `openai`.
- `heuristic` needs no key and no network. It scores with keyword rules, marks the run as
  degraded in the digest, and exists so a bad day produces a provisional digest rather
  than nothing.

Two caps keep a heavy arXiv day from exhausting the free tier or running up a bill.
`capability_delta.max_model_papers` decides how many papers get a model call, and
`llm.max_calls_per_run` is a hard stop on the client. When the budget runs out the
remaining papers are scored by rules. Which papers get the model is decided by a
structural triage (code link, claim size, system claim, cross-listing) that uses day-zero
information only, and `paper_features.capability_source` records the path each row took,
so a rules-scored row is never mistaken for a model-scored one.

Every call is logged to `data/costs/spend.jsonl` and the `llm_calls` table, so cost is
visible from the first run. Prices in `src/llm/client.py` are mid-2026 list prices and
move monthly. Re-check before trusting a budget.

## Schedules

| Workflow | When | What |
| --- | --- | --- |
| `daily.yml` | 01:17 UTC | ingest, score, log, render, commit |
| `outcomes.yml` | 06:43 UTC | collect due observations, recompute labels |
| `tests.yml` | push and PR | pytest, including the leakage test |
| `pages.yml` | on `site/**` | publish the static digest |
| `keepalive.yml` | Mondays | stop the 60-day scheduled-workflow cutoff |

arXiv announces at roughly 20:00 ET, which is 00:00 UTC in summer, so the daily run
starts about an hour later. The cron minute is offset from the top of the hour because
scheduled runs bunch there and get delayed. Every run is idempotent: re-running a day
re-uses stored features and overwrites the digest, and a missed day is recovered by the
widened lookback window rather than by anyone noticing.

State is a SQLite file committed to the repo. Free, versioned, debuggable, and it works
offline. Turso is the move once a front end needs concurrent reads.

## Output

`data/digest/YYYY-MM-DD.md` is the daily read. `site/` holds the same digest as static
HTML for Pages, with an archive nav. Both carry the prediction log summary once labels
exist, hits and misses together, because showing calibration openly is what makes the
rest believable.

## Legal boundaries

arXiv metadata is CC0 and can be stored and displayed. Full text is not redistributable,
so nothing here downloads or rehosts PDFs. Summaries are written here, never copied from
Hugging Face, whose generated summaries are theirs. Requests carry a descriptive
User-Agent and respect the three-second arXiv rule. APIs are used wherever they exist
rather than scraping.

## What is next

Phase 2 starts with the validation gate, and nothing else in Phase 2 ships before it:

1. Build a labelled set of at least thirty 2024 and 2025 papers that became companies,
   products or widely adopted tooling, plus controls from the same weeks.
2. Replay the scorer over those weeks with date-bounded OpenAlex queries, so priors are
   computed as they stood then. `AuthorPriorService` already takes a `cutoff_date` and
   sums citations from the per-year breakdown rather than from today's total.
3. Measure precision-at-10 on the day each winner appeared. If the winners do not surface,
   the rubric in `prompts/capability_delta.md` is wrong, and no pipeline engineering fixes
   that.

Then weekly clustering, idea generation from clusters rather than single papers, and the
critic pass whose job is to kill ideas. Published evidence says on-paper novelty is
inflated and negatively predictive once ideas are executed, so the critic is the valuable
component, not the generator. The advance criterion is that it kills at least half of
first-pass ideas in ways a human reader agrees with.

Phase 3 is the startup dedupe layer and a public front end carrying the prediction log.

## Layout

```
config.yaml            thresholds, weights, model tiers, schedules
prompts/               versioned prompt templates, one file each
src/ingest/            arXiv API, category RSS, lab blogs
src/features/          day-zero extraction. Cannot import from outcomes
src/score/             component weighting, two-bucket ranking, prediction log
src/outcomes/          HF, GitHub, OpenAlex, S2, HN, composite labels, calibration
src/llm/               prompt loading, tiered providers, spend logging
src/render/            markdown digest and static HTML
src/store/             schema, migrations, the feature-scoped connection
src/pipelines/         the daily and outcomes runs
src/backtest/          Phase 2 gate, not built
src/themes/ ideas/ memory/   Phase 2, not built
src/dedupe/            Phase 3, not built
data/signal.db         committed SQLite
data/digest/           daily markdown
tests/                 including test_no_leakage.py
```

## Writing style

Digests and summaries follow one house style: British English, no em dashes, no inflated
adjectives, no "not just X but Y", no consultant filler. `tests/test_pipeline.py` checks
the generated digest for the worst offenders, and `tests/test_prompts.py` checks the
prompt files.
