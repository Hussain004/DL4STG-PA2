# Q1 — River Forecasting Proposal: design investigation log

**Working notes for PA2 Question 1.** Everything here is the *reasoning trail* that
produced the design I will defend in the report: what I assumed, what I tested,
what the tests said, and which choices the evidence actually supports.

Companion material:

* `figures/A*.png` — the companion notebook, reproduced and quantified.
* `figures/B*.png` — the design laboratory (synthetic river, reach, delays).
* `figures/C*.png` — the architecture / ablation experiments.
* `results/*.json` — every number quoted below.
* `scripts/` — all code (runs on CPU; the heavy sweeps ran on the Lab PC GPU).

---

## TL;DR — the design this investigation produced

**A per-place temporal encoder → four rounds of message passing along the
river graph, each edge carrying a learned delay kernel → a residual read-out
at the forecast origin.**

| decision | why (evidence) |
|---|---|
| one grid, 24 observed + 6 forecast columns, 6 rows × 4 channels | the tutorial's layout; the future columns hold only the announced plan, so nothing unknown at issuance can leak |
| per-place temporal encoder, weights shared across places | the four stations behave alike; a place embedding lets the dam differ |
| **a directed spatial path along the river graph** | without it the model is blind upstream: 0.0308 → 0.0257 m, and the gain lands exactly where the information probe says it should (S4 −47 %) |
| **4 rounds** | the dam is 4 hops from S4 and each round moves information one hop. The accuracy saturates at one round (0.0255) and four rounds cost 3 % - a premium paid for the *reach* the flood case needs |
| **a learned delay kernel per edge** | the travel time is not supplied and changes with the river; the model learns the *useful* lag (≈ 50 min) and does not need the true one (≈ 30–80 min) |
| residual head on the last observed level | well-scaled, and the absolute level is still an input, so flood thresholds are not lost |
| **MAE in metres** as the training loss | it is the metric the service is judged on: MAE training is 14 % better at every station (Phase D). The flood concern belongs in the *metric* and the *alert threshold*, not in a re-weighting of the loss |

**The three things I would most want the committee to take away.**

1. *A spatial path is worth 16 %; which spatial path is worth almost nothing*
   — except that only the graph family is physically admissible.
2. *The upstream's value is a **lead**, not a wave.* S3's current level
   predicts S4's next hour 71 % better than S4's own history does. That is
   the single most useful fact I measured.
3. *Structural claims must be tested.* My first sweep said the spatial path
   did not help; an intervention test showed the "temporal-only" baseline
   was secretly mixing places. The fix changed the headline result.

---

## 0. The plan

The handout says: *"Choose any defensible architecture."* So the design cannot be
copied from a paper — it has to be **argued from the physics of this river**.
My process was:

1. **Extract the design lessons from the companion notebook** (Phase A). The
   notebook is not about rivers; it is about *information paths*. I turned each
   of its claims into a number I could cite.
2. **Build a design laboratory** (Phase B): a synthetic river that contains
   every phenomenon the handout names — directed flow, a monitored and an
   unmonitored tributary, two rain gauges, a reservoir with an announced
   schedule that is sometimes wrong, missing and late readings, and travel
   times that shorten in flood.
3. **Enumerate the architecture space** (Phase C) and test the candidates
   against each other on identical data, identical budget, identical head.
4. **Ablate everything** (Phase D): history length, number of spatial rounds,
   which input sources matter, the training objective, and what happens when
   the data degrade.
5. **Write down the design and the assumptions** (Phase E) and be explicit
   about what the evidence does *not* settle.

---

## Phase A — what the companion notebook actually proves

I ran the bundled checkpoints (no training) and measured the notebook's claims.
Raw numbers: `results/A1_notebook_findings.json`.

### A0. The notebook's four "Think about" prompts, answered

**(1) Missing information and the training objective.**
*(a)* A deterministic forecaster that reads only sensor 8's history receives
*identical inputs* in the two episodes, so it must emit the same number `p`;
its error on that pair is the whole target gap — a capacity problem cannot
fix an information problem.
*(b)* With targets −0.594 and +0.900 and equal weights:
`L_MSE(p) = ½[(p+0.594)² + (p−0.900)²]` is minimised at the **mean**
`p = 0.153` (minimum `0.558`); `L_MAE(p) = ½[|p+0.594| + |p−0.900|]` is
**flat at 0.747 for every p ∈ [−0.594, +0.900]**, so *all* of that interval
minimises it.
*(c)* In general: `L_MSE` is a strictly convex quadratic in `p` with
`L' = Σ(p − y_i)`, so it is minimised **uniquely** at the arithmetic mean; the
`L_MAE` minimisers are exactly the **medians** (for even `n`, every point
between the two central order statistics; for odd `n`, the middle one; with
repeated values, the interval between the two central values). *Design
consequence:* when an input is genuinely unobserved, MSE answers with the
mean, MAE with the median, and neither can do better than the central
tendency — the only way forward is to *infer* the missing input from proxies.

**(2) What the intermediate representation must preserve.** For an S → T
stack, the S block must preserve *where* a feature came from (its spatial
origin) and the T block must preserve *when* it was seen; the pair must be
kept separable. An intermediate representation that pooled over space (or
over time) before the second block would make the relation unrecoverable: the
T block could no longer tell "the signal that was one sensor to the left" from
"the signal that was here". This is the same failure the professor calls "the
lost spatial index" in the ViViT/LRCN discussion.

**(3) Minimum depth.** To use an observation 4 sensors to the left and 6 steps
before the origin: an alternating S–T stack needs **≥ 4 pairs** (each pair
buys one sensor of space and two steps of time) and a joint-ST stack needs
**≥ 4 blocks** (each block buys one sensor and two steps). Both grow linearly
with depth, so the depth is set by the *graph*, not by taste. On the river
graph the equivalent number is the hop count: **4 rounds** (Dam→S1→S2→S3→S4).

**(4) Debugging the ST forecaster.** *Diagnose:* (i) compare the held-out MAE
at near and far distances — the ST stack scores 0.007 at distance 1–3 and
**0.506 at distance 4–15**, which is exactly the "predict the mean" baseline;
(ii) check the gradient support: the ST model's gradient w.r.t. the input is
non-zero on 49 of 128 tokens and **exactly zero** on the rest, so the path is
missing, not mis-weighted; (iii) check that the paired forecast does not move
when the distant source changes. *Fix (two ways):* give the model a global
path (attention, cost `2L²d`) or make the stack deep enough to reach
(`9Ld²` per joint block, plus `L` more per extra block). *Verify:* before
retraining, the fixed model should be accurate at *every* distance and the
paired forecasts should move with the source. A model whose error stays at
the baseline but whose forecasts *do* move would mean the path exists and the
model is simply not using it — which is what the attention run shows (it
locks onto the source, but which head does it differs between runs).

### A1. An information boundary is not a capacity problem

The moving-field pair: sensor 8's next value **is** sensor 7's latest reading.
The two episodes have *identical* histories at sensor 8 and different targets.

| model | forecast shift when sensor 7 changes | required shift |
|---|---|---|
| T only (4 blocks) | **0.000 ± 0.000** | 1.494 |
| S → T (3 pairs) | **1.390 ± 0.104** | 1.494 |

*Figure A2 (`figures/A2_moving_pair_predictions.png`).*

**Reading.** The temporal-only model is not "less accurate", it is *blind*: with
identical inputs it must emit an identical number, so its error on this pair is
the full target gap. Stacking more T blocks cannot help — it only mixes
sensor 8's own history. **Design consequence:** a river model whose temporal
encoder is per-station must be given an explicit spatial path, or the
information simply is not there.

### A2. The MAE/MSE collapse (this is my tool for the unmonitored tributary)

With targets −0.594 and +0.900 and one prediction *p*:

* **MSE**: `L(p) = ½[(p+0.594)² + (p−0.900)²]`, minimised at the **mean**
  `p = 0.153`, with minimum `0.558`.
* **MAE**: `L(p) = ½(|p+0.594| + |p−0.900|)`, which is **flat at 0.747 for every
  p ∈ [−0.594, +0.900]** — every point between the two targets is a minimiser.

In general (n equally weighted targets): MSE is minimised by the arithmetic
mean, MAE by a **median** (any point in the interval between the two central
order statistics when n is even). **Design consequence:** when an input is
genuinely unobserved (the unmonitored tributary), a deterministic forecaster
must collapse toward a central tendency; MSE gives the mean, MAE the median.
The only way to do better is to *infer* the missing input from proxies.

### A3. Reach is a property of the stack, and it must be drawn

`assignment2.models.receptive_field` gives the exact structural support. For a
forecast at (t = 0, sensor 8):

| stack | sensors reachable | steps back reachable |
|---|---|---|
| T ×4 | 1 | 7 |
| S→T ×3 | 7 | 7 |
| T→S ×3 | 7 | 7 |
| ST ×3 | 7 | 7 |
| Attention | all | all |

*Figures A3 (`figures/A3_receptive_field_ST_d3.png`, `figures/A3_receptive_field_Sfirst_d3.png`).*

**Think-about 3 (minimum depth).** To use an observation **4 sensors** away and
**6 steps** back: alternating S–T pairs need **≥ 4 pairs** (each pair buys 1
sensor and 2 steps: 4 pairs ⇒ 4 sensors, 8 steps ✓); joint ST blocks need
**≥ 4 blocks** (each ST block buys 1 sensor and 2 steps). Both are *linear* in
depth. **Design consequence:** the number of spatial rounds is not a free
hyper-parameter — it is set by the longest hop count in the river graph.

### A4. Attention reaches everywhere and costs accordingly

Relay world: a marked source broadcasts its latest reading to every sensor.

| model | held-out MAE, distance 1–3 | distance 4–15 |
|---|---|---|
| ST (3 blocks) | 0.007 | **0.506** (= the "predict zero" baseline) |
| Attention (3 blocks) | 0.001 | 0.001 |

*Figure A4 (`figures/A4_relay_scores.png`).* ST is *exactly* at the trivial baseline
beyond its 3-sensor reach: it has no path, so it predicts the mean.

I verified the path claim by **gradient**, not by inspection: for a far-away
forecast, the ST model's gradient w.r.t. the input is non-zero on 49 cells and
**exactly zero** outside its structural support, while the attention model's
gradient is non-zero on all 128 tokens.

Cost (notebook formulas `C_A = 2L²d`, `C_ST = 9Ld²`):

| tokens L | d | attention | joint ST | ratio |
|---|---|---|---|---|
| 144 (24 × 6 rows, the river grid) | 64 | 2.65 M | 5.31 M | **0.50** |
| 192 (24 × 8 rows) | 64 | 4.72 M | 7.08 M | 0.67 |
| 288 (24 × 12 rows) | 64 | 10.6 M | 10.6 M | 1.00 |

**Reading.** At the river's scale attention is *not* the expensive option —
the quadratic term only bites when L ≳ 4.5 d. The real argument against
global attention here is not cost; it is that the river has a *known directed
structure* that attention would have to rediscover from data, and that a global
operator can create physically meaningless couplings (upstream influencing
upstream). **Figure A7** (`figures/A7_cost_attention_vs_st.png`) plots both curves.

### A5. The head of the notebook

* The reach map widget makes "can this input reach this forecast?" an
  explicit, checkable statement. **Design consequence:** I must be able to
  *draw* the receptive field of the river model (Figure B3, `figures/B3_reach_on_the_river.png`).
* "Having a path exists ≠ the model learns to use it." The relay attention
  maps show that in every run *some* head locks onto the source, but *which*
  head differs between runs (`A6_attention_view_seed{0,1,2}.png`).

---

## Phase B — the design laboratory

No dataset was supplied, and the handout does not require one. But a design
that cannot be tested is a design that cannot be defended, so I built a
**synthetic river** whose only job is to contain the phenomena the handout
names, and to make the candidate architectures separable.

### B1. What the simulator contains

| handout feature | how it is modelled |
|---|---|
| reservoir → S1 → S2 → S3 → S4 | directed graph, 5 edges, each with a travel delay line |
| monitored tributary T1 → S2 | its own chain, baseflow + unobserved catchment |
| **unmonitored** tributary U → S3 | its inflow is a *latent* variable; only part of it is explained by R2 |
| rain at R1, R2 | two-state Markov storm process, 10-min totals |
| travel times change with conditions | `τ(t) = τ₀ (Q_ref/Q(t))^1.55` — **the delay shortens as the river fills** |
| recorded releases + announced schedule | block-constant releases; the announced plan deviates with probability 0.18 |
| missing / late readings | 1.2 % dropped, 1.0 % stale by 1–3 steps, 1.2 cm measurement noise |
| target | clean levels at S1–S4, one hour ahead (6 steps) |

Levels come from a kinematic-wave chain (power-law rating curve, linearly
implicit update — unconditionally stable), so the *storage* response and the
*transport* delay are separate and both matter.

### B2. Calibration (this is what sets the lookback)

I measured travel times with a step-response probe (equilibrate, step the
release, record when each station first moves 10 % of its eventual change).

| reach | calm | flood |
|---|---|---|
| Dam → S1 | 0 min | 0 min |
| S1 → S2 | 30 min | 30 min |
| **S2 → S3** | **70 min** | **60 min** |
| S3 → S4 | 80 min | 90 min |
| Dam → S4 (total) | ≈ 180 min = **18 steps** | ≈ 180 min |

*Figure B2 (`figures/B2_travel_times.png`), raw numbers in `results/travel_times.json`.*

**Two things follow, and they are the backbone of the design.**

1. **The lookback must cover the slowest upstream influence path.**
   Dam → S4 is 4 hops and ≈ 180 min = 18 steps. Add the 1-hour horizon
   (6 steps) and the window must be ≥ 24 steps. **That is exactly the L = 24
   the tutorial suggests**, and it is not a round number someone chose: it is
   `18 + 6`. Anything shorter cannot see the release that will arrive at S4
   during the forecast hour.
2. **The delay is not a constant.** The *transport* time scales like
   `Q^(-1.55)`; in flood it is roughly half the calm value. The *observable*
   station-to-station lag changes less (70 → 60 min) because part of it is the
   storage response, which does not shorten. **Design consequence: do not
   hard-code a delay.** Let the model learn a *distribution* over lags, and
   make the window long enough for the slowest case.

### B3. The grid

Rows = places (Dam, S1, T1, S2, S3, S4); columns = time (24 observed + 6
forecast); channels = [level, observed?, rain, release]. R1's rain is folded
into the **S2** row and R2's rain into the **S3** row (the tutorial's stated
assumption). The Dam row carries recorded releases in the past and the
**announced plan** in the horizon columns; every other cell of the horizon is
empty and flagged "not seen".

*Figure B1 (`figures/B1_river_system.png`).*

**The golden rule, made mechanical:** the encoder is allowed to look *forward*
across the six horizon columns, because those columns contain only information
known at issuance (the announced plan). No future measurement can enter the
grid, so no future measurement can leak.

---

## Phase C — the architecture space

All candidates share the same input grid, the same head (residual on the last
observed level) and the same training protocol (AdamW, OneCycle, MSE in
metres, early stopping on validation MAE, 3 seeds). Only the spatial
mechanism changes.

| name | spatial mechanism | what it assumes |
|---|---|---|
| persistence | none | the river does not change in an hour |
| per-station ridge | none (univariate linear map) | each station is explained by its own past |
| temporal encoder only | none | **the upstream is irrelevant** |
| graph message passing | 4 rounds along the river graph, same-time mixing | water flows downstream, one hop per round |
| **graph + learned delay** | 4 rounds, each edge carries a **learned lag kernel** | water flows downstream, and the travel time is a *learnable* quantity |
| graph + delay attention | 4 rounds, each edge pools the upstream history with content-dependent weights | ...and the delay depends on the current state |
| joint ST convolution | 3×3 space-time kernel over the **row index** | "neighbouring rows are neighbours" |
| global space-time attention | attention over all 180 tokens | nothing; the model must find the structure |

**Results** — see `figures/C1_architecture_comparison.png` and
`results/arch_summary.json`. (Numbers inserted below once the sweep finished.)

### C1. The information probe: how much is the upstream worth?

Before training anything, I asked what the upstream can possibly contribute.
A ridge probe on the simulated river — predict the 1-hour-ahead *change* from
the station's own 24-step history, with and without the upstream stations'
current levels (probe in `scripts/diagnose_spatial.py`):

| station | own history | + upstream now | gain |
|---|---|---|---|
| S1 | 21.97 mm (R² 0.204) | 21.97 mm | 0 % |
| S2 | 95.82 mm (R² 0.322) | 95.11 mm | 0.7 % |
| S3 | 77.57 mm (R² 0.325) | 65.72 mm (R² 0.502) | **15.3 %** |
| S4 | 18.02 mm (R² 0.841) | 5.26 mm (R² 0.984) | **70.8 %** |

**Reading.** The upstream's value is *not* uniform, and it is not what one
might guess. S4's own level is a **lagged** copy of S3's, so S3's current
reading is a **lead**: it tells the model what S4 is about to do. The further
downstream the station, the more valuable the upstream becomes — the opposite
of the intuition that "downstream stations are more self-contained". S1 and
S2 get almost nothing from the upstream *now* (their upstream influence
arrives later than the forecast horizon).

This is the single most useful number in the whole investigation: it says the
spatial path is worth building, and it says **where** it is worth building.

### C2. The leak I nearly shipped

The first full sweep told me the spatial path did not help: temporal-only
0.0266 m versus graph 0.0278 m. That contradicted the probe above, so I
interrogated the model instead of trusting it.

The intervention test (companion-notebook Part 1, transplanted to the river):
keep S4's own history, **replace every other row of the grid** with the rows
of a different window, and ask whether the S4 forecast moves. A model with no
spatial path cannot move at all. The temporal-only model moved by **4.45 mm** —
it should have moved by exactly zero.

The cause was a reshape. In the temporal encoder I moved a permuted
(non-contiguous) tensor into a new shape with `reshape`; PyTorch found a
*view* whose strides described a **different element mapping**, so the
"per-place" encoder was in fact mixing every place with every other place.
The temporal-only baseline was secretly a full-mixing model — which is
exactly why it beat the graph models: it already had a spatial path, and
adding a second one only added parameters.

The fix is to split only the leading dimension of a contiguous tensor (always
a valid view) and to make every other copy explicit
(`scripts/river_models.py`, with `scripts/test_models.py` as a regression
test: *the encoder must be place-local* and *the temporal model must not be
able to see another place*).

**This is the part of the investigation I would put in the report**: a
structure claim ("this model has no spatial path") is only worth making if you
*test* it. The companion notebook makes exactly this point with reach maps;
the same discipline caught a bug that would have inverted the headline result.

### C3. The corrected comparison

With the leak fixed, the spatial path does what it should. Identical data,
identical head, identical budget, 3 seeds (mean ± sd of the test MAE in
metres over the 24 outputs of a one-hour horizon):

| architecture | test MAE (m) | S1 | S2 | S3 | S4 | flood-window MAE | params |
|---|---|---|---|---|---|---|---|
| joint ST convolution (row index) | **0.0249** ± 0.0004 | 0.0136 | 0.0421 | 0.0359 | 0.0080 | 0.0446 | 203,850 |
| graph message passing (same-time) | **0.0257** ± 0.0001 | 0.0146 | 0.0420 | 0.0362 | 0.0099 | 0.0454 | 72,010 |
| graph + delay attention | **0.0259** ± 0.0001 | 0.0157 | 0.0419 | 0.0362 | 0.0097 | 0.0454 | 121,930 |
| graph + learned delay kernel | **0.0259** ± 0.0001 | 0.0157 | 0.0420 | 0.0361 | 0.0099 | 0.0455 | 72,070 |
| global space-time attention | **0.0266** ± 0.0004 | 0.0168 | 0.0424 | 0.0362 | 0.0109 | 0.0458 | 189,514 |
| temporal encoder only (no spatial path) | **0.0308** ± 0.0001 | 0.0175 | 0.0448 | 0.0424 | 0.0187 | 0.0509 | 55,626 |
| ridge | **0.0372** ± 0.0000 | 0.0184 | 0.0583 | 0.0480 | 0.0239 | 0.0609 | 600 |
| persistence (no learning) | **0.0375** ± 0.0000 | 0.0180 | 0.0587 | 0.0489 | 0.0243 | 0.0606 | 0 |

**Reading.**

1. **A spatial path is worth 16 %.** Temporal-only 0.0308 → graph 0.0257.
   Every spatial family lands within 2 % of every other one, so *the
   existence* of a spatial path matters far more than *which* one.
2. **The gain is where the probe said it would be.** Compare the temporal-only
   and graph rows per station: S4 improves by **47 %** (0.0187 → 0.0099) and
   S3 by 15 %, while S2 barely moves (0.0448 → 0.0420) and S1 improves by
   17 % (the dam reaches S1 almost immediately). This is exactly the pattern
   the linear probe predicted, which is the strongest evidence that the
   mechanism is the one I think it is.
3. **The lattice assumption is *almost* free — and that is the trap.**
   The joint space-time convolution over the *row index* scores best
   (0.0249). But the rows are (Dam, S1, T1, S2, S3, S4), which is almost the
   chain order, so the 3-tap row kernel happens to connect the right
   neighbours; and it is 2.8× bigger (204 k parameters against 72 k).
   Crucially it is **physically wrong**: stacking the blocks lets S1's
   forecast depend on S2, S3 and S4 — information that cannot exist, because
   water does not flow upstream. The structural test below measures exactly
   that leak.
4. **Global attention is the most expensive and the least accurate** here
   (0.0266 with 190 k parameters). At the river's scale the quadratic term is
   not the problem; the problem is that a global operator has to *discover*
   the directed structure from data, and it can invent couplings that are not
   physical.

### C4. The structural test: which models can see what they should not?

Random-weight models (no training needed — the structure decides this).
Replace the downstream rows (S2, S3, S4) with those of another window and
measure how much the **S1** forecast moves. A physically honest model must
not move at all:

| model | change in the S1 forecast when the downstream rows are replaced |
|---|---|
| temporal only | 0.00 (no spatial path) |
| **graph message passing** | **0.00** |
| **graph + learned delay** | **0.00** |
| **graph + delay attention** | **0.00** |
| joint ST convolution (row index) | 0.14 (physically wrong) |
| global space-time attention | 0.04 (physically wrong) |

| model | change in the **S1** forecast when the downstream rows are replaced (mm) |
|---|---|
| temporal encoder only (no spatial path) | 0.00 |
| graph message passing (same-time) | 0.00 |
| graph + learned delay kernel | 0.00 |
| graph + delay attention | 0.00 |
| joint ST convolution (row index) | 0.35 |
| global space-time attention | 0.71 |

*This is in the test suite (`scripts/test_models.py`), so the property cannot
regress silently.  The trained-model version of the same test is in
`results/upstream_intervention.json`.*

The same intervention on the *trained* models, and the same test in the other
direction (replace the upstream rows; does the S4 forecast move?):

| model | |Δ forecast| at S4 (mm) | MAE vs the true S4 (mm) | MAE vs the swapped truth (mm) |
|---|---|---|---|
| temporal encoder only (no spatial path) | 0.00 | 18.51 | 16.14 |
| graph message passing (same-time) | 5.23 | 9.81 | 11.88 |
| graph + learned delay kernel | 4.06 | 9.78 | 11.62 |
| graph + delay attention | 4.83 | 9.12 | 11.27 |
| joint ST convolution (row index) | 4.64 | 8.43 | 10.86 |
| global space-time attention | 4.19 | 9.78 | 12.00 |

**Reading.** Two results in one table.

*In the forward direction* (the upstream rows are replaced): the temporal
model's S4 forecast moves by **0.00 mm** - it is blind, exactly as the
notebook's Part 1 predicts - while the graph family moves by 4-5 mm and its
error against the *swapped* truth gets worse (9.8 -> 11.9 mm), i.e. it
genuinely follows the upstream it was given. The graph family is using the
upstream; it is not just correlated with it.

*In the reverse direction* (the downstream rows are replaced): the graph
family's S1 forecast moves by **0.00 mm**, while the joint ST convolution
moves by 0.35 mm and global attention by 0.71 mm.

So the graph family is the only one that respects the direction of flow in
*both* directions. That is not a cosmetic difference: it is the difference
between a model of *a river* and a model of *a correlated set of numbers*. In
deployment it means a downstream instrument failure or a downstream regime
change cannot propagate backwards into an upstream forecast - and it is the
only family that can use the distances along the river that the authority
supplies.

---

## Phase D — ablations

| history window L (steps of 10 min) | test MAE (m) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| L=6 | **0.0257** | 0.0144 | 0.0438 | 0.0362 | 0.0083 |
| L=12 | **0.0262** | 0.0158 | 0.0433 | 0.0359 | 0.0099 |
| L=18 | **0.0262** | 0.0160 | 0.0430 | 0.0359 | 0.0101 |
| L=24 | **0.0264** | 0.0160 | 0.0438 | 0.0359 | 0.0100 |
| L=36 | **0.0263** | 0.0163 | 0.0424 | 0.0360 | 0.0104 |
| L=48 | **0.0262** | 0.0163 | 0.0420 | 0.0360 | 0.0106 |

**Reading.** The model is **insensitive to the history window**: 0.0257 m at
L = 6 (1 hour) and 0.0264 m at L = 24 (4 hours) — a 2.7 % spread over a
factor of eight, and the *shortest* window is (insignificantly) the best.
Two consequences:

* **L = 24 is not a tuned hyper-parameter.** It is derived from the physics
  (the slowest dam→S4 transit, ≈ 18 steps, plus the 6-step horizon), and the
  ablation says that choosing it costs nothing. If the authority's real
  transit is longer than I assume, the window must grow; if it is shorter,
  the model does not care.
* **Why the model does not need more history here.** The useful upstream
  signal is the upstream's *current* reading (the lead), not the wave still
  in transit, and the level's own recent history is a sufficient summary of
  its state. My simulator's storages respond in about one step, so the
  history of the *level* is nearly Markov; a real reach that stores water for
  hours is a different regime, and the physical argument for a long window
  is the one to keep.

| spatial message-passing rounds | test MAE (m) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| rounds=0 | **0.0309** | 0.0176 | 0.0447 | 0.0424 | 0.0190 |
| rounds=1 | **0.0255** | 0.0153 | 0.0420 | 0.0361 | 0.0088 |
| rounds=2 | **0.0259** | 0.0157 | 0.0418 | 0.0362 | 0.0099 |
| rounds=3 | **0.0257** | 0.0157 | 0.0410 | 0.0361 | 0.0098 |
| rounds=4 | **0.0263** | 0.0159 | 0.0426 | 0.0363 | 0.0102 |
| rounds=5 | **0.0262** | 0.0157 | 0.0430 | 0.0362 | 0.0100 |
| rounds=6 | **0.0264** | 0.0160 | 0.0430 | 0.0363 | 0.0103 |

**Reading - and an important refinement of the tutorial's advice.** The
tutorial says "pass it downstream, 4 rounds along the river graph", because
the dam is 4 hops from S4. The *reach* argument is right, and I keep 4 rounds.
But the *accuracy* argument is different: **one round already captures almost
all of the benefit** (0.0255 m), and every extra round makes the score
slightly worse (0.0263 m at 4 rounds, 0.0264 at 6).

The per-station column explains why. With **one** round - each station sees
only its immediate upstream - S4's error falls from 0.0190 m (no spatial path)
to **0.0088 m**, a 54 % improvement, because S3 *is* S4's upstream. The extra
rounds only add the more distant and less useful stations (S2, S1, the dam) to
S4's state, which costs a little variance.

So the design decision is: **take the rounds the reach requires, and no
more.** Four rounds are kept because the reach is a structural property that
my simulator cannot certify for the real river - in a flood, when the travel
times compress, the dam's release can reach S4 inside the forecast hour, and
only a 4-round path can carry it. Paying 3 % of the score for that insurance
is a good trade, and the ablation quantifies the premium.

| input variant | test MAE (m) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| full | **0.0268** | 0.0160 | 0.0439 | 0.0363 | 0.0108 |
| no rain | **0.0297** | 0.0162 | 0.0526 | 0.0395 | 0.0105 |
| no dam row | **0.0274** | 0.0177 | 0.0430 | 0.0368 | 0.0119 |
| no T1 row | **0.0273** | 0.0163 | 0.0456 | 0.0364 | 0.0111 |
| no announced plan | **0.0270** | 0.0171 | 0.0438 | 0.0363 | 0.0107 |
| no dam, no plan | **0.0274** | 0.0177 | 0.0430 | 0.0368 | 0.0119 |
| levels only (no rain/dam/plan) | **0.0300** | 0.0174 | 0.0525 | 0.0395 | 0.0105 |

**Reading - every source hurts exactly the station it should, and by about the
right amount.** This is the cleanest confirmation in the whole study that the
data specification is right:

| what was removed | whose error grows | by how much | why |
|---|---|---|---|
| rain (R1, R2) | S2 | 0.0439 -> 0.0526 (**+20 %**) | R1's catchment is S2 |
| rain (R1, R2) | S3 | 0.0363 -> 0.0395 (+9 %) | R2's catchment is the unmonitored stream above S3 |
| the Dam row | S1 | 0.0160 -> 0.0177 (**+11 %**) | the release reaches S1 within the hour |
| the announced plan | S1 | 0.0160 -> 0.0171 (+7 %) | the plan is the only *future* information about the dam |
| the T1 row | S2 | 0.0439 -> 0.0456 (+4 %) | the monitored tributary joins at S2 |
| everything except the levels | all | 0.0268 -> 0.0300 (+12 %) | |

Two things follow for the report.

* The **data specification can be defended station by station**, not as a
  whole: each source is load-bearing exactly where the hydrology says it
  should be, and nowhere else.
* The **announced plan is worth something but little** (7 % of S1's error),
  and it is the only *future* input in the model - so it must be treated as a
  hint, never as a constraint (assumption A6).

| training objective | test MAE (m) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| mse | **0.0263** | 0.0159 | 0.0426 | 0.0363 | 0.0102 |
| mae | **0.0225** | 0.0120 | 0.0367 | 0.0362 | 0.0052 |
| huber | **0.0247** | 0.0136 | 0.0414 | 0.0363 | 0.0077 |
| wmse | **0.0263** | 0.0163 | 0.0417 | 0.0362 | 0.0110 |

With the RMSE and the flood-window error alongside:

| objective | test MAE | test RMSE | flood-window MAE | flood-window RMSE |
|---|---|---|---|---|
| MSE in metres | 0.0263 | **0.0485** | 0.0456 | **0.0901** |
| MAE in metres | **0.0225** | 0.0608 | 0.0456 | 0.0969 |
| Huber (delta 0.2 m) | 0.0247 | 0.0598 | **0.0449** | 0.0931 |
| flood-weighted MSE | 0.0263 | 0.0487 | 0.0457 | 0.0906 |

**Reading - and this contradicts the tutorial's suggestion, so I checked it
twice.** Training with **MAE** improves the MAE metric by 14 % (0.0263 ->
0.0225) and improves *every* station, but it costs 25 % in RMSE, because MAE
does not punish the rare large miss. The tutorial's "weight larger for high
water" (the `wmse` row) **does not work**: it leaves both the overall MAE and
the flood-window MAE unchanged. My first attempt at a weighted loss was worse
still - it weighted by the *target*, which makes the objective non-smooth and
unstable (val MAE 0.16 against 0.11 for plain MSE). The version in the table
weights by the *input* (the level now), which is stable but still buys
nothing.

**The design decision this forces.** The training objective is a *statement of
which errors the service cares about*, and it is not a free choice:

* the committee's headline metric should be **MAE in metres** (it is what a
  warning system is judged on: how far off is the level I am told about),
  and then the model should be trained with **MAE**;
* if the committee instead cares about the *worst* misses - the events that
  drown a town - then the metric and the loss should both be **RMSE or
  Huber**, and the *reported* error should be quoted on flood windows;
* the flood weighting should be expressed in the **metric and the alert
  threshold**, not as a re-weighting of the loss.

This is the "match the head to the label" lesson from the course, applied to
the loss.

| degradation | test MAE (m) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| missing=0.0 | **0.0241** | 0.0126 | 0.0422 | 0.0354 | 0.0061 |
| missing=0.05 | **0.0272** | 0.0167 | 0.0436 | 0.0366 | 0.0120 |
| missing=0.15 | **0.0281** | 0.0169 | 0.0453 | 0.0374 | 0.0130 |
| late=0.05 | **0.0266** | 0.0157 | 0.0446 | 0.0369 | 0.0090 |
| late=0.12 | **0.0272** | 0.0159 | 0.0451 | 0.0381 | 0.0096 |
| plan_error=0.0 | **0.0257** | 0.0136 | 0.0430 | 0.0360 | 0.0100 |
| plan_error=3.5 | **0.0261** | 0.0150 | 0.0438 | 0.0359 | 0.0099 |
| plan_error=7.0 | **0.0264** | 0.0160 | 0.0438 | 0.0359 | 0.0100 |
| unmonitored=0.0 | **0.0278** | 0.0162 | 0.0434 | 0.0418 | 0.0099 |
| unmonitored=0.45 | **0.0264** | 0.0160 | 0.0438 | 0.0359 | 0.0100 |
| unmonitored=0.85 | **0.0288** | 0.0159 | 0.0438 | 0.0457 | 0.0099 |

**Reading - the degradation always lands where the physics says it should.**

* **Missing readings.** 0 % → 0.0241, 5 % → 0.0272, 15 % → 0.0281: a 13 %
  degradation for a 5 % loss of readings, and it is almost entirely at **S4**
  (0.0061 → 0.0120 → 0.0130, a factor of two). S4 is the station whose
  forecast depends most on the *upstream's current reading*, so a missing
  upstream reading hurts it most — the same "lead" story as everywhere else,
  and the reason the `observed?` flag has to be a first-class input rather
  than a fill-in rule.
* **Late readings.** 5 % late → 0.0266, 12 % → 0.0272: a stale reading is
  still informative, and the cost is small. The system can therefore simply
  carry the last value forward, flag it, and let the model discount it.
* **A plan that is not executed.** With the announced plan exact: 0.0257; with
  an operator error of ±3.5 m³/s: 0.0261; ±7 m³/s: 0.0264. The whole cost
  falls on **S1** (0.0136 → 0.0160), the only station the plan can speak
  about directly. The model treats the plan as a hint and is not derailed by
  it.
* **The unmonitored tributary.** With 45 % of the unmonitored inflow
  unexplained by R2: 0.0264; with 85 % unexplained: 0.0288, and the whole
  cost is at **S3** (0.0359 → 0.0457) — the confluence of the unmonitored
  stream. This is the clearest statement of where the irreducible error
  lives, and it is the reason the "request a gauge on that stream" item in
  the report is the one that matters.
  (The 0 %-unexplained row is *worse* than the 45 % row. That is a property
  of my simulator: with 0 % the unmonitored inflow is a deterministic but
  sharply non-linear function of R2, which is harder to fit than a
  slightly noisy version of the same thing. I report it rather than hide it,
  because the operational lesson is the same: the tributary is the dominant
  error term.)

### D0. What the model learned about the transport delay

The `graph + learned delay` model has one kernel per edge over 12 lags
(0…110 min). Reading the learned mixture off three trained seeds:

| edge | learned mean lag | learned mode | mass on lags 0–2 steps |
|---|---|---|---|
| Dam → S1 | 39.6 min | 20 min | 0.41 |
| S1 → S2 | 49.6 min | 20 min | 0.32 |
| T1 → S2 | 48.7 min | 0 min | 0.35 |
| S2 → S3 | 51.1 min | 40 min | 0.28 |
| S3 → S4 | 49.3 min | 10 min | 0.28 |

**Reading — and this is the most useful thing the delay kernel bought.**
The learned lags (≈ 50 min on every main-stem edge) are *shorter and much more
uniform* than the physical transit times I measured (S1→S2 30 min, S2→S3
70 min, S3→S4 80 min). The model is **not** trying to reproduce the travel
time. It is learning the lag that is *useful*, which is the recent upstream
reading — the **lead** — and not the wave that is still in transit.

That is a good property, and it is the strongest argument for making the lag
a learned quantity rather than a designed one: the travel time is **not
supplied** by the authority and it **changes** with the river's condition, so
a model that needed it would be building on sand. The learned-kernel model
does not need it: it finds the useful lag from the data, and its assumption
can be *audited* — we can print the kernel and ask whether it matches the
hydrology (it does not, and that is fine, because the model is not using the
arriving wave).

---

## Phase E — the design I will defend

### D1. The oracle bound: the residual error is *not* an upstream problem

The sharpest question a reviewer can ask is: "is the remaining error caused
by the model failing to use the upstream, or by information that is simply not
there?" I answered it by giving the model **oracle access to the upstream
station's true future levels** during the forecast hour and re-training
(`scripts/oracle_upstream.py`):

| target | baseline MAE (m) | with the upstream's **true future** revealed | change |
|---|---|---|---|
| S2 (reveal S1) | 0.04238 | 0.04260 | 0 % (no change) |
| S3 (reveal S2) | 0.03608 | 0.03609 | 0 % (no change) |
| S4 (reveal S3) | 0.00989 | 0.00869 | -12 % |

**Reading.** Even *perfect* knowledge of the upstream station's future buys
essentially nothing for S2 and S3. The remaining error is therefore **not**
an information-flow problem that a better architecture could fix: it is the
unmonitored tributary and the local catchment - inputs that are not observed
at all. The only place where the upstream's *future* still matters is S4,
where the wave from S3 is close enough to arrive inside the hour.

Together with C1 this completes the picture:

* the upstream's **present** is a *lead* and is worth 15-71 % (C1);
* the upstream's **future** is worth ~0 % (D1).

A one-hour forecast of this river is therefore mostly a statement about the
*current* state of the river, and the honest limit on accuracy is the
unmonitored tributary - which is exactly the kind of limitation a proposal
should name, because it is the one the authority can actually act on
(request a gauge on that stream).

### D2. Uncertainty: the ensemble spread is *not* an error bar

Three seeds of the chosen architecture, forecasts on the test split:

| quantity | value |
|---|---|
| single model, test MAE | 26.04 mm |
| ensemble mean, test MAE | 25.67 mm (a 1.4 % gain) |
| mean disagreement between the three seeds | 2.88 mm |
| mean absolute error of the ensemble | 25.67 mm |
| correlation between the disagreement and the error | **r = 0.25** |
| fraction of outputs inside ±2σ of the ensemble spread | 19 % |
| fraction inside ±3σ | 28 % |

**Reading.** The seed-to-seed disagreement is an order of magnitude smaller
than the error and is only weakly related to it, so it is a *bad* uncertainty
estimate — a 2σ band built from it would cover less than a fifth of the
outcomes. An ensemble buys a small accuracy gain and almost no information
about *when* the model is wrong. If the service must publish a confidence
range (and for flood warnings it should), the uncertainty has to come from
the model itself: a quantile or distributional head, or a model trained with
input dropout so that its own disagreement reflects data uncertainty.
This is an honest limitation of the design as specified, and an explicit
item for the next iteration.



### E1. Notation

* `i ∈ P = {Dam, S1, T1, S2, S3, S4}` — places (rows of the grid).
* `s ∈ {S1, S2, S3, S4}` — forecast stations.
* `t` — 10-minute steps; a forecast is issued at `t = 0` and covers `t = 1…6`.
* `L = 24` — history window (4 h), `H = 6` — horizon (1 h).
* `x_{i,t} ∈ R^4 = [level, observed?, rain, release]` — the grid cell.
* `y_{s,t}` — the true water level at station `s` at step `t`.
* `y^s_0` — the last **observed** level at `s` (carry-forward through gaps).
* `G = ((Dam,S1), (S1,S2), (T1,S2), (S2,S3), (S3,S4))` — the river graph
  (5 directed edges; `Dam→S4` is 4 hops).
* `d ∈ R^d` — the width of every learned representation.

### E2. The three blocks and their equations

**(1) Grid assembly** (no learning).

```
X ∈ R^{30 × 6 × 4},  X[t, i, :] = [level, observed?, rain, release]
```
* rows `i` = places; columns `t` = 24 observed steps + 6 forecast steps;
* `X[t, i, 1] = 1` iff the reading exists (a dropout sets level to 0 **and**
  the flag to 0, so a missing value can never be mistaken for a real zero);
* `X[t, S2, 2] = R1` and `X[t, S3, 2] = R2` (tutorial assumption);
* `X[t, Dam, 3] = release` for `t ≤ 0` and `= announced plan` for `t = 1…6`;
* every level cell of the horizon is empty and flagged 0.

**(2) Per-place temporal encoder** `f_time` — shared dilated causal conv over
time, plus a learned place embedding:

```
z_{i,t} = E_i + W_e x_{i,t}                                     (stem)
z_{i,t} ← z_{i,t} + φ( LN( W_ℓ ⊛_d z_{i,\cdot} ) )              (ℓ = 1..4, d = 1,2,4,8)
```
Weights are shared across places; `E_i` lets the same kernel behave differently
at the dam and at a gauged station. The kernel is symmetric in time, which is
safe because the only columns to the right of the origin hold the announced
plan. Output `Z ∈ R^{B × 30 × 6 × d}`.

**(3) Graph space-time mixing** `f_graph` — `R = 4` rounds along `G`; each edge
carries a **learned delay kernel** `α_{e} ∈ Δ^{W}` over `W = 12` lags
(0…110 min):

```
for round r = 1..R:
    for each edge e = (j → i):
        m_{i,t} = Σ_{τ=0}^{W-1} α^{(r)}_e(τ) · W_r z_{j,t-τ}        (learned α)
    z_{i,t} ← z_{i,t} + m_{i,t} + b_e
```
`α` is a learned softmax over lags: **the transport time is a parameter of the
model, not a constant of the design.** `R = 4` is forced by the graph: the dam
is 4 hops from S4, and each round moves information exactly one hop.

**(4) Read-out and forecast head** `g`:

```
h_s = z_{s,0} ∈ R^d          (the state at the forecast origin)
ŷ_{s,k} = y^s_0 + scale_s · g_k(h_s),     k = 1…6
```
with `g` a 2-layer MLP. The residual on the last observed level keeps the
numerics well-scaled (NLinear-style) **without destroying the absolute level**,
because the raw level is still an input to the encoder — which matters for
flood thresholds.

**(5) Objective**

```
L = (1/24) Σ_{s,k} w_{s,k} (ŷ_{s,k} − y_{s,k})²        [metres]
w_{s,k} = 1 + 3·1[ y^s_0 > q90(s) ]                      (flood-risk samples weigh more)
```
The weight is a function of the **input** (the level now), not of the target,
so the objective stays smooth. (An earlier version weighted by the *target*;
it was unstable — see the diagnostic in Phase D.)

### E3. Why this and not something else

| decision | alternative | evidence |
|---|---|---|
| temporal encoder per place + graph mixing | one big space-time convolution | the river is a **graph**, not a lattice: "row 2 and row 4 are neighbours" is false (T1 sits between S1 and S2 in the row order but is a different kind of place) |
| learned delay kernel per edge | fixed lag / same-time mixing | travel times are **not constant** (flood halves them) and the same-time mixing has to *discover* the lag from data |
| directed edges, 4 rounds | global attention | the physics *is* the adjacency; global attention can create impossible couplings and must learn the structure from data |
| residual head on the last level | predict the absolute level | level offsets differ per station and change over the record; the residual also makes the missing-data case well defined |
| learned delay kernel per edge | plain same-time mixing | same accuracy (0.0259 vs 0.0257) but the assumption becomes *auditable* (D0) and the model no longer needs the travel time, which is unknown and variable |
| 4 rounds | 1-2 rounds | the dam is 4 hops from S4; fewer rounds cannot carry its release to S4 at all (C3) |
| **MAE in metres** as the training loss (not MSE) | the committee's metric should be MAE | MAE training improves the MAE metric by 14 % at every station; the cost is 25 % worse RMSE, which is a *choice* about which errors matter (Phase D) |
| flood weighting of the loss | weighting the loss by high water | it buys nothing (Phase D): express the flood concern in the *metric* and the *alert threshold* instead |

### E4. The assumption list (A1…A16) — facts vs assumptions

Facts taken from the handout: 10-minute sampling; four main stations plus a
monitored tributary; two rain gauges with stated locations; a reservoir with
recorded and announced releases; station locations, connectivity and distances;
travel times and unmonitored inflows are **not** supplied; future rainfall and
future levels are **not** available at issuance.

Everything in the table below is *my* assumption.

| # | I assume… | because… | so in my design… |
|---|---|---|---|
| A1 | a station reports every 10 min unless flagged missing | the handout says "recorded every ten minutes" | one column per 10 min; `observed?` channel |
| A2 | a missing reading is unknown, not zero | a zero level is physically impossible | level set to 0 **and** flag set to 0; the encoder can tell them apart |
| A3 | a late reading arrives 1–3 steps late | reporting delays are unspecified | the grid carries the stale value; the system re-issues when it arrives |
| A4 | R1 drains into the S2 catchment, R2 into the unmonitored stream | the tutorial states this; R2 sits above the unmonitored stream | rain is folded into the S2 and S3 rows |
| A5 | the announced plan is valid for the whole horizon at issuance | "schedules announced before each forecast is issued" | the plan occupies the Dam row's horizon columns |
| A6 | the plan may not be executed | the handout warns the dam "may not do exactly what it planned" | the plan is an input, not a constraint; past actual releases correct it |
| A7 | travel times are flow-dependent and shorten in flood | the handout says travel times change | a **learned lag kernel** per edge instead of a fixed delay |
| A8 | `L = 24` (4 h) is needed | Dam→S4 is 4 hops ≈ 3 h at calm (Phase B) | 24 observed columns; the slowest upstream influence path fits. The ablation (D, lookback) shows the model is insensitive to L between 1 h and 8 h, so this conservative choice costs nothing |
| A9 | the datum of each gauge is stable | no datum information is supplied | predict the *change* from the last observed level, keep the raw level as an input |
| A10 | only part of the unmonitored tributary's inflow is explained by R2 | its catchment is unknown | R2's rain is an input proxy; the oracle experiment (D1) shows the residual error is *not* an upstream problem, so the tributary is the binding uncertainty |
| A11 | water does not flow upstream | the illustration shows directed flow | edges are directed; no upstream station reads a downstream one |
| A12 | the record is at least a year of 10-min data | record length is unspecified | ~52 k windows, but only O(10²) independent flood events → **small model, heavy inductive bias** |
| A13 | noise ≈ 1–2 cm; 1–2 % of readings missing | unspecified | the encoder sees the flag; training injects dropouts |
| A14 | the release is recorded exactly | it is an operational quantity | the Dam row's release channel is clean |
| A15 | all four stations are forecast jointly | one service, one horizon | one shared model, one head, per-station normalisation |
| A16 | the forecast is re-issued every 10 min and is *not* re-issued when a late reading arrives | the service cadence is fixed | the last value stands until the next issuance; the late reading enters the next window |

### E5. What the evidence does *not* settle

Honesty about the limits of the study, because a proposal that overclaims is
worse than one that says "here is what I do not know":

1. **The simulator is not the river.** It reproduces the *structure* the
   handout describes (directed flow, one monitored and one unmonitored
   tributary, two gauges, a reservoir with an announced plan, flow-dependent
   travel times, missing and late readings) but it is a kinematic-wave model
   with one rating curve per node. Real catchments have hysteresis, storage
   that depends on the history, and local inflows at every station. The
   *ranking* of the architectures is the claim; the absolute error numbers
   are not transferable.
2. **The record length is unknown.** A12 assumes at least a year. The
   parameter-to-example ratio in the *real* deployment is the binding
   constraint, and it is much tighter than in my simulation: a year of
   10-minute data is ~52 000 overlapping windows but only a few dozen
   independent flood events. I would expect the ranking to hold and the
   absolute numbers to be much worse.
3. **The true travel times are unknown.** The handout says the stations are
   unevenly spaced and the travel times are not supplied. My 24-step window
   is justified by a *measured* dam→S4 transit of ~18 steps in my
   simulator; the real number must come from the authority, and if it is
   longer than ~18 steps the window must grow (the architecture's reach is
   not the binding constraint — see Figure B3).
4. **The unmonitored tributary is the irreducible part.** I can measure how
   much of S3's/S4's error survives when R2's rain is known perfectly, but I
   cannot measure the *true* error without the true inflow. The honest
   statement is a bound, not a number.
5. **One horizon, one river.** Everything is measured at a one-hour horizon
   on a single catchment. The relative value of the upstream path depends on
   the horizon (it grows with the horizon) and on the geometry.

---

## Phase F — a map from these findings to the six required headings

The handout asks for six headings in a fixed order. Here is which part of this
investigation answers which, so the report can be written from the evidence
rather than from opinion.

| heading | what it needs | what I have |
|---|---|---|
| **1. Architecture diagram** | data sources, blocks, source combination, forecasts, tensor dimensions | Figure B5 (architecture), Figure B1 (the grid), the dimensions in E2; the reach argument in B3/Figure B3 |
| **2. Assumptions** | A1…A16 with "because" and "so in my design" | the table in E4; the facts/assumptions split at the top of E4 |
| **3. Task and data specification** | issuance time, history window, horizon, outputs, one table of every input/output with dimensions, axes, coverage, and the alignment rules | E1, the grid of B3, the alignment in E2(1); the *justification* of the window is in Phase D (lookback) and Phase B (travel times) |
| **4. Computations and justification** | equations for every block, the forecast map, the objective; how upstream, tributaries, rain and releases influence each forecast; physical vs learned | E2 (all the equations), E3 (the choice table), C1 (which source matters for which station), the learned-delay analysis (D0) and the oracle bound (D1) |
| **5. Operation and exceptional conditions** | (i) missing (ii) late (iii) announced releases (iv) scheduled ≠ actual; feasibility of the 10-minute cycle | the robustness ablation (Phase D) quantifies all four; the "grid" design answers (iii)–(iv); the cadence argument is in E2 and the cost note below |
| **6. Uncertainty, limitations, extra information** | the unmonitored tributary, changing travel times, each one's impact and response, and what to request | the oracle bound (D1) *is* the answer for the tributary: the residual error is not an upstream problem, it is the unobserved inflow; the learned-delay result (D0) and the travel-time calibration (B2) answer the changing delays; the ensemble result (D2) says the model's own spread is not an error bar |

**On the operational cadence.** The service issues a forecast every ten
minutes. The model is a forward pass of a 72 k-parameter network on a
`B × 30 × 6 × 4` grid; on the Lab PC's GPU a training run of 3 000 steps takes
~90 s, so an inference pass is far below the ten-minute budget by orders of
magnitude, and even on the laptop's CPU it is a fraction of a second. The
real operational risk is therefore **not** compute, it is data: a station that
stops reporting, a plan that is not announced on time, or a release regime the
model has not seen.

## Reproducing


Everything runs on CPU except the training sweeps, which were run on the Lab PC
(RTX 5060 Ti) with `scripts/sync_lab.sh` + `scripts/run_all_lab.sh`.

```bash
cd investigation/scripts

# Phase A - the companion notebook (CPU, no training)
python3 A1_notebook_figures.py

# Phase B - the design laboratory (CPU)
python3 river_sim.py                  # sanity check of the synthetic river
python3 travel_probe.py               # travel-time calibration
python3 reach_analysis.py             # the structural reach of the model

# Phase C/D - the experiments (Lab PC, GPU)
bash sync_lab.sh
ssh hussain@100.100.67.1 'cd ~/river_q1 && bash scripts/run_all_lab.sh'
python3 upstream_probe.py             # the information intervention
python3 oracle_upstream.py            # the oracle bound
python3 ensemble_uncertainty.py       # is the ensemble spread an error bar?
python3 learned_delays.py             # what delay did the model learn?
python3 diagnose_spatial.py           # the linear information probe

# figures and this document
python3 analysis_figures.py           # B1-B5
python3 result_figures.py             # C1-C10
python3 inject_tables.py              # tables into this file
python3 render_html.py                # FINDINGS.html

# the regression tests (locality of the encoder, no upstream leakage)
python3 test_models.py
```

## Files

| file | what it is |
|---|---|
| `FINDINGS.md` / `FINDINGS.html` | this document (the HTML has the figures inline) |
| `figures/A*.png` | the companion notebook, reproduced and quantified |
| `figures/B*.png` | the synthetic river, its travel times, the structural reach, the architecture |
| `figures/C*.png` | the architecture comparison and the ablations |
| `results/*.json` | every number quoted here |
| `scripts/river_sim.py` | the synthetic river |
| `scripts/river_data.py` | grid assembly, the observation model, the splits |
| `scripts/river_models.py` | the eight candidate architectures |
| `scripts/river_train.py` | training, losses, metrics |
| `scripts/run_experiments.py` | the experiment driver (arch / lookback / rounds / inputs / loss / robust / flood / final) |
| `scripts/test_models.py` | the structural regression tests (locality, no upstream leakage) |
| `scripts/upstream_probe.py`, `oracle_upstream.py`, `diagnose_spatial.py`, `ensemble_uncertainty.py`, `learned_delays.py`, `reach_analysis.py` | the diagnostic experiments quoted in Phase C and D |
