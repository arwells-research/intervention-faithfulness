Method invariants (non-negotiable)
I1) Purely empirical, model-agnostic

The diagnostic uses only measured trials.

No simulation, no microscopic assumptions, no curve fitting required.

I2) The target is intervention faithfulness of a reduced state representation

We test whether a chosen mapping 
𝑠
𝑡
=
𝑅
(
𝑦
0
:
𝑡
)
s
t
	​

=R(y
0:t
	​

) is sufficient to support prediction under intervention.

The state can be “whatever the model interface currently uses” (state_* columns).

I3) What is being tested is distributional invariance, not determinism

The criterion is: for histories 
ℎ
1
,
ℎ
2
h
1
	​

,h
2
	​

 that collapse to the same reduced state, the conditional continuation distributions under the same intervention must match.

This defuses “stochasticity” objections: randomness is fine if the conditional distribution is invariant.

I4) Continuation fracture is a regime-dependent signature of representational collapse

Fracture is not “noise” or “non-Markovianity in general.”

It is the empirical signature that the reduction collapsed histories that are continuation-relevant under intervention.

I5) The primary output is operational, not interpretive

Faithfulness maps + safe envelope + certificate are meant to answer:

“When is my state good enough?”

“Where does it fail?”

“What minimal augmentation restores validity?”

Canonical data contract (what the core assumes)

Your collaborator’s plugin-layer proposal matches the method perfectly:

Required

trial_id

intervention_id (categorical or numeric)

outcome (scalar; vector later if needed)

state_* columns (the candidate reduced state)

Optional

history_* columns (full-history descriptors / engineered features / labels)

Invariant: the core operates on this canonical table. Everything else is adapters.

Metrics (what “fracture” means, in v0.1 terms)

You have two conceptions now; both are consistent with the original METHODS:

M1) Refinement fracture (state vs history refinement)

Interpretation: “how much additional predictive structure exists inside the collapsed state when you condition on a finer key.”

Compare 
𝑃
(
𝑦
∣
𝑠
,
𝐼
)
P(y∣s,I) vs 
𝑃
(
𝑦
∣
ℎ
,
𝐼
)
P(y∣h,I) where 
ℎ
h refines 
𝑠
s.

This matches your initial implementation style (state_key vs history_key refinement).

M2) Pairwise fracture (within-state pairwise divergence)

Interpretation: “do different history classes inside the same state yield different continuation distributions?”

For a given 
𝑠
,
𝐼
s,I, sample pairs of history classes 
ℎ
𝑖
,
ℎ
𝑗
⊂
𝐻
(
𝑠
)
h
i
	​

,h
j
	​

⊂H(s) and compute 
𝐷
(
𝑃
(
𝑦
∣
ℎ
𝑖
,
𝐼
)
 
∥
 
𝑃
(
𝑦
∣
ℎ
𝑗
,
𝐼
)
)
D(P(y∣h
i
	​

,I)∥P(y∣h
j
	​

,I)).

This matches your “pairwise fracture implementation cleanly” track and the n_pairwise_pairs knob.

Invariant: both are legal operationalizations of “continuation fracture,” and the paper can present one as primary and the other as robustness.

Recommendations (minimal completion) invariants
R1) Recommendations are repair suggestions, not causal explanations

They propose state augmentation candidates that reduce fracture.

They do not claim “this is the true microscopic state.”

R2) Two modes

Single: rank individual candidate features by fracture reduction 
Δ
𝐹
ΔF.

Greedy/sets: rank small sets of features that jointly reduce fracture (your rank_minimal_completion_sets path).

R3) A valid “negative control” behavior exists

In a faithful regime, the recommender should often return:

empty or near-zero deltas,

“no action needed” (or low-confidence suggestions).
This is part of the scientific contract that the method isn’t a fishing expedition.

Maps / envelope / certificate invariants
V1) Faithfulness maps are “boundary of validity” views

Grid over (x,y) axes (intervention strength, history depth/feature, etc.).

Color is fracture or normalized faithfulness.

V2) Safe envelope is a summary of the map, not a new metric

It reduces the 2D grid into human-actionable segments: safe / unsafe / uncertain.

“Uncertain” is explicitly underpowered (min_samples or NaN).

V3) Certificates must be auditable artifacts

Include hashes of trials table + diagnosis record.

Export bundle should contain:

diagnosis JSON

certificate JSON (curated payload)

optional PDF/HTML certificate

map image(s)

any metadata/config provenance

Phase structure for the paper (the A→B→C→D→E arc)

This is the minimum “section header skeleton” implied by your dialog:

Recognized failure mode (protocol dependence / regime dependence)

Method: invariance of conditional continuation distributions

Metric: continuation fracture + significance + sample warnings

Repair: minimal completion (single + greedy sets)

Operationalization: maps → safe envelope → certificate

Validation of the validator: negative control regime

This keeps nanowires as “canonical positive case” while remaining a general methods paper.