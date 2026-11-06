# World-model planning with the frozen ranking adapter

Open `notebooks/ranking_planning_colab.ipynb`, select GPU and Run all. This
evaluates the multi branch of the user's completed ranking run
`radial_multigoal_ranking_v1/20261106T195304628603Z`. No fitting or collection
for training occurs. Previous experiments and weights remain intact.

Default six goals: (50,42) original seen control, then withheld (46,38),
(40,18), (50,18), (56,10), (50,8). Eight new starts per goal: two at distance
6–10, three at10–18, three at18–30. This gives48 paired tasks and96 episodes.
Starts and goals lie in original R34 about(50,42), strictly right of wallx35;
start sampling uses x38..56/y8..55 and avoids fitting/reference coordinates
by at least0.5. Reference preparation also rejects image aliases.

Both costs use identical starts/goals/reset/CEM seeds and the same fixed
matched-v3 world model, IN1K encoder and pinned Wall renderer. Prepared fresh
goal tensors are shared exactly across methods. Raw terminal MSE is compared
with terminal MSE after applying the frozen ranking adapter to prediction and
goal. Histories remain raw; proprioception conditions dynamics but does not
define the objective. No oracle costs or region penalties.

Reuse the existing planner: horizon5 macros, execute5, population300,30
elites,10 CEM iterations, candidate batch64, elite_mean, min_std0.05,
unbounded components with primitive norm cap3.5, at most20 macros/100 actions.
Success is physical goal distance strictly below4.5. Later CEM candidate
values can diverge because the two costs choose different elites.

All primitive positions/actions are retained. Report departures from the
original fitting region/right side; do not discard those cases. Main success
includes every task, with in-region successes also shown. Summaries cover
each goal, distance range, all goals and withheld goals. Report paired wins,
losses, both-success/both-failure counts and final distances. Path/action
efficiency also uses the exact subset on which both methods succeed.

The notebook first times one paired task, then finishes all cases while reusing
that pair. Timing includes process/model startup and varies with task length.
Each episode has a signed completion record. The interaction journal counts
interrupted and repeated attempts. Rerun an interrupted evaluation cell in
the same runtime; changed GPU/software/configuration needs a fresh timestamp.

Separate `ranking_planning_v1` Drive output saves manifest, source/model/
runtime signatures, shared goal images/tokens, all episodes, numerical reports
and PNG/PDF figures. Figures show case coverage, goal-wise outcomes, paired
efficiency and one8-panel paired-trajectory figure per goal. Limits expand
to show excursions. Videos default off; optional goal markers are display-only.
Reports and plots need persistent source/results, but no GPU models or local
fitting caches. The source loader validates historical artifacts read-only
without weakening training-time source checks.

Local verification uses real CEM with tiny dynamics plus synthetic orchestration;
actual GPU planning results await the user's Colab run. Counts are descriptive
on correlated fixed tasks, not independent model replications.
