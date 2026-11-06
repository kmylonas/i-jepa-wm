# Frozen adapter, multiple equally shifted goals

User approved extending the existing goal-transfer experiment to more goals.
Preserve source adapter, historical results and all existing notebook pins.
Same worktree/branch and authorized Colab publication workflow; no fitting.

Default final distance adapter stage4/R34, source20261106T101715235909Z.
Original50,42 control plus8newgoals at displacement sqrt32≈5.656854, angles
0,45,...315degrees, coordinates rounded to6decimals. Includes46,38 from the
previous test. All remain within original fitting region and reset bounds.
New goals/stage/source editable. Reject empty, malformed, nonfinite, duplicate,
unchanged or outside-region goal lists before encoding/writing output.

Generalize existing evaluator from one new goal to N new goals while preserving
the flat single-goal API and repeated CLI --new-goal x y. Same immutable source
chain/backbone/renderer/precision checks; historical source fingerprint exception
and original-goal consistency control retained. Goals and ball coordinates
encoded once per unique coordinate; references may span multiple encoder batches.
Bounded memory: retain reference tokens and at most ceil(G/batch)*batch pending
tokens before scoring, stream remaining points. Handle more than8references.
Keep target scale34 and original raw calibration unchanged. Freeze/check adapter
and source bytes, exclude images matching any query reference from heldout.

Primary common cohort is the intersection across ALL goals: original fit disk
and no farther than sourceR from any goal. Heldout ball-position IDs must be
identical in every row, excluding original fitting coordinate/image exposure
and every reference coordinate/image. No per-goal relaxed headline subsets.
Empty cohorts (e.g. a small-stage disk with widely spread goals) remain N/A,
never silently broadened or called perfect. Direction tests/ties/denominators
remain as before; goal orientation/equal-edge exclusion can change edge counts.
Per-goal local/full-domain metrics remain secondary and explicitly different.

Save goal image/token for every goal, full-right per-goal heatmaps and calibration
scatter, plus a summary scatter over goal offsets colored by heldout MAE and
uphill fraction. Mark original goal and N/A goals; no clipping. Summarize per-
goal raw/adapted distance MAE/RMSE, delta vs original control on same positions,
uphill/tie counts. All-goal summary with missing metrics must render safely.
The old control values may differ from the two-goal study because this common
cohort is stricter; compare only within the sweep. Do not treat goals as seeds.

Separate radial_goal_sweep_colab.ipynb and timestamped radial_goal_sweep_v1 root.
Default display summary + heatmap for all9goals; calibration files saved and
optional display switch. Source Drive path/weights unchanged, no old feature
cache needed. Dynamic artifacts/hashes and whole-source lineage guard reuse;
changed count/order/coordinates reject, missing figures regenerate without
encoding. Existing single-goal notebook unchanged and its pinned code retained.

Meaningful RED->GREEN tests for >batch references, global cohort, missing metrics,
validation and completed artifact recovery; legacy tests/full suite; synthetic
evaluator/report CLI/notebook reporting smoke; fresh final review; tested source
pin and verified push. Real GPU results and scientific transfer await user run.
