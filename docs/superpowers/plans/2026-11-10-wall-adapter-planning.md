# Multi-layout Wall adapter planning implementation plan

**Goal:** Compare fixed and multi-layout Band 8 cost adapters in online planning with one frozen multi-layout Wall v3 world model.

**Architecture:** A small runner validates the world model and adapter provenance, executes paired held-out cases with the existing latent-cost callback, resumes per-case Drive outputs, and computes paired summaries. A Colab notebook pins source and artifact checksums, runs a smoke case followed by 50 seeds on each of four held-out layouts, and displays saved summary and trajectory plots.

**Inputs:** `datasets/wall_matched_fixed_multi_v3/multi/world_model/best.pt`, its manifest and environment config, the frozen I-JEPA encoder, and `dotwall_layout_transfer_v1/20261110T125759955799Z/{fixed,multi}/band_08.pt`.

**Protocol:** Only the cost adapter changes. Use `cost=latent`, `cem_selection=best`, horizon 5, execute 5 macro actions per plan, 20 macro action budget, population 300, 30 elites, 10 iterations, action norm cap 3.5, and seeds 10000–10049 on each held-out layout. Save both outcomes, success and final distance by layout, seed-cluster paired bootstrap, and example trajectories. The same initial start and goal must be verified across arms.

### Task 1: Paired planning runner

- Write failing tests for paired-case alignment, the paired summary, and changed artifact identity on resume.
- Implement `wall_adapter_planning.py` using `plan_proprio.run_episode` and `adapter_goal_cost`.
- Verify targeted tests and compile the module.

### Task 2: Colab and plots

- Build `wall_multi_adapter_planning_colab.ipynb` with pinned Git checkout and artifact hashes, smoke and full stages, explicit inline figures, and Drive output links.
- Validate notebook schema and compile code cells; inspect the cells for intended paths, settings, and display calls.
- Commit only new files, push the branch, and verify the hosted notebook URL.
