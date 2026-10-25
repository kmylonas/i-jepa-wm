# Matched-layout visual-input test

## Purpose

Determine whether the trained DotWall world model changes its collision prediction when the observed wall layout changes while proprioceptive state and actions remain identical. The existing collision probes vary starting positions across cases, so they cannot isolate the visual input.

## Test

Use predeclared matched comparisons on held-out layouts. Each comparison has two layouts, one shared starting position, and one shared sequence of five primitive actions. Simulate both layouts and predict the post-action state from each layout's I-JEPA tokens. Include doorway changes at wall x=26 and wall-position changes between x=26 and x=38; run each crossing direction. The simulator outcomes in a pair must differ. The world model sees only tokens, normalized proprioceptive state, and the same action vector; it receives no layout label.

Report each layout's actual and predicted final position, crossing decision, position error, and visual-token MSE, plus whether the model gets both contrasting outcomes correct. Keep the cases fixed even if predictions fail. No latent swapping or planning is part of this test.

## Figures and artifacts

Save one side-by-side PNG per matched comparison and a JSON summary to a user-selected output directory. Both panels use identical coordinate limits. Draw the current wall with its actual width and doorway opening, mark the three training wall x positions as faint guides, draw the true five-action ball trajectory, and mark the model's one-step predicted endpoint. Label these distinctly; never draw an invented predicted intermediate path. The figures should make the layout change and the predicted outcome inspectable without reading JSON.

## Constraints

Reuse the pinned DINO-WM Wall adapter, frozen I-JEPA encoder, and existing checkpoint/manifest validation. The command must work after a fresh Colab clone with the encoder and world-model checkpoint restored from Drive; it must not need cached training latents or trajectories. Document one Colab command and the limits of inference: this tests visual dependence for these controlled cases, not every possible map or planning run.
