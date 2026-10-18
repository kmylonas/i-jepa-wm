# Ball displacement versus I-JEPA patch similarity

Run the two cameras separately, keeping only one cache on disk at a time.
Each cache must contain `latents/episode_XXXX.npy` and
`trajectories/episode_XXXX.pt`. Its directory may be named `i_jepa_cache`,
`ijepa_cache`, or anything else; pass its path explicitly. Keep the result
directories **outside** the cache directory so deleting or replacing the
cache does not delete the results.

From the I-JEPA repository root, using the `pointmaze` Python environment,
first run the top-down cache:

```bash
/Users/kmylonas/miniforge3/envs/pointmaze/bin/python \
  -m src.mylonas_ijepa_wm.embedding_similarity \
  --topdown-cache /path/to/i_jepa_cache \
  --output-dir src/mylonas_ijepa_wm/debug/similarity_topdown \
  --max-episodes 100 \
  --pairs-per-bin 30 \
  --seed 42
```

After keeping `similarity_topdown/` and replacing the local cache with the
angled cache, run:

```bash
/Users/kmylonas/miniforge3/envs/pointmaze/bin/python \
  -m src.mylonas_ijepa_wm.embedding_similarity \
  --angled-cache /path/to/i_jepa_cache \
  --output-dir src/mylonas_ijepa_wm/debug/similarity_angled \
  --max-episodes 100 \
  --pairs-per-bin 30 \
  --seed 42
```

Finally, combine the saved results. The two latent caches are no longer
needed:

```bash
/Users/kmylonas/miniforge3/envs/pointmaze/bin/python \
  -m src.mylonas_ijepa_wm.embedding_similarity \
  --combine-results \
    src/mylonas_ijepa_wm/debug/similarity_topdown \
    src/mylonas_ijepa_wm/debug/similarity_angled \
  --output-dir src/mylonas_ijepa_wm/debug/similarity_combined
```

Use the same `--max-episodes`, `--pairs-per-bin`, and `--seed` in both runs.
The combiner requires the same episode IDs, frame pairs, and ball distances;
it stops with an error if they differ. This preserves a paired camera
comparison even though the two caches were never on disk together.

The script needs no GPU and does not load the I-JEPA encoder. Each camera run
selects at most one frame pair per episode and displacement bin, then samples
at most 30 pairs from different episodes for each bin. Pairs come from the
same episode and have the same maze and camera. Both endpoints are positions
actually observed in the environment. This is an observational comparison:
other visible elements may also change between widely separated frames.

Distance bins are `(0, 0.1]`, `(0.1, 0.2]`, `(0.2, 0.4]`, `(0.4, 0.8]`,
`(0.8, 1.6]`, and `(1.6, 3.2]` maze units. Empty bins remain empty. The
similarity for a pair is the mean of cosine similarities between the 256
patches at matching grid coordinates. The CSV also records same-patch mean
squared error for comparison with the world-model training objective.

Each result directory contains:

- `similarity.png`: individual pairs and per-bin median with interquartile range
- `pairs.csv`: pair IDs, frame indices, distances, cosine similarities, and MSE
- `summary.csv`: bin counts and cosine summary per camera
- `metadata.json`: paths or source results, sample seed, selected episode IDs,
  and settings

The script uses a small subset by default, so the main runtime is local file
reading. A full 21-frame latent file is about 13 MB. When running against a
cloud-synced directory, downloading files may take longer than the CPU work.
The local one-episode smoke output is in
`src/mylonas_ijepa_wm/debug/embedding_similarity_smoke/` and should not be
interpreted as a camera comparison.
