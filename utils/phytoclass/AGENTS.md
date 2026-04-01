# AGENTS.md — utils/phytoclass/

Implementation notes for the phytoclass PFT decomposition (Hayward et al. 2023).

## Module layout

| File | Purpose |
|------|---------|
| `__init__.py` | Public entry point (`run_phytoclass`). Orchestrates clustering → SA → NNLS. |
| `cluster.py` | `cluster_samples` — Ward's hierarchical clustering on pigment ratios |
| `annealing.py` | `simulated_annealing` — optimizes F matrix per cluster |
| `nnls_mf.py` | `nnls_factorize` — NNLS solve for C given S and F |
| `config.py` | Default F matrix and bounds loaders; `SDP_TO_INTERNAL` name mapping |

## Call flow

```
run_phytoclass(pigments_df)
  -> cluster_samples(S)          # Ward's linkage, one cluster per dense group
  -> [per cluster] simulated_annealing(S_cluster, F, bounds)
       -> [per iteration] nnls_factorize(S, F_candidate)
  -> stack C_all, return DataFrame
```

## Parallelism — critical

- `run_phytoclass` spawns a `ProcessPoolExecutor` across clusters by default (`n_jobs = min(n_clusters, cpu_count)`).
- **Always pass `n_jobs=1`** when calling `run_phytoclass` from inside an outer `ProcessPoolExecutor` worker (e.g., from `run_phytoclass.py`). Nested process pools deadlock on most systems.
- For debugging a single cluster, pass `n_jobs=1` to run sequentially.

## Seed behavior

Each cluster gets a deterministic seed derived as `seed + cluster_index`. This means the same `seed` arg produces identical results regardless of how many clusters the data splits into at a given threshold.

## Name mapping

`SDP_TO_INTERNAL` in `config.py` maps `run_sdp` output column names (e.g., `"T chla"`) to the internal names used in the F matrix (e.g., `"Tchla"`). If you add a new pigment to the SDP pipeline, add an entry here too.

## Bounds

The bounds CSV (`data/phytoclass/bounds.csv`) constrains individual F matrix entries during SA. Format: `class, pigment, min, max`. Zero-bounded entries prevent implausible pigment assignments.
