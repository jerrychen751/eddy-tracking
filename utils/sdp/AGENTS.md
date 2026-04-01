# AGENTS.md — utils/sdp/

Implementation notes for the Spectral Derivative Pigments (SDP) model (Kramer et al. 2022).

## Module layout

| File | Purpose |
|------|---------|
| `prediction.py` | Inference entry point (`run_sdp`). Called by `run_sdp.py`. |
| `preprocessing.py` | `preprocess_rrs_batch` — Rrs → second derivative of residuals |
| `physics.py` | GSM forward model, seawater optics (Zhang 2009), `get_rrs_residuals` |
| `ancillary.py` | SST/SSS dataset loaders |
| `model.py` | Training only (`rrsModelTrain`). **Not used during inference.** Pre-fitted coefficients live in `coefficients/`. |
| `training.py` | Batch training wrapper that calls `rrsModelTrain` per pigment |

## Inference path

`run_sdp()` in `prediction.py` does **not** call `model.py`. It loads pre-fitted coefficients from `coefficients/*.npy` and applies them directly. Only re-run `training.py` if you want to retrain from new HPLC data.

## Physics gotchas

- `betasw124_ZHH2009` requires scalar `S` and `Tc` (not arrays). It raises `NotImplementedError` for ndarray inputs.
- GSM inversion (`gsm_invert`) solves a 3-IOP optimization per spectrum using `scipy.optimize.fmin`. It is the main bottleneck in `get_rrs_residuals`.
- Wavelength indexing uses `np.searchsorted` — do not assume a fixed integer offset.

## Training gotchas (model.py)

- **Standardization**: validation spectra must use training mean/std, not their own. The data leakage fix is already applied; don't revert it.
- **Seed**: use the `seed` parameter (`np.random.default_rng(seed)`). Never call `np.random.seed()` globally.
- `mdl_pick_metric` must be one of `'MAE'`, `'R2'`, `'RMSE'`, `'avg'`, `'med'`. Anything else raises `ValueError`.
