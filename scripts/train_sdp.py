"""
Train the SDP pigments model and write coefficient CSVs.

Fits the Kramer et al. (2022) PCA + regression model for 13 pigments and saves per-pigment ensemble coefficients to `src/eddy_tracking/packages/sdp/coefficients/`, which `eddy_tracking.packages.sdp.prediction.run_sdp` then loads at inference time.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def train_rrs_model(
    RrsD: np.ndarray,
    hplc_i: np.ndarray,
    pft_index: str,
    n_permutations: int,
    max_pcs: int,
    k: int,
    mdl_pick_metric: str,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """
    Train PCA-based regression model for pigment prediction. One set of coefficients are generated for one pigment at a time.

    Uses a 75/25 train/validation split with k-fold CV within the training set. For each permutation, the mean k-fold coefficients are unstandardized and validated against the held-out 25%.

    Note: max_pcs must be <= 0.75 * (1 - 1/k) * n_samples.

    Args:
        RrsD: 2nd derivative of Rrs residuals, shape (n_samples, n_wavelengths).
        hplc_i: Pigment concentrations for a single pigment (ground truth), shape (n_samples,).
        pft_index: Constraint type. 'pigment' (>= 0), 'EOFs' (unconstrained), or 'compositions' (0-1).
        n_permutations: Number of random 75/25 train/validation splits.
        max_pcs: Maximum number of principal components to evaluate.
        k: Number of folds for cross-validation within each training split.
        mdl_pick_metric: Metric for selecting the optimal number of PCs. One of 'R2', 'RMSE', 'avg', 'med', or 'MAE' (McKinna et al. 2021).

    Returns:
        coefficients: Unstandardized regression coefficients, shape (n_wavelengths, n_permutations).
        intercepts: Unstandardized intercepts, shape (n_permutations,).
        summary_gofs: DataFrame with mean/std of R2, RMSE, percent error, bias, and MAE across permutations.
        all_gofs: Dict of per-permutation goodness-of-fit arrays.
    """

    if np.isnan(hplc_i).any():
        raise ValueError('hplc_i contains NaN values')
    if np.isnan(RrsD).any():
        raise ValueError('RrsD contains NaN values')

    if RrsD.shape[0] != hplc_i.shape[0]:
        raise ValueError(
            f'RrsD and hplc_i row count mismatch: {RrsD.shape[0]} vs {hplc_i.shape[0]}'
        )

    rng = np.random.default_rng(seed)

    # Model form: pigment = RrsD @ betas + alpha, with one (betas, alpha) pair per permutation.
    mean_betas_nonstd = np.zeros((RrsD.shape[1], n_permutations)) # (n_wavelengths, n_permutations)
    mean_alphas_nonstd = np.zeros(n_permutations)

    R2s_final = np.zeros(n_permutations)
    RMSEs_final = np.zeros(n_permutations)
    pct_bias = np.zeros(n_permutations)
    pct_errors = np.zeros((n_permutations,len(hplc_i)-int(len(hplc_i) * 0.75))) # (n_permutations, n_validate)
    med_pct_error = np.zeros(n_permutations)
    avg_pct_error = np.zeros(n_permutations)
    CI_pct_error = np.zeros(n_permutations)
    std_pct_error = np.zeros(n_permutations)
    mae_final = np.zeros(n_permutations)

    for i in range(n_permutations):
        training_indices = rng.permutation(len(hplc_i))[:int(len(hplc_i) * 0.75)]

        pigs_training = hplc_i[training_indices]
        RrsD_training = RrsD[training_indices,:]

        pigs_validate = hplc_i
        pigs_validate = np.delete(pigs_validate,training_indices)
        RrsD_validate = RrsD
        RrsD_validate = np.delete(RrsD_validate, training_indices, axis=0)

        pig_len = len(pigs_training)

        rand_ns = rng.permutation(pig_len)

        CV_indices = np.full((k, int(np.ceil(len(pigs_training) / k))), np.nan)
        n_leftovers = pig_len % k
        counter_start = n_leftovers
        counter_end = n_leftovers + pig_len // k
        for j in range(k):
            CV_indices[j, :(pig_len // k)] = rand_ns[counter_start:counter_end]
            counter_start += pig_len // k
            counter_end += pig_len // k

        # Only pig_len % k folds get a leftover, so the last slot of the rest stays NaN and the fold loop below drops it.
        leftovers = rand_ns[:n_leftovers]
        na_array = np.full((k - len(leftovers)), np.nan)
        leftovers = np.concatenate([leftovers, na_array]) # (n_leftovers,) + (k - n_leftovers,) -> (k,)

        CV_indices[:, CV_indices.shape[1]-1] = leftovers

        n_modes_to_use = np.zeros(k, dtype=int)
        betas = np.zeros((RrsD_training.shape[1], k))
        alpha = np.zeros(k)
        CV_R2s = np.zeros(k)
        CV_RMSEs = np.zeros(k)

        for j in range(k):
            these_CV_indices = CV_indices[j, :]
            these_CV_indices = these_CV_indices[~np.isnan(these_CV_indices)].astype(int)
            CV_valid_pigs = pigs_training[these_CV_indices]
            CV_valid_spec = RrsD_training[these_CV_indices, :]
            CV_train_pigs = np.delete(pigs_training, these_CV_indices, axis=0)
            CV_train_spec = np.delete(RrsD_training, these_CV_indices, axis=0)

            train_mean = np.mean(CV_train_spec, axis=0)
            train_std = np.std(CV_train_spec, axis=0)
            CV_train_spec = (CV_train_spec - train_mean) / train_std
            CV_valid_spec = (CV_valid_spec - train_mean) / train_std

            # The standardization above already centered the spectra, so the SVD needs no further centering.
            U, S, VT = np.linalg.svd(CV_train_spec, full_matrices=False)

            CV_EOFs_train = VT[:max_pcs].T # (max_pcs, n_wavelengths) -> (n_wavelengths, max_pcs)
            CV_AFs_train = U[:, :max_pcs] * S[:max_pcs] # (n_train, max_pcs) * (max_pcs,) -> (n_train, max_pcs)

            n_val = len(CV_valid_pigs)
            percent_errors = np.zeros((n_val, CV_AFs_train.shape[1]))
            all_bias = np.zeros((n_val, CV_AFs_train.shape[1]))
            mean_percent_error = np.zeros(CV_AFs_train.shape[1])
            median_percent_error = np.zeros(CV_AFs_train.shape[1])
            bias = np.zeros(CV_AFs_train.shape[1])
            MAE = np.zeros(CV_AFs_train.shape[1])
            R2s = np.zeros(CV_AFs_train.shape[1])
            RMSEs = np.zeros(CV_AFs_train.shape[1])
            ensemble = np.zeros(CV_AFs_train.shape[1])
            pearson = np.zeros(CV_AFs_train.shape[1])

            for n_pcs in range(1, CV_AFs_train.shape[1]+1):
                lin_model = LinearRegression()
                lin_model.fit(CV_AFs_train[:, :n_pcs], CV_train_pigs)

                this_alpha = lin_model.intercept_
                these_betas = lin_model.coef_

                spec_betas = CV_EOFs_train[:, :n_pcs] @ these_betas # (n_wavelengths, n_pcs) @ (n_pcs,) -> (n_wavelengths,)

                CV_modeled_pigs = CV_valid_spec @ spec_betas + this_alpha # (n_val, n_wavelengths) @ (n_wavelengths,) -> (n_val,)

                if pft_index == 'pigment':
                    CV_modeled_pigs[CV_modeled_pigs < 0] = 0
                elif pft_index == 'compositions':
                    CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
                elif pft_index == 'EOFs':
                    pass  # No constraints applied

                # A CV_valid_pigs entry of zero (pigment below the detection limit) divides by zero here.
                percent_errors[:n_val, n_pcs-1] = ((CV_valid_pigs - CV_modeled_pigs) / CV_valid_pigs) * 100
                mean_percent_error[n_pcs-1] = np.mean(np.abs(percent_errors[:, n_pcs-1]))
                median_percent_error[n_pcs-1] = np.median(np.abs(percent_errors[:, n_pcs-1]))

                all_bias[:n_val, n_pcs-1] = CV_modeled_pigs - CV_valid_pigs
                bias[n_pcs-1] = np.mean(all_bias[:, n_pcs-1])
                MAE[n_pcs-1] = np.mean(np.abs(all_bias[:, n_pcs-1]))

                reg = LinearRegression()
                reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs) # (n_val,) -> (n_val, 1)

                R2s[n_pcs-1] = reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs) # (n_val,) -> (n_val, 1)
                RMSEs[n_pcs-1] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)
                pearson[n_pcs-1] = np.corrcoef(CV_modeled_pigs, CV_valid_pigs)[0, 1]

                ensemble[n_pcs-1] = (1 - R2s[n_pcs-1] + RMSEs[n_pcs-1]) / 100

            if mdl_pick_metric == 'MAE':
                n_modes_to_use[j] = np.argmin(MAE) + 1
            elif mdl_pick_metric == 'R2':
                n_modes_to_use[j] = np.argmax(R2s) + 1
            elif mdl_pick_metric == 'RMSE':
                n_modes_to_use[j] = np.argmin(RMSEs) + 1
            elif mdl_pick_metric == 'avg':
                n_modes_to_use[j] = np.argmin(mean_percent_error) + 1
            elif mdl_pick_metric == 'med':
                n_modes_to_use[j] = np.argmin(median_percent_error) + 1
            else:
                raise ValueError(
                    f"Unknown mdl_pick_metric: {mdl_pick_metric!r}. "
                    "Must be one of: 'MAE', 'R2', 'RMSE', 'avg', 'med'."
                )

            X_train = CV_AFs_train[:, :n_modes_to_use[j]]
            y_train = CV_train_pigs

            lin_mdl = LinearRegression()
            lin_mdl.fit(X_train, y_train)

            alpha[j] = lin_mdl.intercept_
            these_betas = lin_mdl.coef_

            betas[:, j] = CV_EOFs_train[:, :n_modes_to_use[j]] @ these_betas # (n_wavelengths, n_modes) @ (n_modes,) -> (n_wavelengths,)

            CV_modeled_pigs = CV_valid_spec @ betas[:, j] + alpha[j] # (n_val, n_wavelengths) @ (n_wavelengths,) -> (n_val,)

            if pft_index == 'pigment':
                CV_modeled_pigs[CV_modeled_pigs < 0] = 0
            elif pft_index == 'compositions':
                CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
            elif pft_index == 'EOFs':
                pass # No constraints applied

            CV_reg = LinearRegression()
            CV_reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs) # (n_val,) -> (n_val, 1)
            CV_R2s[j] = CV_reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs) # (n_val,) -> (n_val, 1)
            CV_RMSEs[j] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)

        mean_betas = np.mean(betas, axis=1) # (n_wavelengths, k) -> (n_wavelengths,)
        mean_alphas = np.mean(alpha)
        std_betas = np.std(betas, axis=1) # (n_wavelengths, k) -> (n_wavelengths,)
        std_alphas = np.std(alpha)

        spec_std = np.std(RrsD_training, axis=0, ddof=0)  # MATLAB default is population std (ddof=0)
        spec_mean = np.mean(RrsD_training, axis=0) # (n_train, n_wavelengths) -> (n_wavelengths,)

        mean_betas_nonstd[:, i] = mean_betas / spec_std
        mean_alphas_nonstd[i] = mean_alphas - np.sum(mean_betas * (spec_mean / spec_std))

        modeled_pigs = RrsD_validate @ mean_betas_nonstd[:,i] + mean_alphas_nonstd[i] # (n_validate, n_wavelengths) @ (n_wavelengths,) -> (n_validate,)

        if pft_index == 'pigment':
            modeled_pigs[modeled_pigs < 0] = 0
        elif pft_index == 'compositions':
            modeled_pigs = np.clip(modeled_pigs, 0, 1)
        elif pft_index == 'EOFs':
            pass # No constraints applied

        model = LinearRegression().fit(modeled_pigs.reshape(-1, 1), pigs_validate) # (n_validate,) -> (n_validate, 1)

        R2s_final[i] = model.score(modeled_pigs.reshape(-1, 1), pigs_validate) # (n_validate,) -> (n_validate, 1)
        RMSEs_final[i] = root_mean_squared_error(pigs_validate, model.predict(modeled_pigs.reshape(-1, 1))) # (n_validate,) -> (n_validate, 1)

        # A zero pigment value would make the percent errors below divide by zero.
        pigs_validate_safe = np.where(pigs_validate == 0, 1e-4, pigs_validate)

        pct_bias[i] = np.mean(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        pct_errors[i, :] = np.abs(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        med_pct_error[i] = np.median(pct_errors[i, :])
        avg_pct_error[i] = np.mean(pct_errors[i, :])

        sort_pct_errors = np.sort(pct_errors[i, :])
        CI_pct_error[i] = sort_pct_errors[int(np.ceil(0.95 * len(sort_pct_errors))) - 1]

        std_pct_error[i] = np.std(pct_errors[i, :])

        mae_final[i] = np.mean(np.abs(modeled_pigs - pigs_validate))

    coefficients = mean_betas_nonstd
    intercepts = mean_alphas_nonstd

    summary_gofs = [
        np.mean(R2s_final), np.std(R2s_final),
        np.mean(RMSEs_final), np.std(RMSEs_final),
        np.mean(avg_pct_error), np.std(avg_pct_error),
        np.mean(med_pct_error), np.std(med_pct_error),
        np.mean(pct_bias), np.std(pct_bias),
        np.mean(mae_final), np.std(mae_final)
    ]

    summary_gofs_df = pd.DataFrame([summary_gofs], columns=[  # pyright: ignore[reportArgumentType]
        'Mean_R2', 'SD_R2',
        'Mean_RMSE', 'SD_RMSE',
        'Mean_mean_pct_error', 'SD_mean_pct_error',
        'Mean_median_pct_error', 'SD_median_pct_error',
        'Mean_pct_bias', 'SD_pct_bias',
        'Mean_MAE', 'SD_MAE'
    ])

    all_gofs = {
        'R2s': R2s_final,
        'RMSEs': RMSEs_final,
        'mean_pct_error': avg_pct_error,
        'median_pct_error': med_pct_error,
        'pct_bias': pct_bias,
        'all_pct_errors': pct_errors,
        'all_mae': mae_final
    }

    return coefficients, intercepts, summary_gofs_df, all_gofs


def train_model(RrsD: np.ndarray | pd.DataFrame, hplc: np.ndarray) -> None:
    """
    Train SDP model on Rrs residuals and HPLC data.

    Takes 2nd derivative of Rrs residuals, trains model for all 13 pigments using 100 permutations, 30 max PCs, 5-fold CV, MAE metric. Saves A (wavelength coefficients, shape [n_wl, 100]) and C (intercepts, shape [100]) to CSV files in src/eddy_tracking/packages/sdp/coefficients/.
    """

    diffD2 = np.diff(RrsD, 2, axis=0) # (n_wavelengths, n_samples) -> (n_wavelengths - 2, n_samples)

    n_permutations = 100
    max_pcs = 30
    mdl_pick_metric = 'MAE' # one of 'MAE', 'R2', 'RMSE', 'avg', 'med'; anything else raises ValueError
    k = 5
    pft_index = 'pigment'

    pigs2mdl = np.array([
        'Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
        'Neo','Viola','Fuco','Chlc12','Chlc3','Perid'
    ])

    # column order of the HPLC dataset passed in as hplc
    hplc_vars = [
        'Tchla','Tchlb','Tchlc','ABcaro','ButFuco','HexFuco','Allo','Diadino','Diato',
        'Fuco','Perid','Zea','MVchla','DVchla','Chllide','MVchlb','DVchlb','Chlc12','Chlc3',
        'Lut','Neo','Viola','Phytin','Phide','Pras'
    ]

    start = time.time()

    # Coefficient output directory lives in the inference package so run_sdp can load them
    output_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "eddy_tracking"
        / "packages"
        / "sdp"
        / "coefficients"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(pigs2mdl)):
        pigment = pigs2mdl[i]
        pigment_idx = hplc_vars.index(pigment)
        hplc_i = hplc[:, pigment_idx]
        # diffD2.T: (n_wavelengths - 2, n_samples) -> (n_samples, n_wavelengths - 2)
        coefficients, intercepts, summary_gofs, all_gofs = train_rrs_model(diffD2.T, hplc_i, pft_index, n_permutations, max_pcs, k, mdl_pick_metric, seed=100)

        a_filepath = output_dir / f"a_coefs_{pigment}.csv"
        c_filepath = output_dir / f"c_coefs_{pigment}.csv"
        pd.DataFrame(coefficients).to_csv(a_filepath, index=False)
        pd.DataFrame(intercepts).to_csv(c_filepath, index=False)

    print(f"elapsed_seconds: {time.time() - start}")
