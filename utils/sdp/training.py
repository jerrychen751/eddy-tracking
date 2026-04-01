import numpy as np
from pathlib import Path

from utils.sdp.model import rrsModelTrain
import pandas as pd
import time
def train_model(RrsD: np.ndarray | pd.DataFrame, hplc: np.ndarray) -> None:
    """
    Train SDP model on Rrs residuals and HPLC data.
    
    Takes 2nd derivative of Rrs residuals, trains model for all 13 pigments using
    100 permutations, 30 max PCs, 5-fold CV, MAE metric. Saves A (wavelength coefficients,
    shape [n_wl, 100]) and C (intercepts, shape [100]) to CSV files in models/sdp_pigments/coefficients/.
    """

    # Use the 2nd derivative of the residual as model input
    diffD2 = np.diff(RrsD, 2, axis=0)

    n_permutations = 100 # the number of independent model validations to do (each formulates and validates a model)
    max_pcs = 30 # max number of spectral pc's to incorporate into the model - 30
    mdl_pick_metric = 'MAE' # pick the metric by which to evaluate pigment model fit - R2, RMSE, avg, med, ens, bias, MAE
    k = 5
    pft_index = 'pigment'

    pigs2mdl = np.array(['Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
                         'Neo','Viola','Fuco','Chlc12','Chlc3','Perid'])
    
    # column order of the HPLC dataset passed in as hplc
    hplc_vars = ['Tchla','Tchlb','Tchlc','ABcaro','ButFuco','HexFuco','Allo','Diadino','Diato',
                 'Fuco','Perid','Zea','MVchla','DVchla','Chllide','MVchlb','DVchlb','Chlc12','Chlc3',
                 'Lut','Neo','Viola','Phytin','Phide','Pras']
    
    start = time.time()

    # Create model coefficient output directory if it doesn't exist
    output_dir = Path(__file__).resolve().parent / "coefficients"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start modelling
    for i in range(len(pigs2mdl)):
        pigment = pigs2mdl[i]
        pigment_index = hplc_vars.index(pigment)
        hplc_i = hplc[:, pigment_index]
        coefficients, intercepts, summary_gofs, all_gofs = rrsModelTrain(diffD2.T, hplc_i, pft_index, n_permutations, max_pcs, k, mdl_pick_metric, seed=100)

        # Save coefficients to CSV files in the coefficients folder
        a_filepath = output_dir / f"a_coefs_{pigment}.csv"
        c_filepath = output_dir / f"c_coefs_{pigment}.csv"
        pd.DataFrame(coefficients).to_csv(a_filepath, index=False)
        pd.DataFrame(intercepts).to_csv(c_filepath, index=False)

    print('Time taken:', time.time() - start)
