# Implementation audit: how the repository turns PACE OCI Rrs into SDP pigments

Audit date 2026-10-13. Read-only. Files were audited as they sit on disk. Three audited files carry uncommitted edits from another session: `src/eddy_tracking/pipeline/build_gold_table.py` (the `tchla_zero_frac` columns), `src/eddy_tracking/validation/quality.py` (`keep_valid_sdp_rows`), and `src/eddy_tracking/preprocess/pace.py` (a filename pattern). Every other file cited here is committed.

Sources used throughout:
- Paper: Kramer, Siegel, Maritorena, Catlett (2022), Remote Sensing of Environment 270, 112879, cited below by section, equation, or table. https://doi.org/10.1016/j.rse.2021.112879
- Authors' MATLAB code: https://github.com/sashajane19/Rrs_pigments (files `Kramer_hyperRrs.m`, `gsm_invert.m`, `gsm_cost.m`, `Kramer_Rrs_pigments.m`, `rrsModelTrain.m`). Read through a fetch tool that summarized the files and quoted the key lines, so quoted formulas are reliable and anything else from it is marked as summarized.
- NASA's own implementation exists as `get_sdp.c` in OCSSW l2gen (L. McKinna, December 2022). https://oceancolor.gsfc.nasa.gov/docs/ocssw/get__sdp_8c_source.html The page redirected and could not be read, so it was not compared.
- Checks I ran are small scripts in the session scratchpad under `scratchpad/audit/` (`check_inputs.py`, `check_training.py`, `check_sampling.py`, `check_pixels.py`, `check_silver.py`, `check_stale.py`, `check_misc.py`). Each states its sample. The real-pixel sample is 150 eddy-interior pixels from 6 eddies and 28 composites, so those numbers are indicative, not population estimates.

## Summary: real issues ranked, and steps found faithful

### Takeaway
One coding defect changes every SDP number the pipeline reports: PACE band centers are truncated to whole nanometers before the spline, which adds roughly +0.18 mg m-3 to Tchla and smaller offsets to most other pigments. Beyond that, the model itself returns zero or impossible values in clear warm water even on its own training data, and the pigment files on disk are older than the Rrs files they should match.

### Cited Findings
Real issues, most important first. Each lists evidence, how it was confirmed, effect size, and confidence.

- Issue 1. Band wavelengths are truncated to integers. `src/eddy_tracking/pipeline/collocate_pace.py:150-152` reads the file's wavelength coordinate and applies `.astype(int)`, then names columns `Rrs_<int>`. `src/eddy_tracking/packages/sdp/prediction.py:259-262` rebuilds the wavelength vector from those names and hands it to the spline. The V3.2 L3m coordinate is float32 with fractional centers (346.017, 348.468, ..., 400.178, 402.654, ..., 419.988, ...), so each band is placed 0 to 1 nm too short (mean 0.51 nm, max 0.996 nm) with a band-to-band jitter. Same line in `src/eddy_tracking/pipeline/background.py:123`, and in the sibling repo `sdp-validation/sdpval/matchup.py:54`. Confirmed by running: (a) the stored column labels equal floor(true wavelength) for all 172 bands; (b) on 150 real interior pixels, pipeline labels against true float labels give mean Tchla 0.457 against 0.313 (ratio 1.46), Fuco 1.43, HexFuco 1.36, Chlc12 1.45, Chlc3 1.47, Allo 1.55, Zea 1.20, Perid 1.21, DVchla 1.03; (c) the shift is additive, pipeline = 0.98 x true + 0.185 for Tchla, +0.04 for Fuco, +0.03 for HexFuco, +0.010 to 0.015 for Zea, +0.001 to 0.004 for DVchla; (d) on the 145 noise-free training spectra resampled at PACE band centers, true labels reproduce the trained means within 0.91 to 1.16 while integer labels give 1.23 to 1.34 for 11 of 13 pigments. Effect: every concentration is biased high, most in clear water (pipeline/true mean Tchla ratio 2.45 in the lowest chlor_a tercile, 1.24 in the highest). Ratios to Tchla are biased low where Tchla is small: Zea/Tchla 0.329 against 0.635 in the low tercile, 0.071 against 0.077 in the high tercile; DVchla/Tchla overall 0.059 against 0.084; Fuco/Tchla nearly unchanged (0.219 against 0.224). Because the bias depends on water type, it distorts cross-front contrasts and changes over an eddy's life, not only absolute levels. Confidence: high on the defect and its direction, medium on the exact sizes (150 pixels). The Level-2 path does not have this defect (`src/eddy_tracking/preprocess/pace.py:181-191` keeps three decimals and `prediction.py:154,197-200` uses the float attrs), so an L2 against L3 comparison would show a spurious offset of this size.
- Issue 2. SDP Tchla collapses to zero in clear warm water, and the truncation offset currently hides most of it. Confirmed by running: with true labels 42 of 150 interior pixels have Tchla clipped to zero (30 of 50 in the lowest chlor_a tercile, median chlor_a 0.068 mg m-3, median SST 25.1 C; 0 of 50 in the highest tercile). With pipeline labels it is 14 of 150. This is a property of the linear model, not only of PACE noise: applying the committed coefficients to the model's own 145 training spectra gives a negative median Tchla (clipped to 0) for 15 samples, a predicted DVchla at or above predicted Tchla for 22 samples (the largest measured DVchla/Tchla there is 0.65), and a regression slope of predicted on measured Tchla of 0.67 (R2 0.81). Effect: Tchla, and every ratio that divides by it, is not usable on the warm side of the Gulf Stream; `keep_valid_sdp_rows` (`src/eddy_tracking/validation/quality.py:104-122`, uncommitted) therefore removes rows selectively from clear warm water, which `docs/research_scope.md:51` already observes ("nearly all in eddies that stay south of the Gulf Stream between October and January"). After issue 1 is fixed, far more rows would fail that rule. Confidence: high for the training-set result, medium for the PACE fractions.
- Issue 3. SDP Tchla runs well above PACE standard chlorophyll. Confirmed by running on the same 150 pixels against `chlor_a` from the matching `PACE_OCI...L3m.8D.BGC.V3_2` file: pipeline Tchla / chlor_a is 1.94 on the mean (1.89 median of pixel ratios); with true labels 1.33; the GSM chlorophyll from the inversion step is 1.00 on the mean. Effect: absolute Tchla from the pipeline is about twice the standard product, and about two thirds of that excess (0.61 of 0.94) is issue 1. Confidence: medium (one sample, and chlor_a is itself an estimate).
- Issue 4. Stored pigments are older than the Rrs they are meant to match, and the SDP stage will not refresh them. `src/eddy_tracking/pipeline/run_sdp.py:34-39` returns early when the output file exists. On disk, all `silver/collocate_pace` files were rewritten 2026-10-13 19:29 and now run past 2025-12-29, while `silver/pigments` files date from 2026-09-08 and 09; 151 eddies have Rrs and no pigment file, and the gold table (written 19:26) ends 2025-12-29. Confirmed by running: in 91 shared eddies the track ids and centers agree on all 520 shared dates (so nothing is mislabeled today), and the new Rrs files hold 20 composite dates that the old pigment files lack. Effect if `run_sdp` is rerun without clearing the folder: new eddies get full series, continuing eddies silently stop at the old end date, which biases late-age bins. Related: `src/eddy_tracking/pipeline/build_gold_table.py:151-157` joins pigments to track observations by nearest date with no tolerance, so a stale file under a reused track id would join without error. Confidence: high on the mechanism; whether it bites depends on how the other session finishes its rerun.
- Issue 5. Zero-clipping happens per pixel before the interior mean. `prediction.py:121-123` clips, `build_gold_table.py:43-46` averages with zeros included. Confirmed by running on 164,067 stored interior pixels: zero fraction is Tchla 4.8%, DVchla 20.8%, Neo 21.2%, MVchlb 14.9%, Viola 12.6%, Allo 10.9%, Fuco 9.8%, Chlc12 6.1%. Effect (inferred, assuming the clipped values are symmetric scatter): the interior mean is biased high by about 1% for a pigment with 5% zeros and about 13% for one with 20% zeros, so ratios such as DVchla/Tchla are inflated in exactly the eddy-composites with many zeros. Confidence: medium.
- Issue 6. The 0.8 coverage gate is applied before SDP drops pixels. `collocate_pace.py:66-72` computes coverage from finite Rrs; `build_gold_table.py:38` stores that value as `valid_frac`; pixels later dropped for missing SST or SSS or a failed inversion (`prediction.py:248-257,276-283`) are not counted. Confirmed by running on 873 eddy-composites: 1.0% of interior pixels are lost at the SDP stage, 32 composites lose more than 10%, and 31 (3.6%) end below 0.8 true coverage, 4 below 0.5. Effect: small. Confidence: high.
- Issue 7. PACE has no Rrs bands between 588.1 and 612.7 nm, and the spline bridges the gap. Confirmed by reading the file coordinate and by running the training spectra through PACE band positions with and without that gap filled: the gap alone moves mean Tchla by +5%, Fuco +6%, Perid +9%, DVchla -6%. 24 of the 299 regression coefficients sit on interpolated values. Effect: small and systematic. Confidence: medium (noise-free simulation).
- Issue 8. Second-derivative amplitude from PACE exceeds the training range in the red. Confirmed by running: rms of the 1 nm second derivative of the residual, PACE pixels against training spectra, is 2.7e-6 against 1.3e-6 at 660-700 nm, 2.3e-6 against 1.4e-6 at 613-660 nm, 1.1e-5 against 7.5e-6 at 401-450 nm, and about equal at 450-588 nm. The regression is linear, so extra variance in those bands passes straight into the pigments. The same test showed that merely passing the training spectra through the pipeline's own spline and 5 nm mean a second time moves individual predictions by 12 to 25% of the median (means within -4 to +9%) and lowers Tchla R2 against HPLC from 0.81 to 0.74. Effect: per-pixel values are fragile to spectral smoothing at the 10 to 25% level; means are sturdier. Confidence: medium.
- Issue 9. Negative Rrs is not screened. `collocate_pace.py:63` only requires finite values. Confirmed by running on one composite (2025-01-17): 2.6% of valid pixels have at least one negative Rrs between 396 and 704 nm, mostly near 400-413 nm and 700 nm; the training set has none. Effect: small share of pixels with an unphysical inversion input. Confidence: high on the count, low on the effect.
- Issue 10. One paper step is missing. The paper sets modeled values below the HPLC detection limits to zero before analysis (section 2.5); the code only clips negatives (`prediction.py:123`). Effect: tiny positive values survive and can enter ratios. Confidence: high that it differs, low that it matters after averaging.

Steps checked and found faithful (to the authors' MATLAB code, which is the reference the coefficients were trained with):
- Product selection and version pinning: `configs/gulf_stream_20240305_20260531/config.yaml:53-58`, `src/eddy_tracking/downloads/pace.py:41-45`. All 103 files report processing_version 3.2 and 172 bands (ran).
- Units (sr^-1), latitude order (descending, handled by `np.meshgrid(lon, lat)` with a `(lat, lon, wavelength)` array, `collocate_pace.py:51-57`), longitude convention (`src/eddy_tracking/preprocess/tracks.py:64-80`).
- 1 nm grid, 5 nm centered moving mean, 4 nm trim, 400-700 nm, no extrapolation: `src/eddy_tracking/packages/sdp/preprocessing.py:85-115` against paper section 2.2.
- Rrs to rrs conversion `physics.py:264` (paper Eq. 2); g = 0.0949, 0.0794 `physics.py:184,326` (Eq. 1); aw from Mason et al. 2016 and the A, B table, both starting at 400 nm at row 50 (`physics.py:269-278`, ran); bbw = 0.5 bsw from Zhang et al. 2009 (`physics.py:291-295`); adg slope and bbp exponent formulas identical to `Kramer_hyperRrs.m`; start values 0.15, 0.01, 0.0029 and tolerances identical to `gsm_invert.m` (`physics.py:211-225`).
- Residual defined as measured minus modeled above-surface Rrs (`physics.py:331-334`, paper section 2.3); second-order difference at 1 nm (`prediction.py:104`, MATLAB `diff(x,2,2)`).
- Ensemble of 100 coefficient sets reduced by the median, negatives set to zero (`prediction.py:116-123`, paper section 2.5).
- Training settings 100 permutations, 30 components, 5 folds, MAE selection (`scripts/train_sdp.py:304-308`) match `Kramer_Rrs_pigments.m`.
- SST and SSS have almost no influence on the result (see the ancillary section), so the choice of ancillary product is not a source of error.
- Composite date, SST date, and eddy contour date agree (see the last section).

### Inferences
- Fixing issue 1 will lower all stored concentrations and raise the count of zero-Tchla pixels, so the two cannot be judged separately: the retrieval-quality rule adopted today was tuned on biased values.
- Because the truncation offset is additive and nearly constant in mg m-3, relative changes "from first observation" in concentration are compressed in clear water and less affected in productive water.
- The BATS matchups in `sdp-validation` used the same integer labels, so validation numbers from that path carry the same offset. The earlier PVST validation code is deleted, so which labels it used cannot be checked.

### Gaps
- NASA's `get_sdp.c` could not be read, so the pipeline was not compared with the agency implementation or its coefficient file.
- The effect sizes come from 150 pixels and from noise-free resampling of training spectra. A full rerun of one experiment with float labels would give the real distribution; it was outside the compute limit.

## Input: product, wavelengths, grid mapping, units, missing and negative Rrs

### Takeaway
The input is PACE OCI Level-3 mapped 8-day 4 km AOP Rrs, version 3.2, read correctly for units and orientation. The mapping to the 1 nm model grid follows the paper except for the truncated band centers (issue 1) and an unmeasured 588-613 nm gap (issue 7).

### Cited Findings
- Product: `PACE_OCI.<start>_<end>.L3m.8D.AOP.V3_2.4km.nc`, collection C4184125822-OB_CLOUD, version string pinned in the filename match so two reprocessings cannot mix (`config.yaml:53-58`, `downloads/pace.py:41-45`). File attributes: processing_version 3.2, measure "Mean", 4.638 km, Rrs in sr^-1 as float32 on `(lat, lon, wavelength)` (ran `check_inputs.py`).
- Bands: 172 from 346.017 to 719.298 nm. Spacing is about 2.5 nm to 588.086 nm, then nothing until 612.730 nm, about 2.5 nm to 639.8 nm, and about 1.25 nm from 641.0 to 714.3 nm (ran). All 172 are stored per pixel; the model uses 396 to 704 nm.
- A pixel is kept only if all 172 bands are finite (`collocate_pace.py:63`). In the file checked, pixels are all-or-nothing (72,816 any finite, 72,816 all finite), so the rule drops nothing extra (ran).
- Mapping: per pixel, a cubic spline (`scipy.interpolate.CubicSpline`, default not-a-knot ends, `extrapolate=False`) onto 396 to 704 nm at 1 nm (`preprocessing.py:85,105-109`); a centered 5 nm moving mean that returns NaN unless all 5 values are finite (`preprocessing.py:18-44,111`); trim to 400 to 700 nm, 301 values (`preprocessing.py:113-115`). The bands at 395.2 and 704.4 nm bracket the grid, so there is no extrapolation at either end (ran, read).
- Paper: "All spectra were interpolated to 1 nm resolution and smoothed using a 5 nm moving mean bandpass filter... the first and last 4 nm of all spectra were removed... restricted to 400-700 nm" (section 2.2). Same steps.
- Wavelength labels: see issue 1. `collocate_pace.py:150-152`, `prediction.py:259-262`.
- Negative Rrs: passed through unchanged (issue 9). A non-finite value cannot reach the inversion on the L3 path; if one did, the simplex search would hit its evaluation limit and the pixel would be dropped as non-convergent (ran `check_misc.py`: "raises GSMInversionError ... warnflag=1").

### Inferences
- The paper's measured spectra were continuous through 590-610 nm, where pure-water absorption has a sharp shoulder. A spline across that gap cannot follow the shoulder, which is the likely cause of the small bias in issue 7.
- PACE bands are about 5 nm wide before the 5 nm mean; the training radiometers were broader. The effective smoothing at prediction time is therefore not identical to training, and issue 8 shows the output is sensitive to that.

### Gaps
- The paper does not name the interpolation method used for Rrs (it names cubic spline only for the A and B table, section 2.3), so the spline choice cannot be checked against it.

## Ancillary: SST and SSS sources, pairing, missing values, sensitivity

### Takeaway
SST and SSS enter only through seawater backscattering and change the pigments by well under 1%. The only practical consequence of the ancillary step is that a pixel with no SST or SSS is dropped.

### Cited Findings
- SST: Aqua MODIS L3m 8-day 4 km `sst` (11 micron daytime, R2019.0), on the same grid and the same 8-day periods as PACE; each file is stamped with its date-range midpoint (`src/eddy_tracking/preprocess/sst.py:22-30`, `config.yaml:60-61`). `qual_sst` (values 0, 1, 2 present) is not applied (ran).
- SSS: SMAP JPL V5 8-day running mean on a 0.25 degree grid, one file per day stamped with its center date (`src/eddy_tracking/preprocess/sss.py:22-28`, `config.yaml:63-64`). 809 daily files from 2024-02-26 to 2026-06-05 with gaps of at most 5 days (ran).
- Pairing: nearest neighbor in time, latitude, and longitude with no tolerance (`src/eddy_tracking/packages/sdp/ancillary.py:44-74`), called with the composite midpoint date (`prediction.py:239-245`). SST matches the composite exactly (103 SST files for 103 PACE files); SSS is within a few days.
- Missing value: no constant or climatological fallback. A pixel with NaN SST or SSS is removed and a count is printed (`prediction.py:248-257`). In the 222 pixels sampled, one lacked SST and none lacked SSS; across 873 eddy-composites 1.0% of interior pixels are lost at the SDP stage for all causes (ran).
- Sensitivity, run on 71 real pixels with all else fixed: SST +5 C changes the median pixel by at most 0.05% for any pigment; SST -10 C by at most 0.39% (DVchla), Tchla 0.11%; SSS -2 by at most 0.25%; replacing both with constants (20 C, 35) by at most 0.17%. Mean ratios stay within 0.997 to 1.004.
- Paper: temperature and salinity for bbw came from the World Ocean Atlas quarter-degree climatology (section 2.3). The repo uses concurrent satellite fields instead. The authors' MATLAB takes per-sample T and S (`Kramer_hyperRrs.m`, summarized).

### Inferences
- A 10 C error across a front, or a climatological value, would not change any reported pigment by more than about 0.4%. The reason is in the code: T and S only set bbw (`physics.py:291-295`), bbw is smooth in wavelength, and the free bbp term plus the second derivative absorb almost all of the change.
- The caution in `docs/research_scope.md:38` that pigment covariance with SST and SSS is not independent because both enter SDP is, by this test, not a practical concern.
- A constant T and S would give the same pigments and would stop the loss of pixels where MODIS or SMAP has gaps (coastal SMAP cells, cloudy MODIS composites).

### Gaps
- The share of pixels lost specifically to SMAP land contamination near the shelf was not separated from other losses.

## Inversion: model terms, fixed and free parameters, optimizer, failures

### Takeaway
The inversion is a line-for-line port of the authors' MATLAB: an unbounded Nelder-Mead fit of three parameters. A failed fit drops the pixel; a fit that converges to unphysical values is kept without any flag.

### Cited Findings
- Model: rrs = g1 u + g2 u^2 with u = bb / (a + bb), g = 0.0949, 0.0794 (`physics.py:184-192`; paper Eq. 1). a = aw + A chl^B + adg443 exp(S (lambda - 443)); bb = bbw + bbp443 (443/lambda)^eta (`physics.py:186-188,285,301`; Eqs. 3, 4, 6).
- Fixed per pixel: aw (Mason et al. 2016 table, `physics.py:269-273`); A and B (`aph_A_B_Coeffs_Sasha_RSE_paper.csv`, `physics.py:276-278`); bbw from Zhang et al. 2009 with depolarization 0.039 (`physics.py:110-171,291-295`); S = -(0.01447 + 0.00033 Rrs490/Rrs555) from above-surface Rrs (`physics.py:284`); eta = 2.0 (1 - 1.2 exp(-0.9 rrs440/rrs555)) from below-surface rrs (`physics.py:298-300`).
- Free: chl, adg(443), bbp(443). Start values 0.15, 0.01, 0.0029 (`physics.py:211`). No bounds.
- Optimizer: `scipy.optimize.fmin` (Nelder-Mead), xtol 1e-9, ftol 1e-9, 2000 iterations and 2000 evaluations (`physics.py:216-225`). The authors use `fminsearch` with the same four settings and the same start (`gsm_invert.m`, quoted by the fetch).
- Cost: unweighted sum of squared differences in below-surface rrs over all 301 wavelengths (`physics.py:191-192`; `gsm_cost.m` quoted in full).
- Failure: a non-zero `warnflag` raises `GSMInversionError` (`physics.py:227-231`); `_predict_pace_spectra` bisects the batch to isolate the pixel and marks it (`prediction.py:311-348`); the L3 wrapper removes it and prints a count (`prediction.py:276-283`). The pixel is absent from the output, which is the equivalent of the NaN the MATLAB code leaves. It never becomes a zero or a normal-looking number.
- Bound or unphysical solutions: nothing checks them. On the training set 5 of 145 fits return negative bbp(443); on 150 PACE pixels none returned a negative adg or bbp and none failed (ran).
- The inversion's chlorophyll is discarded. Reported Tchla comes from the regression, not from this fit.
- Two places where the paper's text and the authors' code disagree, and the repo follows the code: paper Eq. 5 prints Sdg = -0.01447 + 0.00033 Rrs(490)/Rrs(555), the code uses -(0.01447 + 0.00033 ratio); the paper says eta is a function of rrs(490)/rrs(555), the code uses rrs(440)/rrs(555) (section 2.3 against `Kramer_hyperRrs.m`).

### Inferences
- Since the repo's coefficients were trained with these same formulas, the paper-text differences do not create a train and predict mismatch. They matter only if someone later "corrects" the code to the printed equations without retraining.
- The GSM chlorophyll matched PACE `chlor_a` on the mean (ratio 1.00) while the regression Tchla did not, which points at the regression step, not the inversion, as the source of the Tchla bias.

### Gaps
- The port's GSM chlorophyll against HPLC Tchla on the training set gives R2 0.72 (log) and 0.78 (linear); the paper reports 0.86 (Fig. 4B). The cause was not found. It could be the climatological T and S the paper used, a different axis convention, or optimizer differences between MATLAB and SciPy.

## Residual and derivative

### Takeaway
The residual and its second derivative match the paper and match what the coefficients were trained on. No smoothing is applied after the residual.

### Cited Findings
- Residual: measured minus modeled above-surface Rrs, after converting the modeled rrs back with Rrs = 0.52 rrs / (1 - 1.7 rrs) (`physics.py:321-334`); `run_sdp` takes element 1, the above-surface one (`prediction.py:103`). Paper section 2.3 defines the same quantity; MATLAB `RrsD = Rrs - modRrs`.
- Derivative: `np.diff(residual, 2, axis=0)`, the plain second difference at 1 nm with no division, 299 values for 401 to 699 nm (`prediction.py:104`). Paper: "second-order finite difference approximation" (section 2.3). MATLAB: `diff(hgsm_diff,2,2)`. Training uses the identical call (`scripts/train_sdp.py:302`).
- Range: all 299 values are used; the paper also uses the full 400-700 nm range (section 4.4).
- Smoothing: only the 5 nm mean on Rrs before the inversion. None on the residual or the derivative, in the paper, the MATLAB, or the repo.

### Inferences
- A 1 nm second difference of a spectrum that was sampled every 2.5 nm (1.25 nm in the red) and spline-interpolated carries the spline's own curvature and the band noise. This is why the label error in issue 1 and the smoothing differences in issue 8 have such leverage.

### Gaps
- None for this step.

## Coefficients: provenance and train against predict consistency

### Takeaway
The coefficients are not the authors' published values. They were retrained in Python on the authors' public 145-sample in situ dataset with a port of the authors' training routine, using the same residual and derivative code that prediction uses. The procedure is consistent; the input spectra at prediction time are not of the same kind.

### Cited Findings
- The 26 CSV files in `src/eddy_tracking/packages/sdp/coefficients/` are byte-identical to `old/models/sdp_pigments/coefficients/` in the git history of `sdp-validation` (hashes compared for 5 pigments, ran). That folder's README says they are "written by `models/sdp_pigments/train.py`". Shape 299 x 100 per pigment plus 100 intercepts (ran).
- The archived trainer (`git show HEAD:old/models/sdp_pigments/train.py` in `sdp-validation`) reads `Kramer-etal_2021.csv`, the PANGAEA compilation the paper cites (https://doi.pangaea.de/10.1594/PANGAEA.937536): 145 rows, Rrs400 to Rrs700 at 1 nm, in situ `Temp` and `Sal`, 25 HPLC pigments. It calls `get_rrs_residuals` on those spectra directly and then `train_model`.
- The two reference tables are byte-identical between the repos, and the current `physics.py` differs from the archived one only in formatting, comments, and the added convergence error (diff, ran).
- Training settings in `scripts/train_sdp.py:304-308` (100 permutations, at most 30 components, 5 folds, MAE, non-negative constraint) match `Kramer_Rrs_pigments.m`. Spectra are z-scored by the standard deviation before the SVD (`scripts/train_sdp.py:121-124`) and coefficients are un-standardized afterward (`:226-230`).
- The authors' repository ships no A or C coefficients, only the training code and a 17-spectrum test file (repo listing, summarized).
- In-sample check, ran: the committed coefficients applied to the 145 training spectra give R2 of Tchla 0.81, Fuco 0.77, Chlc12 0.78, Chlc3 0.74, DVchla 0.65, Perid 0.64, ButFuco 0.63, HexFuco 0.62, Zea 0.48, Allo 0.48, Neo 0.42, MVchlb 0.35, Viola 0.30, and means within 15% of the measured means. The ordering agrees with the paper's Table 2 and Fig. 6 range (R2 0.51 to 0.73 for Tchla and five biomarkers).
- The same check gives slopes of predicted on measured between 0.20 and 0.67 for every pigment, so the model compresses the range even in-sample. The paper reports slopes of 0.74 to 0.94 for Tchla and five biomarkers (section 3.4) without stating the regression direction.
- Training preprocessing against prediction preprocessing: both run the same `get_rrs_residuals` and the same `np.diff`. Units are sr^-1 in both. Wavelength range and spacing are the same. The regression is not conditioned on Tchla in either.
- What differs is the spectrum fed in: training spectra are in situ, already at 1 nm and already smoothed by the authors; prediction spectra are 8-day mean satellite Rrs at 2.5 nm (1.25 nm in the red), missing 588-613 nm, splined and smoothed by the repo, with truncated band labels. Issues 1, 7, and 8 quantify each part.

### Inferences
- The committed files cannot be regenerated from `eddy-tracking` alone. `scripts/train_sdp.py` has no entry point that builds the residuals, the training CSV lives only in the other repo's history, and the script now seeds `np.random.default_rng(100)` (`scripts/train_sdp.py:60`) where the archived trainer used `np.random.seed(100)`, so a rerun would give a different, statistically equivalent ensemble. This is a provenance note; it changes no reported number.
- Because the ensemble is a different random draw from the authors', results should be described as "SDP method retrained on Kramer et al. 2021 data", not as the authors' or NASA's coefficients.

### Gaps
- The retraining was not rerun (about 25 s per pigment, over the limit), so reproducibility of the committed files from the archived trainer is inferred from the README and the hash match, not demonstrated.
- No comparison with NASA's coefficient file was possible.

## Prediction: ensemble, clipping, internal consistency, flags

### Takeaway
The prediction step does what the paper says, and nothing more: no detection-limit floor, no consistency check among pigments, and no per-pixel quality flag on the L3 path.

### Cited Findings
- For each pigment, the 299-value derivative is multiplied by the 299 x 100 matrix, the 100 intercepts are added, the median over the 100 results is taken, and a negative median is set to zero (`prediction.py:116-123`). Paper Eq. 7 and section 2.5 ("The median pigment value of those 100 modeled values was used").
- No floor at the HPLC detection limits, which the paper applies (section 2.5). See issue 10.
- No consistency rule at the pixel level. On stored interior pixels, DVchla is at or above Tchla in 6.0% of pixels (1.2% of pixels with Tchla above zero), and the 12 accessory pigments sum to more than Tchla in 83% of pixels with Tchla above zero (ran on 164,067 pixels).
- The L3 path returns no status column; it drops failed pixels (`prediction.py:234-283`). The L2 path does return `sdp_status` (`prediction.py:140-231`).
- The only quality screen sits downstream and in uncommitted code: `keep_valid_sdp_rows` removes an eddy-composite when at least half of its interior pixels have zero Tchla or when its mean DVchla is at or above its mean Tchla (`quality.py:104-122`). On the current gold table 104 of 3,571 rows fail the first test and 54 the second (ran).
- Spread across the 100 ensemble members is not kept. On the training set the interquartile range of the 100 predictions, relative to the median, has a median of 0.18 for Zea, 0.29 for DVchla, 0.53 for Tchla, and about 1.0 for Fuco, Chlc12, Allo, and Neo (ran).

### Inferences
- The ensemble spread is a free per-pixel uncertainty estimate that the pipeline discards. For Fuco it is as large as the value itself even on training data.
- A zero is a censored value, not a measurement of absence. Treating zero-Tchla pixels as valid observations of "no chlorophyll" would be wrong; the gold table's `tchla_zero_frac` is the right kind of record for this.

### Gaps
- Whether the 83% share of pixels with accessory pigments summing above Tchla is abnormal was not tested against HPLC data.

## Aggregation: order of operations, interior pixels, coverage gates, zeros, ratios

### Takeaway
SDP runs per pixel and the interior mean is taken afterward, with clipped zeros included. Ratios in the paper notebooks are ratios of interior means, which is the sturdier choice. The coverage gate does not account for pixels lost inside SDP.

### Cited Findings
- Order: SDP per 4 km pixel (`run_sdp.py:41`), then a mean over interior pixels per eddy and composite date (`build_gold_table.py:28-47`). The paper expects aggregation of satellite Rrs before the method is applied: "spatiotemporal aggregation of remotely-sensed Rrs will likely be required to improve and increase the signal-to-noise ratio" (section 4.1). The 8-day composite is temporal aggregation; there is no spatial aggregation of spectra.
- Interior: a pixel whose center lies inside the py-eddy-tracker speed contour of the chosen day (`tracks.py:149-156`, `collocate_pace.py:57`). Pixels within 2 speed radii are also stored for ring analyses and are excluded from the gold mean by `inside_contour` (`collocate_pace.py:61-64`, `build_gold_table.py:28`).
- Gates: at least 80% of interior pixels with finite Rrs (`collocate_pace.py:66-72`, `config.yaml:101-104`); at least 10 interior pixels with pigments, hard-coded (`build_gold_table.py:138-142`). See issue 6 for the coverage mismatch.
- Zeros: included in the mean as zeros (`build_gold_table.py:43-46`); counted in `tchla_zero_frac` (`build_gold_table.py:30-40`, uncommitted). See issue 5.
- Ratios: `log_ratio_<pigment>` is log(eddy mean / background mean), a ratio of means, with infinities set to NaN (`build_gold_table.py:213-234`). The notebooks divide `eddy_mean_<pigment>` by `eddy_mean_Tchla` after `keep_valid_sdp_rows` (`research_paper/figures.ipynb` cell 4, `research_paper/eddy_evolution.ipynb` cell 1; read). On 164,067 stored pixels the Fuco/Tchla ratio of means and mean of ratios agree per eddy-composite (median 0.218 against 0.213), but pixel ratios reach the hundreds or thousands where Tchla is near zero (max 179 for Fuco, 1,555 for Zea) (ran).
- Background: the per-date regional mean over at most 2,000 randomly chosen calm, non-eddy pixels of the whole 29-44 N box, run through the same SDP path and the same truncated labels (`src/eddy_tracking/pipeline/background.py:123-175`).

### Inferences
- Because the model is linear after the inversion, the mean of per-pixel pigments is close to the pigment of the mean spectrum except for the clipping. Averaging spectra first (or averaging unclipped predictions and clipping once) would remove the bias in issue 5 and match the paper's stated expectation.
- The truncation offset is the same in the eddy and the background, but it is additive, so it does not cancel in `log_ratio`: adding 0.18 to a numerator of 0.1 and a denominator of 0.4 moves the ratio from 0.25 to 0.48.
- A single regional background mixes slope water and Sargasso water; that is a design question outside this audit.

### Gaps
- The ring aggregation cells in the notebooks (0.2 R rings, 3-pixel gate) were not audited line by line.

## Silent wrong numbers: index, orientation, date, version, unit checks

### Takeaway
Wavelength labeling is the one silent error found. Latitude order, dates, product versions, units, and table offsets are correct. Stale pigment files are a second silent risk in the current state of the disk.

### Cited Findings
- Wavelength labels: wrong by 0 to 1 nm, silently (issue 1).
- Wavelength index into the tables: `aw` and A, B both have 400 nm at row 50 and end at 700 nm; the code asserts this for `aw` only (`physics.py:270-278`; ran for both). The 440, 490, and 555 nm columns used for the slopes exist by name after preprocessing (`physics.py:256-259,281-299`). Training and prediction difference the same 400-700 grid, so coefficient row k means the same wavelength in both.
- Latitude order: descending in PACE and SST files; grids are built with `np.meshgrid(lon, lat)` and the Rrs cube is reshaped in `(lat, lon)` order, consistently (`collocate_pace.py:51-55`, `tracks.py:149-156`). SST is sampled by coordinate value, not by index (`ancillary.py:53-61`).
- Dates: an 8-day file is given the date start + 3 days (`src/eddy_tracking/preprocess/pace.py:276-282`); the SST file gets the same midpoint (`sst.py:27`); the eddy contour is the detected (non-interpolated) observation nearest that midpoint within the 8 days (`tracks.py:64,87-105`). So Rrs, SST, and contour refer to the same period. The contour is one day's outline laid over an 8-day mean.
- Product versions: one version, one band set, across all 103 PACE files (ran).
- Units: Rrs sr^-1, SST degrees C, SSS in 1e-3 (practical salinity), matching what `physics.py` expects (ran, read).
- Stale outputs: see issue 4 (`run_sdp.py:34-39`, `build_gold_table.py:151-157`).
- Joins without tolerance: `merge_asof(direction="nearest")` for track features and dynamics (`build_gold_table.py:151-157,180-186`). With matching files the nearest observation is within 4 days.

### Inferences
- An eddy that moves 5 km per day shifts about 20 km between the ends of a composite, so pixels near the contour are a blend of interior and exterior water. With radii of 50 to 100 km this blurs edges and rings more than interior means.
- Nearest-neighbor ancillary sampling without a tolerance would silently reuse a distant date if a file were missing; with the present 103 for 103 SST files it does not.

### Gaps
- Track-id stability was checked for the current rerun only (91 eddies). A rerun that changes `min_track_days` or the amplitude threshold can renumber tracks, and the skip-if-exists rule would then pair old pigments with new eddies.
