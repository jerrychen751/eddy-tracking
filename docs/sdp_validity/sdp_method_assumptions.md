# The SDP method of Kramer et al. (2022): steps, assumptions, training data, skill, limits, and later work

Research date: 2026-10-13. Scope: the method itself. Regional oceanography, PACE data quality, and Jerry's own code are covered by other notes.

How to read the labels in this file:
- "Read in source" means I read the statement in the paper, slide deck, data file, or web page named beside it.
- "Read through a fetch summary" means a web-fetch tool summarized the page or source file for me and quoted lines from it. Treat exact code lines as likely right but confirm them in the repository before you rely on one.
- "Computed here" means I calculated the number from the local copy of the PANGAEA training file. It is not a published number.
- "Inference" means my own reasoning. It is kept in the Inferences blocks.

Source shorthand used below:
- K22 = Kramer, Siegel, Maritorena, Catlett (2022), Remote Sensing of Environment 270, 112879, https://doi.org/10.1016/j.rse.2021.112879. Local PDF: `/Users/jerry/Zotero/storage/3S3D8JQG/Kramer et al. - 2022 - Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on globa.pdf`. Open access, CC BY.
- KS19 = Kramer and Siegel (2019), JGR Oceans 124, 7557 to 7574, https://doi.org/10.1029/2019JC015604. Local PDF: `/Users/jerry/Zotero/storage/D98SQ8VD/Kramer and Siegel - 2019 - ...pdf`.
- KSG20 = Kramer, Siegel, Graff (2020), Frontiers in Marine Science 7, 215, https://doi.org/10.3389/fmars.2020.00215.
- K24 = Kramer, Maritorena, Cetinic, Werdell, Siegel (2024), Optics Express 32(20), 34482, https://doi.org/10.1364/OE.529906.
- PANGAEA file = Kramer et al. (2021), https://doi.org/10.1594/PANGAEA.937536. Local copy: `/Users/jerry/school/research/sdp-validation/experiments/kramer_hplc_clustering/inputs/in_situ_hplc/Kramer-etal_2021_global_PANGAEA-937536.tab`.
- MATLAB repo = https://github.com/sashajane19/Rrs_pigments (the authors' code).
- Python repo = https://github.com/max-danenhower/rrs-SDP-pigments (the port NASA's tutorial uses).
- NASA tutorial = https://nasa.github.io/oceandata-notebooks/notebooks/oci/oci_sdp.html
- Feb 2025 slides = https://pace.oceansciences.org/docs/D3-SAT3-1330-Kramer-SDPoverview.pdf
- Sep 2025 slides = https://www.star.nesdis.noaa.gov/star/documents/seminardocs/2025/20250924_Kramer.pdf
- C23 = Cael, Bisson, Boss, Erickson (2023), Limnology and Oceanography Letters, https://doi.org/10.1002/lol2.10319.
- PF25 = Prochaska and Frouin (2025), Biogeosciences 22, 4705 to 4728, https://doi.org/10.5194/bg-22-4705-2025.

## 1. Step by step, what does the published method do?

### Takeaway
SDP fits a three-parameter GSM-style reflectance model to each 400 to 700 nm spectrum, subtracts the fit from the measurement, takes the second difference of that residual at 1 nm spacing, and applies 100 sets of linear coefficients per pigment, taking the median and clipping negatives to zero. Every spectral shape in the forward model is fixed in advance, the number of principal components is not fixed (it is chosen by cross-validation, up to 30), and the paper and the authors' code disagree on two formulas.

### Cited Findings

Inputs and preprocessing
- Input is above-water hyperspectral Rrs restricted to 400 to 700 nm, the range common to all eight campaigns. Source: K22 section 2.2, page 3 (read in source).
- All spectra were interpolated to 1 nm, smoothed with a 5 nm moving mean, and the first and last 4 nm were removed after smoothing. Source: K22 section 2.2, page 3 (read in source).
- Each spectrum was inspected by eye. Spectra with repeated spikes of 2 to 5 times the mean in the roughly 610 to 660 nm range were removed: 33 of 178, about 19 percent, leaving 145. Source: K22 section 2.2, page 3, and Table 1 (read in source).
- HPLC and Rrs counted as concurrent when measured within plus or minus 2 hours at the same location. HPLC samples were from 7 m or shallower. Source: K22 sections 2.1 and 2.2 (read in source).
- The underlying in situ radiometry has an effective spectral resolution of about 3.35 nm and was linearly interpolated to 1 nm, according to an independent paper that used the same data family. Source: C23, section on sea surface Rrs, page 2 (read in source). K22 itself only says the spectra were interpolated to 1 nm.

The reflectance model (the "GSM-like" inversion)
- Form: the Gordon et al. (1988) quadratic in bb/(a+bb) with the original g coefficients, with below-surface rrs obtained from Rrs by rrs = Rrs/(0.52 + 1.7 Rrs) after Lee et al. (2002). Source: K22 equations 1 and 2, page 3 (read in source). The code uses g = [0.0949, 0.0794]. Source: MATLAB repo, `Kramer_hyperRrs.m` (read through a fetch summary).
- Free parameters, three per spectrum: Tchla, adg(443) (combined CDOM and non-algal particle absorption), and bbp(443). They come from a non-linear fit of modeled to measured reflectance as in Maritorena et al. (2002). Source: K22 section 2.3, page 3 (read in source).
- Fixed, phytoplankton absorption: aph(lambda) = A(lambda) * Tchla^B(lambda). A and B were regressed per wavelength on NOMAD (18 wavelengths), then spline-interpolated to 1 nm over 350 to 700 nm. Values are in Table S6 of the supplement. The paper states NOMAD is independent of the 145 matchups. Source: K22 equation 3 and following text, page 3 (read in source).
- Fixed, adg shape: an exponential referenced to 443 nm whose slope Sdg is a linear function of Rrs(490)/Rrs(555), credited to Carder et al. (1999) and to a SeaBASS data fit. Source: K22 equations 4 and 5, page 3 (read in source).
- Fixed, bbp shape: a power law referenced to 443 nm whose exponent eta is a function of a below-surface blue-to-green rrs ratio after Lee et al. (2002). Source: K22 equation 6, page 3 (read in source).
- Fixed, pure water: absorption from Mason et al. (2016); backscattering from Zhang et al. (2009). Source: K22 section 2.3, page 3 (read in source).
- How SST and SSS enter: only through the seawater backscattering term bbw (Zhang et al. 2009). In the paper the temperature and salinity came from the World Ocean Atlas 2013 quarter-degree climatology, not from the ship or a satellite. Source: K22 section 2.3, page 3 (read in source). The code takes one T and one S per spectrum, calls `betasw_ZHH2009`, and sets bbw to half of the total scattering; the water absorption file is fixed and is not adjusted for T or S. Source: MATLAB repo `Kramer_hyperRrs.m` (read through a fetch summary).
- Not in the model: chlorophyll fluorescence and Raman scattering. The paper notes that the residual is largest in the blue and in the red, and it names chlorophyll fluorescence as active in the red. Source: K22 section 3.2, page 6 (read in source). The absence of Raman is my reading of equations 1 to 6, which contain no inelastic term.
- Fit details in the Python port: an unbounded Nelder-Mead simplex (`scipy.optimize.fmin`), starting guess [0.15, 0.01, 0.0029] for [Tchla, adg443, bbp443], cost equal to the sum of squared rrs differences over every wavelength from 400 to 700 nm, tolerances 1e-6, 2000 iterations. No bounds, no exit-flag check. Source: Python repo `src/sdp/Kramer_hyperRrs.py` (read through a fetch summary). I did not read the MATLAB `gsm_invert.m` and `gsm_cost.m`.

The residual, the derivative, and the regression
- Residual: measured Rrs minus modeled Rrs. Source: K22 section 2.3, pages 3 and 4 (read in source).
- Derivative: the second derivative of the residual by a second-order finite difference, following Catlett and Siegel (2018). The paper gives no spacing. Source: K22 section 2.3, page 4 (read in source). In the code it is MATLAB `diff(x, 2, 2)` on the 1 nm grid, which is the plain second difference of adjacent 1 nm points with no division by the step, giving 299 features from 301 wavelengths. Source: MATLAB repo `Kramer_Rrs_pigments.m`; Python repo `src/sdp/core.py` uses `np.diff(..., 2)` (both read through a fetch summary). The NASA l2gen port also carries 299 coefficients. Source: https://oceancolor.gsfc.nasa.gov/docs/ocssw/get__sdp_8c_source.html (search-engine summary only; the page redirected when fetched).
- No smoothing is applied after the residual or after the derivative in the code. The only smoothing is the 5 nm moving mean on the input Rrs. Source: MATLAB and Python repos (read through a fetch summary).
- Regression: principal components regression on the standardized second-derivative spectra, one model per pigment, pigments in linear concentration units with no log transform. Source: K22 section 2.5, page 6, and Table 2 caption (read in source); `rrsModelTrain.m` (read through a fetch summary).
- Cross-validation: 100 random splits, 75 percent for training and 25 percent for testing, with a fixed random seed. Inside each training set a 5-fold cross-validation picks the number of components that minimizes mean absolute error, testing 1 up to a maximum of 30. An ordinary least squares fit on the retained scores is mapped back to per-wavelength coefficients, and the coefficients are averaged over the 5 folds. Source: K22 section 2.5 (100 splits, 75/25, optimization on mean absolute difference, read in source); `rrsModelTrain.m` and `Kramer_Rrs_pigments.m` (max_pcs = 30, k = 5, metric 'MAE', read through a fetch summary).
- The number of components kept is therefore not one published number. It varies by pigment, split, and fold, with a ceiling of 30. I found no table of the chosen counts. Source: same code files.
- Final model form: pigment = sum over wavelengths of A(lambda) times the residual second derivative, plus an intercept C. Source: K22 equation 7, page 6 (read in source).
- Application: all 100 coefficient sets are applied and the median of the 100 predictions is used. Source: K22 section 2.5, page 6 (read in source); Python `core.py` loads 100 sets per pigment from `original_a_coefs.xlsx` and `original_c_coefs.xlsx` and takes `np.median` (read through a fetch summary).
- Negative predictions: constrained to zero before statistics were computed. In the full-dataset reconstruction, values below the standard HPLC detection limits (Van Heukelem and Thomas 2001) were also set to zero. Source: K22 section 2.5, page 6 (read in source). The Python port clips negatives to zero and applies no detection limit. Source: Python `core.py` (read through a fetch summary).
- The 13 pigments: Tchla, HexFuco, ButFuco, Allo, Fuco, Perid, Zea, DVchla, MVchlb, Chlc12, Chlc3, Neo, Viola. Source: K22 section 2.1, page 2 (read in source).
- Pigments left out because they were below detection in more than 75 percent of the matchups: DVchlb, lutein, prasinoxanthin. Source: K22 section 2.1, page 2 (read in source).
- The authors chose this model among several with similar skill (first plus second derivative of the raw Rrs, and coarser resolutions) because it performed well and needed one spectral input. Source: K22 section 2.5, page 6, and section 4 opening, page 8 (read in source).

Paper against code: two formula mismatches and two wording slips
- adg slope sign. The paper's equation 5 reads Sdg = -0.01447 + 0.00033 * Rrs(490)/Rrs(555). Both the MATLAB and Python code compute -(0.01447 + 0.00033 * Rrs(490)/Rrs(555)). With a blue-to-green ratio of 5 (clear water) the paper's form gives about -0.0128 per nm and the code gives about -0.0161 per nm. Source: K22 equation 5, page 3 (read in source); `Kramer_hyperRrs.m` and `Kramer_hyperRrs.py` (read through a fetch summary, both agree with each other).
- bbp exponent band. The paper says eta depends on rrs(490)/rrs(555). The code computes 2.0 * (1 - 1.2 * exp(-0.9 * r(440)/r(555))), which is the Lee et al. (2002) form with 440 nm. Source: K22 text under equation 6, page 3 (read in source); same two code files (read through a fetch summary). The Python port appears to pass above-surface Rrs to this formula while its comment says below-surface rrs.
- The paper says the spectra were z-scored by dividing by the variance; the code divides by the standard deviation. Source: K22 section 2.5, page 6 (read in source); `rrsModelTrain.m` (read through a fetch summary).
- The paper describes N in the MAD formula as the training set and then gives it as 25 percent of 145, or 36 samples, which is the test fraction. Source: K22 text under equation 8, page 6 (read in source).

### Inferences
- The coefficients that everyone applies (the 100 sets shipped in the Python repo and the 299-coefficient set in l2gen) were produced by the code, so the code's formulas, not the paper's equations 5 and 6 as printed, define the model. A reimplementation that follows the printed equation 5 would build a different residual from the one the coefficients were trained on. Confidence: high that the mismatch exists in the text, medium on which form is the typo (the code form gives steeper CDOM slopes in clearer water, which is the physically expected direction).
- Because the second difference is taken at 1 nm on spectra that were sampled near 3.3 nm in situ and interpolated, part of the fine structure the coefficients respond to depends on how the native samples were interpolated and smoothed. PACE OCI samples at about 2.5 nm with a 5 nm bandpass, so its interpolated 1 nm second differences have a different fine structure. The paper never applied the 1 nm coefficients to coarser data interpolated up to 1 nm (see question 4). Confidence: medium that this matters; size unknown.
- The fit runs over 400 to 700 nm including the fluorescence band, with a model that has no fluorescence term. The residual near 683 nm therefore carries fluorescence, and the regression is free to use it. Fluorescence yield changes with light history and nutrient stress, so that part of the signal need not transfer between midday ship casts and a satellite composite. Confidence: low to medium; untested anywhere I could find.
- Each pigment has its own unconstrained linear model. Nothing ties an accessory pigment to Tchla, so a retrieved ratio of pigment to Tchla can exceed its physical range, most easily where Tchla is clipped or small.

### Gaps
- The supplement (Tables S1 to S6, Figures S1 to S15) was not readable: the Elsevier file and the dissertation PDF (over the fetch size limit) could not be retrieved. I do not have the per-pigment statistics for the alternative models or the A and B table.
- I did not read `gsm_invert.m` or `gsm_cost.m` in the MATLAB repo, so the MATLAB optimizer, bounds, and starting values are unconfirmed.
- Whether the Rrs in the PANGAEA file is already smoothed is not stated in the file header I read. A crude check (an extra 5-point mean lowers the mean absolute second difference by only 16 percent) suggests it is already smooth, but that test is weak.

## 2. The 145 training samples

### Takeaway
The training set is 145 open-ocean matchups from eight campaigns between 2004 and 2018, skewed to clear water: the median Tchla is 0.11 mg m-3 and 46 percent of samples are below 0.1. Sixteen samples fall inside 30 to 45 N, 80 to 50 W, and every one of them was collected in July, August, or September; the box has no winter, spring-bloom, or autumn training sample.

### Cited Findings

Campaigns, regions, and counts
- Eight campaigns in Table 1, with samples kept and (removed): ANT 26 (28), Atlantic; NAAMES 11 (1), Northwest Atlantic; RemSensPOC 27, Northwest Atlantic and equatorial Pacific; SABOR 9, Northwest Atlantic; Tara Oceans 16 (3), global; Tara Mediterranean 29, Mediterranean Sea; BIOSOPE 23 (1), Southeast Pacific; EXPORTS 4, Northeast Pacific. Source: K22 Table 1, page 3 (read in source).
- The same counts come out of the data file, with ANT split as ANT-XXIV/4 7, ANT-XXV/1 1, ANT-XXVI/4 18. Source: PANGAEA file (computed here).
- Tchla range and center per campaign (range, median, mean, mg m-3): ANT 0.033 to 4.15, 0.232, 0.648; NAAMES 0.094 to 0.987, 0.496, 0.540; RemSensPOC 0.049 to 1.09, 0.090, 0.173; SABOR 0.070 to 1.31, 0.252, 0.471; Tara Oceans 0.021 to 0.950, 0.168, 0.194; Tara Med 0.026 to 0.170, 0.055, 0.064; BIOSOPE 0.019 to 1.47, 0.069, 0.326; EXPORTS 0.172 to 0.292, 0.224, 0.228. Source: K22 Table 1, page 3 (read in source).
- Overall Tchla range 0.019 to 4.15 mg m-3, median 0.110. Source: K22 section 2.2, page 3 (read in source). Mean 0.311. Source: PANGAEA file (computed here).
- The authors describe the set as mostly open ocean from the Atlantic, Pacific, and Indian Oceans and the Mediterranean. Source: K22 section 2.2, page 3 (read in source).
- Coverage in the file: 41.2 S to 55.0 N, 158.0 W to 73.9 E, 2004-11-03 to 2018-09-07, depth 0 to 7 m. Source: PANGAEA file header (read in source).
- Radiometers: RAMSES for the ANT cruises; Satlantic HyperPro for NAAMES, SABOR, Tara, RemSensPOC, BIOSOPE, and EXPORTS. Source: https://doi.org/10.1594/PANGAEA.937536 landing page (read through a fetch summary). Processing references per campaign are Chase et al. (2017), Uitz et al. (2015), and Bracher et al. (2015a). Source: K22 section 2.2, page 3 (read in source).
- The file also carries a CTD temperature and salinity per sample: 5.7 to 30.3 C and 28.5 to 39.9. Source: PANGAEA file (computed here).

Tchla distribution (all computed here from the PANGAEA file)
- Below 0.05 mg m-3: 25 samples, 17.2 percent.
- Below 0.1: 67 samples, 46.2 percent.
- Below 0.2: 92 samples, 63.4 percent.
- Below 0.5: 125 samples, 86.2 percent.
- At or above 1.0: 11 samples, 7.6 percent. Maximum 4.151.

Each pigment's range in the 145 samples (min, median, mean, max in mg m-3, then the number of samples at zero; computed here)
- Tchla: 0.019, 0.110, 0.311, 4.151; 0 zeros.
- HexFuco: 0, 0.023, 0.071, 1.551; 1 zero.
- ButFuco: 0, 0.006, 0.018, 0.345; 14 zeros.
- Allo: 0, 0, 0.0096, 0.395; 102 zeros (70 percent).
- Fuco: 0, 0.006, 0.063, 1.789; 23 zeros.
- Perid: 0, 0, 0.0069, 0.065; 78 zeros (54 percent).
- Zea: 0.002, 0.042, 0.050, 0.353; 0 zeros.
- DVchla: 0, 0.013, 0.025, 0.146; 50 zeros.
- MVchlb: 0, 0.005, 0.034, 1.116; 48 zeros.
- Chlc12: 0.001, 0.008, 0.044, 1.157; 0 zeros.
- Chlc3: 0, 0.006, 0.027, 0.688; 9 zeros.
- Neo: 0, 0, 0.0039, 0.142; 107 zeros (74 percent).
- Viola: 0, 0.001, 0.0063, 0.253; 43 zeros.

Seasons
- Samples per calendar month, all hemispheres: Jan 7, Feb 0, Mar 5, Apr 15, May 16, Jun 1, Jul 15, Aug 24, Sep 30, Oct 5, Nov 17, Dec 10. Source: PANGAEA file (computed here).
- Northern Hemisphere samples only: Jan 7, Mar 3, Apr 3, May 15, Jun 1, Jul 15, Aug 24, Sep 27, Oct 4, Nov 3, Dec 1. Source: PANGAEA file (computed here).

Is any sample from the Gulf Stream, slope sea, Mid-Atlantic Bight, or Sargasso Sea?
- Yes, 16 samples lie inside 30 to 45 N, 80 to 50 W: 9 from SABOR (22 July to 5 August 2014) and 7 from RemSensPOC (3 to 8 September 2013). Source: PANGAEA file (computed here).
- SABOR positions with Tchla, temperature, salinity: 41.85 N 69.15 W, 0.696, 18.4 C, 31.3; 40.69 N 67.58 W, 0.429, 17.2 C, 32.6; 36.65 N 66.80 W, 0.070, 27.6 C, 36.3; 36.65 N 67.30 W, 0.084, 27.5 C, 36.2; 36.91 N 75.81 W, 1.051, 23.8 C, 28.5; 38.84 N 74.64 W, 0.252, 23.4 C, 31.3; 39.63 N 72.24 W, 0.106, 24.2 C, 33.1; 39.91 N 70.03 W, 0.238, 23.1 C, 33.7; 41.17 N 70.36 W, 1.312, 16.7 C, 32.0. Source: PANGAEA file (computed here).
- RemSensPOC positions: 39.67 N 51.70 W, 0.090, 24.6 C, 34.4; 37.65 N 54.19 W, 0.057, 27.0 C, 36.3; 35.71 N 56.62 W, 0.049, 27.0 C, 36.1; 33.60 N 59.25 W, 0.070, 27.5 C, 36.4; 32.36 N 62.02 W, 0.067, 27.8 C, 36.0; and two records at 31.67 N 64.17 W taken 9 minutes apart, both 0.087, 28.0 C, 36.1. Source: PANGAEA file (computed here).
- Of the 16 box samples, 9 have Tchla below 0.1 and 2 are above 1.0. Source: PANGAEA file (computed here).
- The 11 NAAMES samples sit at 39 to 54 N, 43 to 39 W, east of the box. Four of them fall inside a wider 25 to 50 N, 82 to 40 W box. Source: PANGAEA file (computed here).

### Inferences
- Reading the positions with their temperature and salinity: five SABOR samples are shelf water (Chesapeake outflow at salinity 28.5, the Delaware shelf, Georges Bank, and the Nantucket Shoals area), two are at the southern New England shelf break (salinity 33.1 and 33.7), and two are warm salty water at 36.6 N (Sargasso side of the stream). The RemSensPOC samples are a late-summer transect across the Sargasso Sea into Bermuda (the last position is the BATS area), with the first one, at salinity 34.4, in fresher water near 40 N 52 W. I see no sample that is clearly in the slope sea between the shelf break and the Gulf Stream north wall, and none I can place in the Gulf Stream core. Confidence: medium; this is water-mass reading from one surface T and S pair per station.
- The regional training support is roughly 9 shelf or shelf-break samples and 7 to 9 Sargasso-type samples, all from stratified summer conditions. Jerry's March 2024 to May 2026 record includes two full winters and three spring blooms for which the region contributed no training sample. The global set does contain spring North Atlantic samples (NAAMES May 2016 and March to April 2018, ANT April to May), but few.
- The two Bermuda records 9 minutes apart are near-duplicates. With random 75/25 splits, one can land in training and the other in testing, which inflates cross-validated skill slightly. The same holds for consecutive stations of one cruise in general (see question 6, Stock and Subramaniam).
- Allo and Neo passed the paper's 75 percent below-detection exclusion narrowly (70 and 74 percent zeros). Their models rest on about 43 and 38 non-zero samples.

### Gaps
- Table 1 lists eight campaigns and the file names ten cruise labels; the mapping is as given above, but I did not check individual station metadata against the original SeaBASS or PANGAEA records.
- I did not confirm the radiometer deployment mode (profiling, above-water, or buoy) per campaign.

## 3. Reported skill, and what the authors say is retrievable

### Takeaway
Cross-validated R2 runs from 0.37 (Zea) to 0.72 (Tchla), and the mean absolute error is 47 to 122 percent of the mean concentration for every pigment. The authors do not claim 13 independent retrievals: they say the pigments resolve into at most five covarying groups, and the paper never compares the retrievals with what Tchla covariation alone would predict.

### Cited Findings

Table 2, mean over 100 cross-validations on the held-out 25 percent, linear scale: R2 (SD), then MAD normalized to the mean retrieved concentration (SD). Source: K22 Table 2, page 8 (read in source).
- Tchla: 0.72 (0.15); 0.498 (0.127).
- Chlc12: 0.70 (0.13); 0.703 (0.235).
- Chlc3: 0.68 (0.13); 0.639 (0.212).
- Fuco: 0.65 (0.15); 0.844 (0.274).
- ButFuco: 0.62 (0.16); 0.588 (0.185).
- DVchla: 0.55 (0.12); 0.594 (0.103).
- HexFuco: 0.54 (0.16); 0.692 (0.201).
- Perid: 0.49 (0.13); 0.783 (0.166).
- MVchlb: 0.42 (0.19); 0.975 (0.295).
- Neo: 0.42 (0.21); 1.127 (0.354).
- Allo: 0.40 (0.19); 1.221 (0.400).
- Viola: 0.38 (0.18); 1.101 (0.370).
- Zea: 0.37 (0.10); 0.472 (0.071).

Other reported numbers
- The paper reports no bias statistic and no per-pigment slope table in the main text. Source: K22 Table 2 and section 3.4 (read in source). The training code does compute percent bias and RMSE per permutation. Source: `rrsModelTrain.m` (read through a fetch summary).
- Slopes: for the in-sample reconstruction (median of 100 models applied to all 145 samples), the slopes of measured against modeled for Tchla and five biomarker pigments, Zea excluded, are 0.74 to 0.94 with R2 0.51 to 0.73. The text does not say which pigment has which slope, and it leaves Zea out of the statement that the slopes are near one. Source: K22 section 3.4, page 8, and Figure 6 (read in source).
- The authors warn that those in-sample statistics are expected to be better than the cross-validated ones because training and validation data are not independent in that step. Source: K22 section 2.5, page 6 (read in source).
- Tchla from the reflectance model inversion (a different Tchla from the PCR one): R2 0.86 and slope 0.96 against HPLC, compared with 0.75 and 0.87 for OC4v6. Source: K22 section 3.2, page 6, and Figure 4 (read in source).
- No relation between the source campaign and reconstruction quality was seen. Source: K22 section 3.4, page 8 (read in source).

What the authors say about which pigments are retrievable
- Their summary is R2 of at least 0.5 for all pigments except Zea and the green algal pigments. By Table 2, Perid (0.49) and Allo (0.40) are also below 0.5. Source: K22 section 3.4, page 8 (read in source).
- For the poorly retrieved pigments (Zea, MVchlb, Neo, Viola, Allo) the authors' argument is not that the concentrations are accurate but that the modeled pigments still cluster into the same covarying groups. Source: K22 section 3.4, page 8 (read in source).
- The five groups, each standing behind one biomarker: Fuco (diatoms), Perid (dinoflagellates), HexFuco (haptophytes), MVchlb (green algae), Zea (cyanobacteria). Source: K22 section 3.1, page 6 (read in source).
- The HPLC covariation sets the ceiling: the modeling can identify the same five groups or fewer. Source: K22 section 4 opening, page 8, and section 4.1, page 10 (read in source).
- The modeled dendrogram needed a looser cutoff (linkage distance 0.80) than the HPLC one (0.65) to give five groups, and modeled Perid clustered nearer the cyanobacterial pigments, while measured Perid clustered nearer the diatom pigments. Source: K22 Figure 3 caption, page 5, and section 3.4, page 8 (read in source).

Does the paper say the pigments carry information beyond covariation with Tchla?
- It does not test it. There is no Tchla-only baseline, no partial correlation, and no comparison of retrieved pigment-to-Tchla ratios with measured ratios. Source: my reading of the whole of K22 (read in source; absence).
- It does say that the correlation spectra of the diatom pigments with the residual are almost the same as that of Tchla, and that cyanobacterial pigments often show the inverse of the Tchla pattern. Source: K22 section 3.3, pages 7 and 8 (read in source).
- It says the residual largely reflects how the Tchla-to-accessory-pigment relationship in the sample differs from the one built into the NOMAD-based aph model. Source: K22 section 4.5, page 11 (read in source).
- It attributes model skill across the whole spectrum to covariation among pigments and of pigments with other group-specific optical properties (fluorescence, scattering, packaging). Source: K22 section 4.4, page 11 (read in source).
- The method is described as empirical and able to reconstruct only the pigment communities present in its dataset; it describes a central tendency and is not meant to detect blooms dominated by one type. Source: K22 section 4.6, page 12 (read in source).
- The one direct quotation in this file, from the conclusions: "this model is limited to the dataset for which it was developed" (K22 section 5, page 12).
- A later paper by the same group states that three communities separate from the residual even after the dominant Tchla, NAP, and CDOM signals are removed. Source: K24 abstract (read through a search summary of the preprint page; the full text would not load).

Covariation inside the training set (computed here from the PANGAEA file; not published numbers)
- Squared linear correlation of each accessory pigment with HPLC Tchla, followed by the paper's cross-validated PCR R2: Chlc12 0.87 against 0.70; Chlc3 0.80 against 0.68; HexFuco 0.79 against 0.54; Fuco 0.78 against 0.65; Allo 0.68 against 0.40; Neo 0.67 against 0.42; ButFuco 0.62 against 0.62; MVchlb 0.61 against 0.42; Viola 0.56 against 0.38; Perid 0.33 against 0.49; Zea 0.12 against 0.37; DVchla 0.02 against 0.55.
- Spearman correlation of each pigment with Tchla: Chlc12 0.97, Chlc3 0.95, Fuco 0.92, HexFuco 0.89, ButFuco 0.87, MVchlb 0.84, Perid 0.77, Viola 0.75, Allo 0.73, Neo 0.69, Zea 0.45, DVchla 0.08.
- Zea against Tchla: Spearman +0.64 for samples below 0.2 mg m-3 and +0.08 at or above 0.2. The Zea to Tchla ratio falls with Tchla (Spearman -0.53). Inside the Gulf Stream box (n = 16) Zea against Tchla is -0.20, not significant.
- DVchla is above zero in 47 of 67 samples below 0.1 mg m-3, 44 of 58 between 0.1 and 0.5, and 4 of 20 at or above 0.5.

### Inferences
- On the training data themselves, a straight-line function of measured Tchla explains as much or more variance than the cross-validated SDP does for 9 of the 12 accessory pigments. Only Perid, Zea, and DVchla are retrieved better than their Tchla covariation would give. The comparison flatters the baseline (in-sample fit, measured Tchla) so it is not a fair contest, but it shows the published R2 values are compatible with most accessory pigments being retrieved through Tchla. This is the published-data counterpart of Jerry's local result that a pigment-against-Tchla law beats SDP on 11 of 12 accessory pigments, and of his finding that most SDP accessory pigments track SDP Tchla at r2 0.7 to 0.98. Confidence: medium to high for the direction, low for any exact count.
- The PCR predicts Tchla itself from the residual second derivative with R2 0.72. The residual step therefore does not remove the Tchla signal; the regression can and does recover it. Confidence: high.
- Zea. The paper's weakest pigment (R2 0.37, left out of the near-unity slope statement) is also the one whose training relationship with Tchla changes sign of usefulness across regimes: it rises with Tchla in clear water and is flat above 0.2 mg m-3, and DVchla is nearly absent above 0.5. Combined with the paper's note that cyanobacterial pigments correlate with the residual inversely to Tchla, the likely behavior is that modeled Zea acts as a clear-water index. In shelf and slope water where HPLC Zea (Synechococcus) rises with biomass in summer, such a model would run opposite to the truth, which matches Jerry's negative correlation. Confidence: medium; no published test.
- Tchla slope. A log-log slope near 0.72 is not at odds with the paper: its in-sample linear slopes already go as low as 0.74, the fit minimizes absolute error in linear units on a set where 86 percent of samples are below 0.5 mg m-3, truncated-component regression shrinks predictions toward the mean, and negatives are clipped to zero. Confidence: medium.
- Reading normalized MAD as a size: a typical error of half the mean concentration for the best pigments and more than the mean for Allo, Neo, and Viola, before any satellite noise is added.

### Gaps
- Per-pigment slopes, bias, and RMSE: not in the main text; the supplement and Figure 6 panels were not digitized.
- No published SDP statistics on log scale.
- No published comparison of SDP with a Tchla-only baseline, by the authors or anyone else, that I could find.

## 4. What the paper tested about resolution, noise, and satellite use, and what it left untested

### Takeaway
The paper tested one thing relevant to PACE: retraining the model on in situ spectra degraded to 5 nm and 10 nm. It tested no added noise, no satellite data, no atmospheric correction error, no regional hold-out, and no composited Rrs, and it says so in part.

### Cited Findings
- Spectral resolution: the analysis was replicated at 5 nm and 10 nm. At 5 nm the same five groups separate and skill is comparable (Table S2, Figures S9 to S12); at 10 nm results are notably worse for all pigments (Table S3). Source: K22 section 4.3, pages 10 and 11 (read in source).
- The stated conclusion is that the model works best at 1 to 5 nm and degrades at 10 nm or coarser. Source: K22 section 5, page 12 (read in source).
- Noise: no noise experiment. The authors state that derivative methods amplify noise, that in situ Rrs has a much lower noise-to-signal ratio than satellite Rrs, and that spatiotemporal aggregation of satellite Rrs will likely be required to reach a tolerable signal-to-noise level. Source: K22 section 4.1, page 10 (read in source).
- Atmospheric correction: not tested. The authors expect its errors to be spectrally broad and therefore largely removed by the residual and derivative, and they call for strict quality control of PACE spectra. Source: K22 section 4.1, page 10 (read in source).
- Quality control dependence: 19 percent of in situ spectra were rejected by eye for noise before training, and the authors say model success depended partly on this. Source: K22 section 2.2, page 3, and section 5, page 12 (read in source).
- Optically complex water: the authors say regimes outside the dataset, naming inland and coastal waters, may need combined or different approaches, and that departures from the global base state will have divergent optics. Source: K22 section 4 opening, page 10 (read in source).
- Regional transfer: in other ecosystems the reconstructed pigments might cluster into different groups. Source: K22 section 4.6, page 12 (read in source). The companion paper says a global algorithm should be applied regionally only with its limits understood, and that a global model applied to a coastal region would likely not separate diatoms from dinoflagellates. Source: KS19 section 4.3, page 7572 (read in source).
- Low chlorophyll: nothing specific. The paper notes only that its matchups are skewed toward lower Tchla and higher Zea than the 4,480-sample global set, with means for Tchla, Fuco, Perid, and HexFuco significantly different (p < 0.01), and that further optimization may be needed to apply the model to all samples in the larger set. Source: K22 section 4.2, page 10 (read in source).
- UV: excluded because not all campaigns measured it, though the authors expect useful information there. Source: K22 section 4.1, page 10 (read in source).
- Satellite application: none in the paper. It was published two years before PACE launched.

### Inferences
- The 5 nm test retrained the regression on 5 nm inputs. It is not a test of what PACE users do, which is to interpolate OCI spectra to 1 nm and apply coefficients trained on 1 nm in situ data. The paper's finding that the model works at 5 nm does not cover that case. Confidence: high on the distinction; unknown on the size of the effect.
- The authors' own expectation that aggregation is needed is an argument for composited or binned input, but they did not test whether an 8-day mean of Rrs behaves like an instantaneous spectrum once it is passed through a non-linear inversion and a second difference. Averaging Rrs over days with different water and different correction errors is not the same as averaging pigments.
- The assumption that atmospheric correction error is spectrally smooth is asserted, not shown. Any band-to-band artifact at the 5 to 10 nm scale (for example near gas absorption features) would pass straight into the second difference.
- Visual rejection of noisy spectra has no stated equivalent for satellite pixels. The published code has no quality gate beyond dropping NaN.

### Gaps
- Table S2 and S3 values (per-pigment skill at 5 and 10 nm) were not retrieved.
- No statement anywhere in the paper on multi-day composites, BRDF normalization, or differences between in-water and satellite Rrs definitions.

## 5. Later work to October 2026

### Takeaway
As of 13 October 2026 there is no official PACE SDP product and no published validation of SDP on OCI data. What exists is the authors' MATLAB code, a Python port with the original 100 coefficient sets, a NASA tutorial (updated 24 July 2026) that runs that port on Level-2 granules, and repeated statements that the method is still being implemented. I found no retrained coefficients and no guidance on Level 2 against Level 3.

### Cited Findings

Papers by the authors
- K24 (Optics Express, 2024): community detection on the Rrs residual of a paired global dataset finds three optical communities; 74 percent of samples get the same community from optics as from HPLC pigment ratios. Source: K24 abstract via https://preprints.opticaopen.org/articles/preprint/Phytoplankton_communities_quantified_from_hyperspectral_ocean_reflectance_correspond_to_pigment-based_communities/25800382 (read through a search summary).
- The count is 120 of 162 samples, and the three communities are labeled haptophytes, cyanobacteria, and diatoms; the dataset is now N = 162. Source: Feb 2025 slides and Sep 2025 slides (read in source).
- One confused sample shown in the talk had a residual community of diatoms plus dinoflagellates plus greens, an HPLC community of cyanobacteria, and 16S plus flow cytometry showing Prochlorococcus. Source: Sep 2025 slides (read in source).
- Kramer et al. (2024), Toward a synthesis of phytoplankton community composition methods for global-scale application, Limnology and Oceanography: Methods, https://doi.org/10.1002/lom3.10602, compares HPLC with other methods. Source: https://www.sashakramer.me/publications (read through a fetch summary). I did not read the paper.
- No 2025 or 2026 paper by this group applies SDP to OCI data, as far as the lab's publication list shows. Source: https://www.sashakramer.me/publications (read through a fetch summary, page state on 2026-10-13).

Status statements
- 20 February 2025, PACE science team slides (Siegel, Maritorena, Kramer): the group was implementing SDP for PACE using example residual spectra from PACE V2 data; next steps were testing with V3 data, comparing to PVST pigments, refining SDP, and testing regional against global. Source: Feb 2025 slides (read in source).
- 24 September 2025, NOAA seminar (Kramer): SDP is being implemented for PACE; the residual spectra are planned as a product; comparison with PVST HPLC is still listed as future work. Source: Sep 2025 slides (read in source).
- 26 March 2025, NOAA seminar (Chase): SDP and GPig are the two pigment approaches being applied to OCI; there is a slide stating that most pigments are correlated with chlorophyll a (citing Kramer 2019 and Chase 2017); ongoing work with NASA Goddard to implement pigment algorithms as PACE products and to evaluate them and characterize uncertainty. Source: https://www.star.nesdis.noaa.gov/star/documents/seminardocs/2025/20250326_Chase.pdf (read in source).
- 26 August 2025, PACE news on the V3.1 release: several phytoplankton community composition products are at various stages of integration and test; none is named or released there. Source: https://pace.oceansciences.org/news_more.htm?id=92 (read through a fetch summary).
- Summer 2026 PACE newsletter: ocean color products reprocessed to V3.2; how-to guides listed for SDP, GPig, and MOANA. Source: https://pace.oceansciences.org/newsletter/Summer2026.html (read through a fetch summary).
- The PACE matchup page covers Rrs, chlorophyll a, and aerosol optical depth, and says composition products will be added as validation data grow. Source: https://pace.oceansciences.org/pace_data_matchups.htm (read through a search summary).

Code and coefficients
- Authors' MATLAB code: https://github.com/sashajane19/Rrs_pigments, with `Kramer_hyperRrs.m`, `Kramer_Rrs_pigments.m`, `rrsModelTrain.m`, the aph A and B coefficient file, the water absorption file, and a 17-spectrum EXPORTS North Atlantic test set; Zenodo DOI 10.5281/zenodo.20298745; it cites both K22 and K24. Source: repository page (read through a fetch summary).
- Python port: https://github.com/max-danenhower/rrs-SDP-pigments, by Max Danenhower from Kramer's MATLAB. It ships `original_a_coefs.xlsx` and `original_c_coefs.xlsx` with 100 coefficient sets per pigment, has `generate_coefficients`, `run_sdp` (expects 400 to 700 nm at 1 nm with SST and SSS), and `sdp_from_pace` (takes a PACE Level-2 AOP file with SMAP salinity and GHRSST temperature files). Source: repository page and `src/sdp/core.py` (read through a fetch summary).
- The Kramer lab page says the SDP code is being implemented for PACE and is meanwhile public in MATLAB and Python at those two addresses. Source: https://www.sashakramer.me/publications (read through a fetch summary).
- NASA l2gen: `get_sdp.c` exists in OCSSW, credited to L. McKinna, dated 19 December 2022, citing K22, with 13 pigments, 299 coefficients, and 100 replicates. Source: https://oceancolor.gsfc.nasa.gov/docs/ocssw/get__sdp_8c_source.html (search-engine summary only; the page redirected on fetch, and the documentation seen was for an older OCSSW version).

The NASA tutorial
- Title: Applying Spectral Derivative Pigments (SDP) Phytoplankton Community Composition Algorithm to OCI data, by Anna Windle and Ian Carroll, last updated 24 July 2026. It says SDP is currently being implemented in OBPG's OCSSW software. Source: NASA tutorial (read through a fetch summary).
- It runs the Python port's `sdp_from_pace` on one Level-2 AOP granule (V3.2 NRT, 2026-05-05) over a box at 37.5 to 40.5 N, 73.5 to 67 W, with JPL SMAP 8-day salinity and MUR SST. Source: NASA tutorial (read through a fetch summary). That box is the Mid-Atlantic Bight and slope sea.
- It gives no validation, no uncertainty, no flags, and no statement on Level-3 or composite input. Its run log shows 110,331 interpolated pixels reduced to 70,338 processed, and runtime warnings for overflow and invalid values in a power operation, which the page tells the reader to consider, without explanation. Source: NASA tutorial (read through a fetch summary).

How the Python port treats PACE input (read through a fetch summary of `src/sdp/core.py`; confirm before relying on it)
- OCI Rrs is linearly interpolated to a 1 nm grid from 346 to 719 nm, cut to 400 to 700 nm, and smoothed with a 5-point rolling mean that, as written, uses the pandas default trailing window, not a centered one.
- No Level-2 flags are applied. Pixels are dropped only if Rrs at 400 nm, SST, or SSS is NaN. Negative Rrs is not screened.
- Outputs are 13 variables with no units or uncertainty attributes.

### Inferences
- The warnings in NASA's own run are what an unbounded simplex fit produces when it steps to negative Tchla inside Tchla^B. Those pixels end as NaN, which would explain part of the pixel loss. Confidence: medium.
- If the trailing 5-point window is real, every spectral feature is shifted by 2 nm relative to a centered mean before the second difference is taken. Whether the training spectra were smoothed with a centered window is not something I could confirm. This is a question for whoever audits the code. Confidence: low until the line is read directly.
- The word original in the coefficient file names and the 299-by-100 layout suggest the public coefficients are still the 145-sample K22 set. The talks show pigment scatter plots that are labeled as including the new K24 data, so a 162-sample refit may exist, but I found no released coefficient set from it.
- The absence of any Level-3 path in either the authors' code or NASA's tutorial means running SDP on 8-day 4 km Level-3 Rrs is a use nobody upstream has documented or tested.

### Gaps
- K24 full text: the publisher page and the preprint page would not load. Its dataset description, limits section, and any refit of the pigment model are unread.
- The current OCSSW source and any SDP coefficient file in the OCSSW data tree were not inspected.
- The PACE data products table is rendered by script and returned no rows, so I could not confirm the listed status of any pigment product there.
- No PVST comparison of SDP has been published or posted that I could find.

## 6. Independent evaluations and the degrees-of-freedom literature

### Takeaway
No independent paper tests SDP against HPLC outside its training set. The information-content literature says hyperspectral Rrs holds about four independent quantities, of which two carry about 90 percent of the variance, and fewer once satellite noise and averaging are added. That is far below 13, and it is consistent with the authors' own ceiling of three to five groups.

### Cited Findings

Degrees of freedom
- C23 analyzed in situ hyperspectral Rrs at 191 stations, 400 to 700 nm, from the same data family as SDP (it cites Chase et al. 2017 and K22 for the dataset). Standardized PCA with a broken-stick rule gives 4 degrees of freedom; the first four components explain 54, 33, 8, and 2 percent. Source: C23 pages 2 to 4 (read in source).
- The MODIS-equivalent 10-band version also has 4; the SeaWiFS-equivalent 6-band version has 3. Multispectral bands predict each hyperspectral wavelength below 578 to 582 nm to within 5 percent. Source: C23 page 4 and Figure 3 (read in source).
- C23 reads the first two modes, about 90 percent of variance, as meaning only two quantities can be estimated accurately, with a third and fourth at low signal-to-noise. Source: C23 page 4 (read in source).
- Satellite climatologies of multispectral Rrs have 2 degrees of freedom, and a suite of six derived products has 1. Source: C23 pages 5 and 6 (read in source).
- C23 expects PACE Rrs to have fewer degrees of freedom than the in situ set because of its larger uncertainty, and calls the result a fundamental challenge to, or a ceiling on, algorithms that extract functional types from Rrs. Source: C23 pages 4 and 7 (read in source).
- C23's caveats: PCA is linear and may overcount; the analysis targets broad first-order variation; coastal and optically complex water likely has more degrees of freedom; UV bands, polarimetry, and added variables such as SST could raise the count; and fine spectral resolution may be most useful for outliers and rare events, such as blooms or data binned over coherent features like eddies. Source: C23 pages 3, 6, and 7 (read in source).
- Cael, Chase, Boss (2020), Applied Optics 59, 3971, https://doi.org/10.1364/AO.389189: global hyperspectral particulate absorption spectra have about 5 degrees of freedom, which C23 treats as a possible upper limit for satellite Rrs. Source: C23 page 2 (read in source).
- K22 itself cites Cael et al. (2020) for the point that strong correlation between nearby wavelengths dilutes the statistical power of individual wavelengths. Source: K22 section 4.3, page 10 (read in source).

Retrieval degeneracy
- PF25: Rrs depends on the ratio of backscattering to absorption, so non-water absorption and backscattering cannot be separated without priors. Multispectral data support no more than three parameters; hyperspectral PACE-like data support four or at most five. Without strong priors (for example on the CDOM slope) a phytoplankton absorption parameter is hard to extract. Source: PF25 abstract, section 3.2, and conclusions (read through a fetch summary).
- PF25 cites K22 once, in its conclusions, among data-driven methods, with the caution that a learned model cannot distinguish degenerate solutions, especially with noise. Source: PF25 conclusions (read through a fetch summary).
- The 2024 preprint of that paper (arXiv 2408.06149, under a different title) was reported by a search summary to say that K22 remains limited by the same degeneracy. I could not confirm that wording in the preprint text. Source: https://arxiv.org/abs/2408.06149 (search summary only).

Evaluations of related empirical pigment algorithms
- Stock and Subramaniam (2020), Frontiers in Marine Science 7, 599, https://doi.org/10.3389/fmars.2020.00599 (local PDF in Zotero): for eight diagnostic pigments from multispectral data, spatial block cross-validation gave a much less confident picture than random fivefold cross-validation. Only fucoxanthin and zeaxanthin were predicted in untrained regions with considerably lower error than a constant null model. Random hold-out underestimates error when samples cluster along cruise tracks. Source: abstract, page 1 (read in source).
- Moffat et al. (2026), arXiv 2608.23348, submitted 24 August 2026: with 33,640 HPLC matchups and multispectral reflectance, machine-learning models beat chlorophyll-only baselines overall, but gains are small for pigments strongly correlated with chlorophyll a. Source: https://arxiv.org/abs/2608.23348 (read through a fetch summary of the abstract page; preprint, not peer reviewed).
- Chase (2025 seminar) lists the correlation of most pigments with chlorophyll a as a standing problem for both SDP and the Gaussian (GPig) approach. Source: https://www.star.nesdis.noaa.gov/star/documents/seminardocs/2025/20250326_Chase.pdf (read in source).
- Cetinic et al. (2024), Remote Sensing of Environment 302, 113964, https://doi.org/10.1016/j.rse.2023.113964 (local PDF in Zotero) cites K22 for the 5 nm resolution result and lists it among pigment-concentration approaches. It reports no independent test of it. Source: local PDF, text near the discussion of spectral interval (read in source).

### Inferences
- Thirteen outputs from a signal with about four degrees of freedom means the 13 pigments are, at best, about four independent numbers re-expressed 13 ways. The authors' own results agree in spirit: five clusters from pigments, three communities from the residual. For Jerry this bounds what a pigment-to-pigment contrast can mean: differences among pigments inside one covarying group are not independent evidence.
- C23's count is based on variance in raw standardized Rrs. SDP works on the second difference of a residual, which deliberately discards the high-variance broad modes. A small-variance component can still carry real pigment information, so "four" is not a hard proof that nothing beyond Tchla is retrievable. It is a statement that whatever lies beyond the first two modes is weak and noise-sensitive. Confidence: medium.
- C23's remark on eddies cuts in Jerry's favor in one respect: averaging many pixels over a coherent feature raises signal-to-noise for whatever weak modes exist. It does not remove bias that comes from applying a global model outside its training regime.
- K22's validation is random 75/25 splitting across cruises. By Stock and Subramaniam's result, its R2 values are optimistic for a region or season the training set does not cover. Confidence: medium to high.

### Gaps
- No published out-of-sample test of SDP against HPLC was found (searches for PACE validation, PVST, intercomparison of SDP, GPig, and MOANA returned none). Jerry's own PVST, BATS, and PAX comparisons appear to be the only such tests in existence.
- I did not read Chase et al. (2017) directly. A search summary of a critic's reading says its residuals showed no strong relation to pigments and that it leans on chlorophyll covariation; I could not confirm this in the paper.
- Cael et al. (2020) was read only through C23 and a search summary.

## 7. Kramer and Siegel (2019) and Kramer, Siegel and Graff (2020): what HPLC covariation can separate

### Takeaway
HPLC pigment ratios separate four groups globally (cyanobacteria, diatoms with dinoflagellates, haptophytes, green algae), up to five or six in regional data, and five in the 145-sample SDP set. The authors state plainly that an algorithm validated on such data cannot resolve more groups than the data do, and that a global model applied regionally will miss locally important groups.

### Cited Findings
- KS19 dataset: 4,480 surface samples (7 m or shallower) from 66 cruises, 2000 to 2018, six laboratories, Tchla 0.006 to 26 mg m-3 with median 0.31, plus 1,607 samples from six time-series sites. Source: KS19 sections 2.1, 2.2, and 3.1 (read in source).
- Method: hierarchical clustering (correlation distance, Ward linkage) and EOF analysis on 17 accessory pigments normalized to Tchla. Source: KS19 sections 2.3 and 2.4 (read in source).
- Why ratios: nearly all pigments are positively correlated with Tchla and with each other in absolute concentration, the exceptions being DVchla and DVchlb, which correlate only with each other and Zea. Normalizing to Tchla removes most of those pairwise correlations. Source: KS19 section 3.1, page 7563, and Figure 2 (read in source).
- Global result: four groups at linkage cutoff 1.0. Diatom and dinoflagellate pigments (Fuco, Perid) form one group; haptophytes are marked by HexFuco and Chlc3; green algae by MVchlb with Pras and other carotenoids; cyanobacteria by DVchla and Zea. Allo (cryptophytes) falls in with the green algal group. Source: KS19 section 3.3, page 7565, and Table 2 (read in source).
- Global EOFs: the first four modes explain 24.4, 14.7, 11.5, and 8.7 percent; six modes explain 72 percent. Only mode 1 relates to environment (R2 0.36 with temperature and with nitrate). Source: KS19 section 3.4, pages 7565 to 7567 (read in source).
- Local results: four groups at MVCO and Bowdoin Buoy, five at BOUSSOLE, CARIACO, and Plumes and Blooms, six at Palmer. Dinoflagellate pigments separate from diatom pigments at all six sites. Cyanobacterial pigments are not detected at MVCO, Palmer, or Bowdoin Buoy. Source: KS19 section 3.5 and Table 2 (read in source).
- MVCO (Martha's Vineyard, the closest site to Jerry's shelf water): 278 samples, median Tchla 1.95, diatom-dominated, four groups (haptophytes, dinoflagellates, green algae, diatoms), median Zea to Tchla 0.017. Source: KS19 Table 2, page 7564 (read in source).
- Stated limits: HPLC supports at most four groups globally and up to six locally; a model built on the global set can detect at most four; a global algorithm applied to a coastal region would probably not separate diatoms from dinoflagellates; a global algorithm should be applied regionally only with these limits understood; models built on present-day data are limited in detecting future change. Source: KS19 section 4.3, page 7572 (read in source).
- Suggested biomarkers per group: Fuco, HexFuco, MVchlb, and DVchla or Zea. Source: KS19 section 4.3, page 7572 (read in source).
- KSG20: 229 surface samples from four NAAMES cruises (November 2015, May to June 2016, August to September 2017, March to April 2018) across subarctic, temperate, subtropical, and Sargasso Sea provinces of the western North Atlantic. Clustering and EOFs give five groups (diatoms, dinoflagellates, haptophytes, green algae, cyanobacteria); network community detection gives four, with diatoms and dinoflagellates merged. Source: KSG20 abstract and results (read through a fetch summary).
- KSG20: cryptophytes (Allo) correlate with green algal pigments and never form their own group; a future satellite model validated on these data could retrieve at most five communities; collinearity among pigments is what prevents finer separation. Source: KSG20 discussion (read through a fetch summary).
- How K22 connects this to SDP: the HPLC covariation in the matchup set constrains the number of groups any model built on it can identify; five groups appear in the 145 samples, one more than globally because diatom and dinoflagellate pigments separate; and conclusions from KS19 carry over to this subset. Source: K22 section 3.1, page 6, and section 4.2, page 10 (read in source).
- The Sep 2025 talk restates it: only four pigment groups separate globally, five in the western North Atlantic, a maximum of five in the paired dataset, and three communities from the residual. Source: Sep 2025 slides (read in source).

### Inferences
- The groups SDP can resolve are, on the authors' own account, at most: diatom-type (Fuco, Chlc12), dinoflagellate (Perid), haptophyte (HexFuco, ButFuco, Chlc3), green algal plus cryptophyte (MVchlb, Neo, Viola, Allo), and cyanobacterial (Zea, DVchla). In K24 this drops to three, with dinoflagellates and greens folded into the diatom community. Pigments inside one group should not be read as separate signals.
- Jerry's cluster notes (cryptophyte and green algal pigments joining) agree with KS19 and KSG20. A cluster joining HexFuco with Zea is not something the HPLC papers show globally or in the western North Atlantic.
- KSG20 is the best published statement of real pigment covariation in Jerry's wider region and season range, but its Rrs matchups in the SDP training set are only 11, all east of 50 W.

### Gaps
- KSG20 was read through a fetch summary of the first 100,000 characters; figure-level details are not checked.
- The local-scale PANGAEA file for KS19 (905882, present locally) was not analyzed.

## 8. Candidate issues for the Gulf Stream application (real issues only)

### Takeaway
Six issues rest on evidence strong enough to report. The first three bear directly on whether SDP accessory pigments mean anything beyond Tchla in this region; the last three are about untested steps between the paper and PACE Level-3 data.

### Cited Findings
Each item gives the evidence, the size where known, and my confidence that it is a real problem for this application.

- Issue A. Most accessory pigments are probably retrieved through their covariation with Tchla.
  - Evidence: no Tchla-only baseline in K22; diatom pigment correlation spectra nearly identical to Tchla's (K22 section 3.3); Tchla alone explains as much or more training-set variance than cross-validated SDP for 9 of 12 accessory pigments (computed here, question 3); about four degrees of freedom in hyperspectral Rrs (C23); the practitioners' own slide that most pigments are correlated with chlorophyll a (Chase 2025).
  - Size: published R2 0.38 to 0.70 for accessory pigments against Tchla-covariation r2 of 0.56 to 0.87 for the same nine pigments in the same data.
  - Confidence: high that this limits interpretation; it also matches Jerry's local 11-of-12 result.
- Issue B. Zea and DVchla are the pigments SDP adds information on in training, and they are the least transferable.
  - Evidence: Zea has the lowest R2 (0.37) and is excluded from the near-unity slope statement (K22 section 3.4); in training Zea follows Tchla only below 0.2 mg m-3, and DVchla is nearly absent above 0.5 (computed here); cyanobacterial pigments respond inversely to Tchla in the residual (K22 section 3.3); cyanobacterial pigments are undetected at the nearest coastal time series (KS19 Table 2, MVCO).
  - Size: unknown in general; Jerry measured a negative correlation with HPLC Zea.
  - Confidence: medium that the cause is regime dependence of the Zea to Tchla relationship; high that Zea from SDP should not be used north of the stream without a local check.
- Issue C. Seasonal and regional training gaps.
  - Evidence: 16 box samples, all July to September, 9 of them below 0.1 mg m-3; no slope-sea sample identified; 11 samples in the whole set at or above 1 mg m-3 (computed here, question 2); the conclusions state the model is limited to its dataset (K22 section 5); regional application caveat (KS19 section 4.3); random-split validation overstates skill in unsampled regions (Stock and Subramaniam 2020).
  - Size: zero regional training samples for roughly 9 of 12 months.
  - Confidence: high that winter and spring-bloom retrievals in shelf and slope water are extrapolation.
- Issue D. The printed equations do not match the code that produced the coefficients.
  - Evidence: adg slope sign and bbp ratio band (question 1).
  - Size: about 25 percent difference in the CDOM slope in clear water between the two forms.
  - Confidence: high that the mismatch exists; whether it affects Jerry depends on which form his implementation follows, which the code audit should settle.
- Issue E. No test of the PACE path: interpolated 2.5 nm sampling, satellite noise, atmospheric correction, composites.
  - Evidence: the 5 nm test was a retraining on in situ data (K22 section 4.3); no noise test and an explicit expectation that aggregation is needed (K22 section 4.1); 19 percent of ship spectra were rejected by eye for noise before training (K22 section 2.2); no validation in the NASA tutorial; no published OCI validation (question 5).
  - Size: unknown.
  - Confidence: high that it is untested; unknown how much it degrades results.
- Issue F. Error size even under ideal conditions.
  - Evidence: K22 Table 2.
  - Size: mean absolute error of 47 to 122 percent of the mean concentration on in situ spectra with matched HPLC. Eddy-scale pigment anomalies smaller than that are within the method's stated error before any satellite effect.
  - Confidence: high.

### Inferences
- Taken together, A and the degrees-of-freedom work suggest the defensible SDP outputs for this region are Tchla-like biomass and perhaps one contrast along the clear-water (cyanobacterial) to productive axis. Anything finer needs local HPLC support.
- Composited Level-3 input is in line with the authors' call for aggregation, so it is not by itself a violation of the method. It is simply unverified.

### Gaps
- No evidence either way on whether SDP pigment-to-Tchla ratios (as opposed to concentrations) carry real signal on satellite data.
- Plausible but unverified, listed so they are not mistaken for findings: the trailing smoothing window in the Python port; fluorescence in the residual behaving differently for satellite composites; differences between in-water radiometer Rrs and BRDF-normalized satellite Rrs.

## 9. Checked and found sound

### Takeaway
Several parts of the method hold up, and some of Jerry's local observations are what the published record predicts.

### Cited Findings
- The training data are public, complete, and match the paper: 145 rows, the same campaign counts, Tchla range and median as Table 1 and section 2.2, 301 Rrs columns from 400 to 700 nm, no negative or missing Rrs. Source: PANGAEA file (computed here) against K22 Table 1.
- The aph parameterization is independent of the matchups (NOMAD), so the residual is not circular with the HPLC used for training. Source: K22 section 2.3, page 3 (read in source).
- The reflectance-model Tchla is credible on in situ data: R2 0.86, slope 0.96, better than OC4v6 on the same samples. Source: K22 section 3.2 and Figure 4 (read in source).
- The validation uses held-out data and reports the cross-validated statistics as the headline, and the authors flag that the in-sample figure statistics are optimistic. Source: K22 section 2.5 (read in source).
- The authors are explicit about the limits: empirical, tied to its dataset, open-ocean, base-state, not for blooms of one type, needs quality control and probably aggregation for satellites. Jerry's concerns are ones the paper itself raises. Source: K22 sections 4, 4.1, 4.6, and 5 (read in source).
- SST and SSS are a minor input. They touch only seawater backscattering, and the paper itself used a climatology, so using satellite or climatological T and S is not a departure from the method. Source: K22 section 2.3 (read in source).
- The pipeline order in the public code (interpolate, smooth, cut to 400 to 700, invert, residual, second difference, 100 coefficient sets, median, clip at zero) matches the paper's description. Source: K22 sections 2.2 to 2.5 against the Python repo (read through a fetch summary).
- The region is not absent from training: 16 summer samples span shelf to Sargasso water, and the western North Atlantic pigment covariation (KSG20) is the five-group structure SDP assumes. Source: PANGAEA file (computed here); KSG20.
- PACE version currency: V3.2 is the current ocean color reprocessing as of the Summer 2026 newsletter, and the NASA tutorial ran on a V3.2 granule. Source: https://pace.oceansciences.org/newsletter/Summer2026.html and NASA tutorial (read through fetch summaries).

### Inferences
- A Tchla slope below one, a weak or wrong-signed Zea, and accessory pigments that follow Tchla are all consistent with the published design and statistics. They do not by themselves indicate an implementation bug.

### Gaps
- Sound here means consistent with the paper. None of these checks is a validation on PACE data.
