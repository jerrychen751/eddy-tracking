# Hypothesis 3: Does PFT Community Composition Evolve Over an Eddy's Lifetime?

## Hypothesis

Ecological succession theory predicts a temporal sequence within mesoscale eddies:
1. **Early stage** (eddy formation, nutrient pulse): Diatom-dominated community — diatoms are fast-growing r-strategists that capitalize on nutrient injections
2. **Mid stage** (nutrient drawdown): Transition to haptophytes and dinoflagellates — these mixotrophs can supplement photosynthesis with grazing
3. **Late stage** (nutrient-depleted, stratified core): Cyanobacteria steady-state — small-celled K-strategists adapted to oligotrophic conditions

If the eddy core is truly isolated from surrounding waters (as a coherent vortex), this succession should unfold over weeks to months.

## Background From Our Analysis

We analyzed **cyclone track 20** from the Gulf Stream region:
- **Lifetime**: 121 days (Jan–May 2025)
- **PACE observations**: 19 dates with valid data (out of ~120 days, due to cloud cover)
- **Analysis performed**: Radial PFT profiles (spatial structure) — we found all PFTs depleted at the center
- **Analysis NOT yet performed**: Temporal evolution of PFT fractions within the eddy

The spatial analysis revealed lateral stirring as the dominant process, but this does not rule out temporal succession occurring *on top of* the stirring pattern.

## What's Needed

1. **Time-series extraction**: For each eddy and each date, compute the mean PFT fraction (or radial profile) — this gives a time-series of community composition per eddy
2. **Trend analysis**: Test for monotonic trends using **Spearman rank correlation** (robust to outliers and non-normality) between eddy age (days since formation) and each PFT fraction
3. **Changepoint detection**: Look for abrupt shifts in community composition using methods like PELT (Pruned Exact Linear Time) or Bayesian changepoint analysis
4. **Multi-eddy pooling**: A single eddy (N=19 time points) has low statistical power — pool across eddies of similar age/polarity to increase sample size

## Confounding Factors

These must be accounted for before attributing temporal patterns to ecological succession:

- **Seasonal cycle**: Jan→May spans winter deep mixing → spring bloom → summer stratification. The entire regional phytoplankton community shifts during this period, independent of eddy dynamics. Must detrend or compare eddy interior vs exterior to isolate eddy-specific effects.
- **Cloud coverage gaps**: PACE observations are irregularly spaced due to clouds — time-series are not evenly sampled. Use methods robust to irregular sampling (e.g., Spearman on actual dates, not indices).
- **Eddy radius changes**: Eddies can grow, shrink, merge, or split over their lifetime — the spatial footprint changes, which affects which PACE pixels are included.
- **Eddy translation**: As the eddy moves, it encounters different background waters — changes in PFT fractions may reflect the eddy moving through a gradient rather than internal succession.
- **Lateral exchange**: If the eddy is not perfectly coherent (i.e., water enters/exits the core), succession is disrupted by continuous external input.

## Proposed Approach

### Single-Eddy Analysis (Exploratory)
1. Plot time-series of mean PFT fractions for cyclone track 20 (already have the data)
2. Compute Spearman correlations between date and each PFT fraction
3. Visually inspect for changepoints or regime shifts

### Multi-Eddy Analysis (Confirmatory)
1. For each eddy, compute PFT fractions at each available date
2. Normalize time axis to "eddy age" (days since formation)
3. Bin by eddy age (e.g., 0–30 days, 30–60 days, 60–90 days, 90+ days)
4. Compare mean PFT fractions across age bins, stratified by polarity
5. Use mixed-effects models: PFT ~ eddy_age + polarity + season + (1|eddy_id)

### Control for Seasonality
- Compare PFT trends inside eddies vs the surrounding background water at the same dates
- If both show the same trend, it is seasonal, not successional
- The eddy-specific signal is the *difference* between interior and exterior trends

## Key Questions

- Does diatom fraction decrease with eddy age (consistent with succession)?
- Does cyanobacteria fraction increase with eddy age (consistent with oligotrophication)?
- Is the temporal signal detectable above the noise of seasonal change and cloud gaps?
- Do cyclones and anticyclones show different temporal trajectories?

## Prerequisites

- [ ] Multi-eddy PACE pixel extraction pipeline (shared with Hypothesis 1)
- [ ] Time-series extraction per eddy per date
- [ ] Background (non-eddy) PFT time-series for seasonal detrending
- [ ] Statistical framework: Spearman, mixed-effects models, changepoint detection
