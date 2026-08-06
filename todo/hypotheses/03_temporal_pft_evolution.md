# Hypothesis 3: Does PFT Community Composition Evolve Over an Eddy's Lifetime?

## Hypothesis

Ecological succession theory predicts a temporal sequence within mesoscale eddies:
1. **Early stage** (eddy formation, nutrient pulse): Diatom-dominated community - diatoms are fast-growing r-strategists that capitalize on nutrient injections
2. **Mid stage** (nutrient drawdown): Transition to haptophytes and dinoflagellates - these mixotrophs can supplement photosynthesis with grazing
3. **Late stage** (nutrient-depleted, stratified core): Cyanobacteria steady-state - small-celled K-strategists adapted to oligotrophic conditions

If the eddy core is truly isolated from surrounding waters (as a coherent vortex), this succession should unfold over weeks to months.

## Background From Our Analysis

We analyzed **cyclone track 20** from the Gulf Stream region:
- **Lifetime**: 121 days (Jan–May 2025)
- **PACE observations**: 19 dates with valid data (out of ~120 days, due to cloud cover)
- **Analysis performed**: Radial PFT profiles (spatial structure) - we found all PFTs depleted at the center
- **Analysis NOT yet performed**: Temporal evolution of PFT fractions within the eddy

The spatial analysis revealed lateral stirring as the dominant process, but this does not rule out temporal succession occurring *on top of* the stirring pattern.

## What's Needed

1. **Time-series extraction**: For each eddy and each date, compute the mean PFT fraction (or radial profile) - this gives a time-series of community composition per eddy
2. **Trend analysis**: Test for monotonic trends using **Spearman rank correlation** (robust to outliers and non-normality) between eddy age (days since formation) and each PFT fraction
3. **Changepoint detection**: Look for abrupt shifts in community composition using methods like PELT (Pruned Exact Linear Time) or Bayesian changepoint analysis
4. **Multi-eddy pooling**: A single eddy (N=19 time points) has low statistical power - pool across eddies of similar age/polarity to increase sample size

## Confounding Factors

These must be accounted for before attributing temporal patterns to ecological succession:

- **Seasonal cycle**: Jan→May spans winter deep mixing → spring bloom → summer stratification. The entire regional phytoplankton community shifts during this period, independent of eddy dynamics. Must detrend or compare eddy interior vs exterior to isolate eddy-specific effects.
- **Cloud coverage gaps**: PACE observations are irregularly spaced due to clouds - time-series are not evenly sampled. Use methods robust to irregular sampling (e.g., Spearman on actual dates, not indices).
- **Eddy radius changes**: Eddies can grow, shrink, merge, or split over their lifetime - the spatial footprint changes, which affects which PACE pixels are included.
- **Eddy translation**: As the eddy moves, it encounters different background waters - changes in PFT fractions may reflect the eddy moving through a gradient rather than internal succession.
- **Lateral exchange**: If the eddy is not perfectly coherent (i.e., water enters/exits the core), succession is disrupted by continuous external input.

## Proposed Approach

### Single-Eddy Analysis (Exploratory)
1. Plot time-series of mean PFT fractions for cyclone track 20 (already have the data)
2. Compute Spearman correlations between date and each PFT fraction
3. Visually inspect for changepoints or regime shifts

### Multi-Eddy Pooling Strategy

Pool across eddies to overcome per-eddy sparsity (cloud gaps leave most individual eddies with too few valid dates for a useful trend), but stratify on confounders so water-mass and seasonal effects don't swamp the successional signal.

**What to split eddies on:**
- **Polarity**: cyclone vs anticyclone
- **Season**: by eddy formation month (or meteorological seasons DJF/MAM/JJA/SON)
- **Origin relative to the Gulf Stream** - four categories, defined using a distance-from-axis threshold (km) to the mean Gulf Stream position:
  1. Definitively north - formed and stayed north of the axis
  2. Definitively south - formed and stayed south of the axis
  3. Meandered south → north - formed south, later crossed to north
  4. Meandered north → south - formed north, later crossed to south

  Northern (slope water, nutrient-rich) and southern (Sargasso, oligotrophic) sides of the Gulf Stream carry very different background PFT communities. Pooling without this split conflates water-mass effects with succession. The two crossing categories are themselves interesting - cross-frontal transport may itself drive compositional change.

**Age axis** - compute trajectories against both:
- **Absolute age** (days since formation): preserves real-time growth/succession rates in ecological units
- **Normalized age** (fraction of total lifetime, 0→1): enables comparison at the "same developmental stage" across eddies with different lifespans

These answer different questions (how fast does succession happen vs how far along is the eddy in its lifecycle) and may give different results - worth running both.

**Analysis steps:**
1. For each eddy, compute PFT fractions (or radial profiles) at each available date
2. Compute both absolute and normalized age for each observation
3. Classify each eddy into one of the four origin categories
4. Within each (polarity × season × origin) stratum, pool all observations and fit PFT vs age
5. Mixed-effects model: `PFT ~ age + polarity + season + origin + (1|eddy_id)` - eddy_id as random effect to handle repeated measurements within the same eddy

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
- [ ] Gulf Stream axis/front definition (mean SSH contour or SST front position) with a distance threshold in km
- [ ] Per-eddy origin classification into the four categories (formation position + trajectory crossing test)
- [ ] Statistical framework: Spearman, mixed-effects models, changepoint detection
