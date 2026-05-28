# Pigment Clustering → Adaptive PFT Resolution for PhytoClass

## Motivation

Our PhytoClass F matrix assumes 6 resolvable PFTs, but the data may not support
separating all of them. Pigments shared across PFTs (e.g., zeaxanthin in both
Cyanobacteria and Green algae) can make certain PFTs indistinguishable in a given
region. We should let the data tell us how many PFTs are actually resolvable.

Inspired by Kramer & Siegel (2019) and the general principle that claim resolution
should match data resolution.

## Method

1. Compute pigment correlation matrix (Spearman) across all collocated samples
2. Hierarchical clustering on pigments (columns, not samples) — group pigments
   that co-vary tightly
3. Cut the dendrogram at varying heights; identify stable, distinct clusters
4. Map each pigment cluster → PFT using the diagnostic pigment within that cluster
   (e.g., peridinin → Dinoflagellates, alloxanthin → Cryptophytes)
5. Compare number of resolvable clusters to our current 6-PFT F matrix

## Key decision: pre-merge in the F matrix

If clustering shows that two PFTs are not separable (e.g., Cyanobacteria and
Green algae share zeaxanthin and cluster together), **merge their rows in
f_matrix.csv and min_max.csv BEFORE running PhytoClass** (option 2, not post-hoc
summing). This:
- Reduces SA parameter space → lower RMSE, more stable estimates for all PFTs
- Avoids the optimizer wasting effort on an impossible decomposition
- Produces a merged "Cyanobacteria + Green algae" super-group whose Tchla
  contribution is well-constrained even though within-group attribution isn't

## What this gives us

- A principled, data-driven justification for our PFT model complexity
- Stronger claims: "our data resolves N distinct PFT groups" backed by clustering
- Region-aware: different experiments/regions may support different PFT counts
- Reviewers can't challenge PFT separation we never claimed to make

## Status

- [ ] Implement pigment correlation + clustering diagnostic (notebook)
- [ ] Generate dendrogram for Gulf Stream dataset
- [ ] Determine which PFTs are resolvable vs. need merging
- [ ] Build region-specific F matrix with merged rows if needed
- [ ] Re-run PhytoClass with adapted F matrix
- [ ] Compare results (merged vs. original 6-PFT) for stability/RMSE
