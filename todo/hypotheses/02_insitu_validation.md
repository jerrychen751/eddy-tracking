# Hypothesis 2: Validate PFT Retrievals Against Published Gulf Stream Pigment Data

## Goal

Confirm that our SDP-derived (Spectral Decomposition of Phytoplankton absorption) PFT fractions from PACE L3 remote sensing reflectance are consistent with known Gulf Stream phytoplankton ecology, using published in-situ pigment and phytoplankton composition data.

## Background

We used the SDP model to decompose PACE OCI hyperspectral Rrs (346–719 nm, 172 bands) into fractional contributions of 7 phytoplankton functional types:
- **Diatoms**: ~5–8% of total absorption
- **Haptophytes**: dominant group, ~25–35%
- **Cyanobacteria**: ~2–3%
- **Dinoflagellates**: ~15–20%
- **Green algae**: ~10–15%
- **Cryptophytes**: ~5–10%
- **Mixed/unresolved**: remainder

These fractions were computed for the Gulf Stream region (~30°N, 75–77°W) during Jan–May 2025.

## What to Validate Against

### In-Situ Data Sources
- **HPLC pigment surveys**: BATS (Bermuda Atlantic Time-series Study), or transects from research cruises crossing the Gulf Stream. HPLC pigments (e.g., fucoxanthin for diatoms, 19'-hexanoyloxyfucoxanthin for haptophytes, zeaxanthin for cyanobacteria) can be converted to PFT fractions via CHEMTAX or diagnostic pigment analysis
- **BGC-Argo floats**: Biogeochemical Argo floats with Chl-a fluorescence and backscatter in the western North Atlantic — some carry fluorometers that allow PFT inference
- **Prior satellite studies**: SeaWiFS/MODIS-based PFT climatologies for the Gulf Stream region (e.g., Hirata et al. 2011 global PFT algorithm, Bracher et al. 2017 PhytoDOAS)

### Specific Checks
1. **Diatom fractions (~5–8%)**: Are these reasonable for the oligotrophic side of the Gulf Stream? Gulf Stream itself is relatively nutrient-poor, but its north wall borders productive slope waters — expect low but non-zero diatoms
2. **Cyanobacteria fractions (~2.5%)**: Consistent with subtropical western North Atlantic? Prochlorococcus dominates numerically but contributes less to absorption due to small cell size — low SDP fraction may be physically reasonable
3. **Haptophyte dominance**: Coccolithophores and other haptophytes are typically dominant in the western North Atlantic transition zone — high fractions are expected
4. **Seasonal pattern**: Jan–May spans winter mixing through spring bloom onset — expect increasing diatom fractions in spring if mixing deepens the nutricline

## Approach

1. **Literature search**: Find published PFT or pigment composition studies in the Gulf Stream / western North Atlantic (30–40°N, 70–80°W)
2. **Extract comparable metrics**: Convert published HPLC pigments to PFT fractions using the same or comparable methods
3. **Compare**: Overlay our mean PFT fractions against reported ranges — are we within published uncertainty?
4. **Document discrepancies**: Note any PFTs where our retrieval seems anomalous and investigate potential causes

## Caveats

- **SDP model training**: The SDP model was trained on a global HPLC + Rrs matchup dataset — it may have regional biases if the Gulf Stream was underrepresented in training data
- **Absorption vs biomass**: SDP recovers fractional absorption, not biomass or cell counts — comparison with HPLC requires careful unit alignment
- **Temporal mismatch**: In-situ observations from different years/seasons may not directly compare to 2025 satellite data — use climatological ranges rather than exact matches
- **Depth integration**: Satellite Rrs reflects the upper ~1 optical depth (~20m in open ocean) — in-situ profiles may sample deeper

## Prerequisites

- [ ] Literature review of Gulf Stream PFT/pigment studies
- [ ] Access to BATS or relevant cruise HPLC datasets (publicly available via BCO-DMO or SeaBASS)
- [ ] Comparison framework: table of PFT fractions (ours vs published) with uncertainty ranges
