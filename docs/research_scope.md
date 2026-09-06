# Gulf Stream eddy research scope

Scope date: 2026-09-05.

This project studies how near-surface chlorophyll-a concentration (CHL) and pigment composition change with eddy age and distance from the eddy center. The target cohorts are southbound cyclones and northbound anticyclones near the Gulf Stream. The mechanism question concerns lateral entrainment of exterior water and possible vertical nutrient supply.

The manuscript outline is in Obsidian: `Work Vault/Work/1. GT Oceanography Lab/Manuscript Planning.md`. That note owns the paragraph and figure plan. This file records the current scientific scope and the limits of the repository.

## Research questions

1. **CHL change within eddies:** How does CHL change over the observed lifetime of each target cohort? The hypothesis predicts preferential CHL loss near cyclone boundaries and gain near anticyclone boundaries. Test each prediction against both interior concentration and local exterior water.
2. **Radial change with age:** Does the CHL profile from the center through two eddy radii change with age? Does a boundary change precede a core change? Compare absolute age and normalized age, with uncertainty from independent eddies.
3. **Pigment composition:** Do fucoxanthin (Fuco), zeaxanthin (Zea), and divinyl chlorophyll-a (DVChla), relative to total chlorophyll-a (TChla), change with age and radius? Compare each ratio with its concentration and with the measured local source-water contrast.
4. **Mechanism evidence:** Do source-facing boundary properties approach exterior-water properties before the core does? Evaluate this evidence for lateral entrainment against wind-related vertical motion, seasonal change, and other biological explanations.

Lateral entrainment means that exterior water enters an eddy. Formation source water means water retained at detachment. Exterior donor water means adjacent water that could enter later. Ambient reference water defines the local anomaly. These roles can involve different locations.

Horizontal transport around its perimeter can produce a pigment pattern without inward exchange. Eddy-induced Ekman pumping means vertical motion associated with interactions between wind stress and an eddy. Polarity alone does not determine which process dominates.

Phytoplankton functional types (PFTs) are groups defined by ecological function. PhytoClass is inactive for now. Its package code remains available, but the pipeline does not estimate taxonomic class contributions. A functional interpretation needs additional assumptions, pigment retrieval accuracy, and a stable pigment-to-class conversion. A fixed diatom-to-haptophyte-to-cyanobacteria sequence is not an assumption of this study. A Kuroshio comparison is an optional extension after the Gulf Stream analysis meets its evidence requirements.

## Literature basis

- [Gaube and McGillicuddy (2017)](https://doi.org/10.1016/j.dsr.2017.02.006) distinguish attached Gulf Stream meanders from detached rings. Their north/south meander transport and post-formation CHL results motivate separate tests of origin, motion, and within-eddy change.
- [Cai et al. (2025)](https://doi.org/10.3389/fmars.2025.1608635) motivate the radial question. Their Figure 12 shows four eddy examples, not a universal cyclone or anticyclone profile.
- [Dove and Freilich (2026)](https://doi.org/10.5670/oceanog.2026.e111) already combine SWOT and PACE for Gulf Stream plankton analysis. The proposed contribution concerns eddy age, radial structure, and source-water evidence, rather than the satellite combination alone.

These studies motivate hypotheses. They do not establish the result in the current dataset.

## Definitions that constrain interpretation

- **Cohort identity:** Geographic southward or northward motion, endpoint sides of the Gulf Stream, and source-water origin are separate properties. A ring can first appear across the front from its source water. An endpoint-side class does not verify detachment.
- **Eddy identity:** Use `(polarity, track_id)`. Track numbers overlap between the two polarity datasets.
- **Age:** Use the physical track timeline. First and last clear PACE observations do not define the lifetime. First and last track detections do not necessarily establish physical formation and termination.
- **Radius:** State whether the radius describes the effective contour or the contour of maximum speed. Compare radius choices and retain true contour membership for irregular eddies. An outer circular annulus is not the actual boundary; use contour distance for the boundary test.
- **Local anomaly:** A CHL difference from exterior water is distinct from a temporal change within the eddy. Interior CHL can increase while its local anomaly remains negative.
- **Pigment fraction:** Define each annular ratio from area-weighted concentration means on identical valid pixels. Compare mean pixel ratios as sensitivity. A ratio can change through its numerator or denominator. Pigments indicate broad composition; they do not uniquely identify species.
- **Independent evidence:** Sea surface temperature (SST) and sea surface salinity (SSS) also enter the Spectral Derivative Pigments (SDP) algorithm. Pigment covariance with those inputs is not independent confirmation of water-mass exchange.
- **Independent sample:** Summarize each eddy once per age-radius cell, then give each eddy equal weight. Treat an eddy as the primary unit. Retain its repeated dates and radial zones together in uncertainty estimates. Account for dependence between nearby tracks when necessary.

## Adopted experiment methods

The canonical experiment is `gulf_stream_20240305_20260531`. Its configuration and data directories use that same name.

- **Open-ocean mask:** Before eddy identification, require finite absolute dynamic topography (`adt`), eastward velocity (`ugos`), northward velocity (`vgos`), and `relative_vorticity`. Keep cells at least eight grid cells from invalid data. Exclude the Great Lakes rectangle from 81°W to 75°W and 40°N to 44°N. The eight-cell distance is approximately 1° on this grid.
- **Speed contour:** Use the contour with the greatest mean speed around the eddy for PACE pixel selection and eddy dynamics.
- **Annual cycle:** Let `d` be the number of days since January 1. Store `cos(2*pi*d/365)` and `sin(2*pi*d/365)` as `time_of_year_cos` and `time_of_year_sin`. These two variables represent the annual cycle without separate seasonal categories.
- **CHL source:** The lifetime analysis uses the `CHL` field of the Copernicus Marine daily L3 plankton product (multi-sensor GlobColour processing, 4 km), composited over NASA 8-day periods, as its only CHL product. PACE `chlor_a` and SDP TChla do not appear in the lifetime notebook; the product comparison that ran on the matched composites was removed on 2026-09-06. Decided 2026-09-06 on coverage. At 50% interior coverage and 10 valid pixels, the southbound-cyclone and northbound-anticyclone cohort has 783 Copernicus composites from 2023-04 to 2025-12, against 314 with an SDP retrieval and 240 with PACE `chlor_a`; a match of all three products keeps 176. Within the PACE `chlor_a` table span, which begins 2024-10-02, Copernicus still holds 366 cohort composites against 240, and PACE has no data before March 2024. Copernicus also observes every cohort eddy from its first detection, so the change-from-first-observation baseline is the first detection for every eddy; a three-product match moves that baseline for 21 of 33 eddies. The gain is in composite coverage and record length, not daily coverage: the daily product passes the same thresholds on 46% of cohort eddy-days, so the analysis stays composite-based.

The [open-ocean mask](../src/eddy_tracking/downloads/swot.py#L82), [contour selection](../collocate_pace.py#L113), and [annual cycle variables](../build_gold_table.py#L217) implement these methods.

## Verified repository capability

The audit below describes source and configuration on 2026-09-05. It does not certify existing output tables or manuscript results.

| Area | Current behavior | Consequence for the research |
| --- | --- | --- |
| Physical product | The canonical configuration selects AVISO DUACS-MIOST Level 4 v3.0, which merges SWOT and nadir altimetry. | Describe the gridded product, not native SWOT swaths. |
| Domain and dates | Longitude 81°W-56°W, latitude 29°N-44°N; altimetry 2023-03-28 to 2025-12-31; requested PACE interval 2024-03-05 to 2026-05-31. | The match interval is 2024-03-05 to 2025-12-31. Authenticated checks on 2026-09-06 confirmed the altimetry limit and PACE composites through 2026-05-25 to 2026-06-01. |
| Optical product | The downloader selects PACE OCI `PACE_OCI_L3M_AOP` at 4 km. The canonical experiment uses eight-day composites. | A composite is not an instantaneous eddy observation. |
| Track criteria | The canonical configuration permits three virtual days and requires at least 60 days per track. | The sample favors persistent features and needs an endpoint audit. |
| Collocation | The stage selects the speed contour near the composite midpoint and applies an 80% interior coverage threshold. | It does not extract a complete exterior through two radii or measure annular coverage. |
| Gold table | The table contains interior pigment means, physical metadata, and ratios to one regional background per date. | It lacks the local-reference, radial, and directional responses required by the research questions. |
| CHL | SDP supplies TChla. An experimental script also reads PACE BGC `chlor_a`. | The direct CHL script needs validation, including its obsolete RRS filename expression. It is outside the default production path. |
| Motion | `NN`, `NS`, `SN`, and `SS` describe the first and last nonvirtual observations relative to the daily axis. | They do not establish net latitude change, detachment, or material-water history. |
| PFTs | PhytoClass is inactive. Only its package code remains. | The pipeline does not produce PFT estimates. |
| Mechanisms | The pipeline supplies radius, amplitude, front distance, annual cycle variables, and Rossby diagnostics. | It does not compute wind-based vertical motion, mixed-layer depth, or nutrient supply. |

Source references: [canonical configuration](../configs/gulf_stream_20240305_20260531/config.yaml#L11), [PACE product selection](../src/eddy_tracking/downloads/pace.py#L28), [contour selection](../collocate_pace.py#L113), [interior means](../build_gold_table.py#L39), [regional background](../background.py#L300), [endpoint classes](../gulf_stream.py#L296), [track age](../build_gold_table.py#L204), [experimental CHL path](../scripts/collocate_pace_chl.py#L73), and [SDP auxiliary inputs](../run_sdp.py#L89).

## Evidence requirements

The main CHL analysis requires a validated direct CHL product, exterior radial pixels, area-aware estimates, and a local reference for each eddy and date. Compare the direct CHL product with SDP TChla without treating them as interchangeable measurements. Choose annulus widths from effective product resolution. Quantify the independent eddy count, annular coverage, and uncertainty across age. Audit direction, endpoint evidence, domain exits, and track discontinuities before a lifetime interpretation.

The composition analysis requires pigment-specific uncertainty and evidence beyond a CHL-only baseline. Use independent pigment observations where available. Check retrieval dependence on SST, SSS, and TChla before a biological interpretation. Pigment covariance does not establish independently resolved PFTs.

The lateral mechanism test requires measured exterior donor-water contrasts, directional structure, and temporal order at a resolvable scale. Test surface heat and freshwater fluxes, front displacement, and contour changes as alternatives to inward exchange. The vertical mechanism test requires a physical diagnostic and suitable temporal support. A lag alone does not demonstrate nutrient supply. State unresolved alternatives when surface observations cannot separate them.

The cohort comparison requires controls for season, front position, source side, and incomplete observations. Age and season can remain inseparable in a short record. Report a restricted inference when the data lack overlap. A comparison between southbound cyclones and northbound anticyclones estimates a cohort difference, not an isolated polarity effect.

Existing notebooks contain exploratory radial and statistical analyses. Their stored results remain historical until the current products, cohort definitions, and uncertainty methods support them. Operational setup, data-download code, model code, configuration, and historical notebooks retain their existing roles.
