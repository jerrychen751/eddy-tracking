# Seasonality Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/seasonality.ipynb` — exploratory analysis of seasonal patterns in eddy pigments and PFT composition across Gulf Stream eddies.

**Architecture:** Three-layer notebook: pooled seasonal overview → polarity-split seasonality → per-eddy calendar trajectories. Reuses the same data loading pattern as `polarity_comparison.ipynb`. No new utility code.

**Tech Stack:** pandas, numpy, matplotlib, scipy, py-eddy-tracker (for zarr loading)

**Spec:** `~/.claude/projects/-Users-jerry-Desktop-school-research-eddy-tracking/specs/2026-03-31-seasonality-notebook-design.md`

---

### Task 1: Setup & Data Loading

**Files:**
- Create: `notebooks/seasonality.ipynb`

- [ ] **Step 1: Create notebook with imports and constants**

```python
# Cell 1
import sys
from pathlib import Path
import datetime as dt

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

EXPERIMENT = "gulf_stream_20241001_20250701"
COLORS = {"cyclone": "tab:blue", "anticyclone": "tab:red"}
DIAGNOSTIC_PIGMENTS = ["Fuco", "HexFuco", "Perid", "Zea", "DV chla", "Allo", "MV chlb"]
PFT_COLS = ["Diatoms", "Dinoflagellates", "Haptophytes", "Cryptophytes", "Green_algae", "Cyanobacteria"]
```

- [ ] **Step 2: Load pigment data**

```python
# Cell 2
frames = []
for polarity in ("cyclone", "anticyclone"):
    pigments_dir = ROOT / "outputs" / EXPERIMENT / "pigments" / polarity
    for fp in sorted(pigments_dir.glob("eddy_*_pigments.csv")):
        df = pd.read_csv(fp, parse_dates=["date"])
        df["polarity"] = polarity
        frames.append(df)

pigments = pd.concat(frames, ignore_index=True)
pigment_cols = [c for c in pigments.columns
                if c not in ["track_id", "date", "pixel_lon", "pixel_lat",
                             "center_lon", "center_lat", "coverage", "polarity"]]
print(f"{len(pigments)} pigment pixels, {pigments.groupby('polarity')['track_id'].nunique().to_dict()}")
```

Expected output: `471305 pigment pixels, {'anticyclone': 34, 'cyclone': 42}`

- [ ] **Step 3: Load PFT data**

```python
# Cell 3
pft_frames = []
for polarity in ("cyclone", "anticyclone"):
    pft_dir = ROOT / "outputs" / EXPERIMENT / "pft" / polarity
    for fp in sorted(pft_dir.glob("eddy_*_pfts.csv")):
        df = pd.read_csv(fp, parse_dates=["date"])
        df["polarity"] = polarity
        pft_frames.append(df)

pfts = pd.concat(pft_frames, ignore_index=True)
print(f"{len(pfts)} PFT pixels, {pfts.groupby('polarity')['track_id'].nunique().to_dict()}")
```

- [ ] **Step 4: Load track properties from zarr**

```python
# Cell 4
def load_track_props(polarity):
    PET_EPOCH = dt.date(1950, 1, 1)
    zarr_path = ROOT / "outputs" / EXPERIMENT / "eddy_track" / polarity / f"{polarity}_tracks.zarr"
    tracked = TrackEddiesObservations.load_file(str(zarr_path))

    unique_ids = np.unique(tracked.track)
    lifetime = {}
    for tid in unique_ids:
        t = tracked.time[tracked.track == tid]
        lifetime[tid] = int(t.max() - t.min()) + 1

    dates = [PET_EPOCH + dt.timedelta(days=int(d)) for d in tracked.time]
    return pd.DataFrame({
        "track_id": tracked.track.astype(int),
        "date": pd.to_datetime(dates),
        "polarity": polarity,
        "center_lon": (tracked.longitude + 180) % 360 - 180,
        "center_lat": tracked.latitude,
        "radius_km": tracked.radius_e / 1000,
        "amplitude_m": tracked.amplitude,
        "speed_avg": tracked.speed_average,
        "lifetime_days": [lifetime[tid] for tid in tracked.track],
    })

track_props = pd.concat(
    [load_track_props(p) for p in ("cyclone", "anticyclone")],
    ignore_index=True,
)
```

- [ ] **Step 5: Merge track properties, add temporal columns, filter bad retrievals**

```python
# Cell 5
pigments = pigments.merge(
    track_props[["track_id", "date", "polarity", "radius_km", "amplitude_m", "lifetime_days"]],
    on=["track_id", "date", "polarity"],
    how="left",
)

# Per-eddy-date spatial medians for pigments
pig_medians = (
    pigments.groupby(["track_id", "date", "polarity"])[pigment_cols]
    .median()
    .reset_index()
)

# Per-eddy-date spatial medians for PFTs
pft_medians = (
    pfts.groupby(["track_id", "date", "polarity"])[PFT_COLS]
    .median()
    .reset_index()
)

# Filter retrieval failures
pig_medians = pig_medians[pig_medians["T chla"] >= 0.01].copy()
pft_medians = pft_medians[pft_medians[PFT_COLS].sum(axis=1) > 0].copy()

# Temporal columns
SEASON_MAP = {10: "Fall", 11: "Fall", 12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer"}
SEASON_ORDER = ["Fall", "Winter", "Spring", "Summer"]

for df in (pig_medians, pft_medians):
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)

print(f"Pigment eddy-dates after filter: {len(pig_medians)} (dropped {816 - len(pig_medians)} bad retrievals)")
print(f"PFT eddy-dates after filter: {len(pft_medians)}")
```

- [ ] **Step 6: Run all cells and verify output**

Run the notebook. Verify:
- Pigment and PFT frames load without errors
- Track properties merge produces no unexpected NaNs in radius_km
- `pig_medians` has ~770-810 rows (816 total minus ~10-40 bad retrievals)
- `month` ranges 1-6 and 10-12, `season` has 4 values

---

### Task 2: Pooled Seasonal Overview (Layer 1)

**Files:**
- Modify: `notebooks/seasonality.ipynb`

- [ ] **Step 1: Add sample size table**

```python
# Cell 6 — markdown
# ## Pooled seasonal overview

# Cell 7
MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6]
MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

pig_with_month = pigments.copy()
pig_with_month["month"] = pig_with_month["date"].dt.month

sample_sizes = []
for m in MONTH_ORDER:
    sub = pig_medians[pig_medians["month"] == m]
    px = pig_with_month[pig_with_month["month"] == m]
    sample_sizes.append({
        "month": MONTH_LABELS[MONTH_ORDER.index(m)],
        "n_eddies": sub["track_id"].nunique(),
        "n_eddy_dates": len(sub),
        "n_pixels": len(px),
    })

sample_df = pd.DataFrame(sample_sizes)
sample_df
```

- [ ] **Step 2: Add monthly T chla box plots**

```python
# Cell 8
monthly_groups = [pig_medians[pig_medians["month"] == m]["T chla"].values for m in MONTH_ORDER]

fig, ax = plt.subplots(figsize=(10, 4))
bp = ax.boxplot(monthly_groups, labels=MONTH_LABELS, patch_artist=True,
                medianprops=dict(color="black"))
for patch in bp["boxes"]:
    patch.set_facecolor("lightblue")
ax.set_xlabel("Month")
ax.set_ylabel("T chla")
ax.set_title("Monthly T chla (all eddies)")
```

- [ ] **Step 3: Add monthly diagnostic pigment heatmap**

```python
# Cell 9
monthly_pig = pig_medians.groupby("month")[DIAGNOSTIC_PIGMENTS].median()
monthly_pig = monthly_pig.loc[MONTH_ORDER]

# Row-normalize so pigments with different magnitudes are comparable
normed = monthly_pig.apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(normed.values.T, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(MONTH_ORDER)))
ax.set_xticklabels(MONTH_LABELS)
ax.set_yticks(range(len(DIAGNOSTIC_PIGMENTS)))
ax.set_yticklabels(DIAGNOSTIC_PIGMENTS)
ax.set_xlabel("Month")
fig.colorbar(im, ax=ax, label="Normalized concentration")
ax.set_title("Diagnostic pigments by month (row-normalized)")
```

Note: `apply` with `axis=1` normalizes each **row** (each month). But we want each **pigment** (column) normalized across months. Fix: transpose before normalizing.

Corrected normalization:

```python
normed = monthly_pig.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
```

This normalizes each column (pigment) independently across months, so the heatmap shows when each pigment peaks relative to its own range.

- [ ] **Step 4: Add seasonal PFT stacked bar chart**

```python
# Cell 10
seasonal_pft = pft_medians.groupby("season")[PFT_COLS].mean()
seasonal_pft = seasonal_pft.loc[SEASON_ORDER]

# Normalize rows to fractions summing to 1
seasonal_frac = seasonal_pft.div(seasonal_pft.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
seasonal_frac.plot(kind="bar", stacked=True, ax=ax, width=0.6)
ax.set_ylabel("PFT fraction")
ax.set_title("Community composition by season")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xticklabels(SEASON_ORDER, rotation=0)
fig.tight_layout()
```

- [ ] **Step 5: Run all Layer 1 cells and verify**

Run cells 6-10. Verify:
- Sample size table shows reasonable counts per month (Oct–Jun)
- Box plots show a seasonal arc (expect higher Tchla in winter/spring)
- Heatmap has no all-NaN rows or columns
- Stacked bars sum to ~1.0 for each season

---

### Task 3: Polarity-Split Seasonality (Layer 2)

**Files:**
- Modify: `notebooks/seasonality.ipynb`

- [ ] **Step 1: Add monthly T chla box plots by polarity**

```python
# Cell 11 — markdown
# ## Polarity-split seasonality

# Cell 12
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

for ax, pol in zip(axes, ("cyclone", "anticyclone")):
    sub = pig_medians[pig_medians["polarity"] == pol]
    groups = [sub[sub["month"] == m]["T chla"].values for m in MONTH_ORDER]
    bp = ax.boxplot(groups, labels=MONTH_LABELS, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[pol])
        patch.set_alpha(0.5)
    ax.set_xlabel("Month")
    ax.set_title(pol.capitalize())

axes[0].set_ylabel("T chla")
fig.suptitle("Monthly T chla by polarity", y=1.02)
fig.tight_layout()
```

- [ ] **Step 2: Add monthly diagnostic pigment profiles by polarity**

```python
# Cell 13
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, pig in enumerate(DIAGNOSTIC_PIGMENTS):
    ax = axes.flat[i]
    for pol in ("cyclone", "anticyclone"):
        sub = pig_medians[pig_medians["polarity"] == pol]
        monthly = sub.groupby("month")[pig]
        medians = monthly.median().reindex(MONTH_ORDER)
        q25 = monthly.quantile(0.25).reindex(MONTH_ORDER)
        q75 = monthly.quantile(0.75).reindex(MONTH_ORDER)

        x = range(len(MONTH_ORDER))
        ax.plot(x, medians.values, "o-", color=COLORS[pol], label=pol, ms=4)
        ax.fill_between(x, q25.values, q75.values, color=COLORS[pol], alpha=0.15)

    ax.set_xticks(range(len(MONTH_ORDER)))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_title(pig)

axes.flat[0].legend()
axes.flat[-1].set_visible(False)
fig.suptitle("Diagnostic pigments by month and polarity (median + IQR)", y=1.01)
fig.tight_layout()
```

- [ ] **Step 3: Add seasonal PFT composition by polarity**

```python
# Cell 14
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

for ax, season in zip(axes, SEASON_ORDER):
    for j, pol in enumerate(("cyclone", "anticyclone")):
        sub = pft_medians[(pft_medians["season"] == season) & (pft_medians["polarity"] == pol)]
        means = sub[PFT_COLS].mean()
        fracs = means / means.sum()
        bottom = 0
        for k, pft in enumerate(PFT_COLS):
            ax.bar(j, fracs[pft], bottom=bottom, color=f"C{k}",
                   label=pft if (season == "Fall" and j == 0) else None)
            bottom += fracs[pft]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cyc", "Anti"], fontsize=9)
    ax.set_title(season)

axes[0].set_ylabel("PFT fraction")
axes[0].legend(bbox_to_anchor=(-0.3, 1), loc="upper right", fontsize=8)
fig.suptitle("PFT composition by season and polarity", y=1.02)
fig.tight_layout()
```

- [ ] **Step 4: Run all Layer 2 cells and verify**

Run cells 11-14. Verify:
- Both polarity box plots render with correct colors
- Pigment profile lines show one line per polarity per subplot, with IQR shading
- Stacked bars have two bars per season panel, each summing to ~1.0
- Legend appears only once per figure (not duplicated per subplot)

---

### Task 4: Per-Eddy Trajectories (Layer 3)

**Files:**
- Modify: `notebooks/seasonality.ipynb`

- [ ] **Step 1: Compute pooled monthly median for reference line**

```python
# Cell 15 — markdown
# ## Per-eddy trajectories

# Cell 16
pooled_monthly = pig_medians.groupby("month")["T chla"].median()
```

- [ ] **Step 2: Add per-eddy T chla spaghetti plot on calendar axis**

```python
# Cell 17
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, pol in zip(axes, ("cyclone", "anticyclone")):
    sub = pig_medians[pig_medians["polarity"] == pol]
    for _, grp in sub.sort_values("date").groupby("track_id"):
        ax.plot(grp["date"], grp["T chla"], color=COLORS[pol], alpha=0.25, lw=0.8)

    # Pooled monthly median as reference
    ref_dates = [dt.date(2025 if m <= 6 else 2024, m, 15) for m in MONTH_ORDER]
    ax.plot(ref_dates, pooled_monthly.reindex(MONTH_ORDER).values,
            "k--", lw=2, label="pooled monthly median")

    ax.set_title(f"{pol.capitalize()} (n={sub['track_id'].nunique()})")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 2)
    ax.legend(fontsize=8)

axes[0].set_ylabel("T chla (spatial median)")
fig.suptitle("Eddy T chla over calendar time", y=1.02)
fig.autofmt_xdate()
fig.tight_layout()
```

- [ ] **Step 3: Select case study eddies**

```python
# Cell 18
# Eddies spanning >= 3 calendar months with >= 8 PACE observations
candidates = pig_medians.groupby(["track_id", "polarity"]).agg(
    n_obs=("date", "size"),
    first=("date", "min"),
    last=("date", "max"),
).reset_index()
candidates["months_spanned"] = (
    (candidates["last"].dt.year - candidates["first"].dt.year) * 12
    + candidates["last"].dt.month - candidates["first"].dt.month + 1
)
candidates = candidates[(candidates["n_obs"] >= 8) & (candidates["months_spanned"] >= 3)]
candidates = candidates.sort_values("months_spanned", ascending=False)

# Pick 3-4 per polarity, spread across different start months
for pol in ("cyclone", "anticyclone"):
    sub = candidates[candidates["polarity"] == pol]
    print(f"\n{pol}: {len(sub)} candidates")
    print(sub[["track_id", "n_obs", "first", "last", "months_spanned"]].head(8).to_string(index=False))
```

After inspecting the output, manually pick 3-4 eddies per polarity that span different seasonal windows. Store them:

```python
# Adjust these IDs after inspecting the candidates table above
CASE_CYCLONES = [5, 12, 22]  # placeholder — replace with actual picks
CASE_ANTICYCLONES = [1, 4, 16]  # placeholder — replace with actual picks
```

- [ ] **Step 4: Add case study small-multiples**

```python
# Cell 19
case_ids = {
    "cyclone": CASE_CYCLONES,
    "anticyclone": CASE_ANTICYCLONES,
}

n_cases = max(len(CASE_CYCLONES), len(CASE_ANTICYCLONES))
fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases), sharex=True)

for col_idx, pol in enumerate(("cyclone", "anticyclone")):
    ids = case_ids[pol]
    for row_idx in range(n_cases):
        ax = axes[row_idx, col_idx]
        if row_idx >= len(ids):
            ax.set_visible(False)
            continue

        tid = ids[row_idx]

        # Pigment T chla
        sub_pig = pig_medians[(pig_medians["track_id"] == tid) & (pig_medians["polarity"] == pol)]
        sub_pig = sub_pig.sort_values("date")
        ax.plot(sub_pig["date"], sub_pig["T chla"], "o-", color="black", ms=3, label="T chla")
        ax.set_ylabel("T chla")

        # PFT fractions on secondary y-axis
        sub_pft = pft_medians[(pft_medians["track_id"] == tid) & (pft_medians["polarity"] == pol)]
        sub_pft = sub_pft.sort_values("date")
        ax2 = ax.twinx()
        top3 = sub_pft[PFT_COLS].mean().nlargest(3).index.tolist()
        for pft in top3:
            ax2.plot(sub_pft["date"], sub_pft[pft], "--", ms=2, lw=1, label=pft)
        ax2.set_ylabel("PFT fraction")

        ax.set_title(f"{pol} #{tid}", fontsize=10)
        if row_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

fig.autofmt_xdate()
fig.suptitle("Case study eddies — T chla + top PFTs over calendar time", y=1.01)
fig.tight_layout()
```

- [ ] **Step 5: Run all Layer 3 cells and verify**

Run cells 15-19. Verify:
- Spaghetti plot has individual eddy lines with the black dashed pooled reference overlaid
- Reference line follows a plausible seasonal arc
- Case study candidates table prints without error and shows eddies spanning 3+ months
- Case study small-multiples render with T chla on the left axis, PFT fractions on the right
- Twin axes don't produce overlapping labels

---

### Task 5: Final Polish

**Files:**
- Modify: `notebooks/seasonality.ipynb`

- [ ] **Step 1: Review all figures for readability**

Run the full notebook top to bottom. Check:
- No figure has overlapping text or truncated labels
- Colors are consistent (tab:blue for cyclone, tab:red for anticyclone throughout)
- Axis labels are short and plain (no verbose descriptions)
- No cells produce warnings or deprecation notices

- [ ] **Step 2: Fix any issues found in Step 1**

Apply fixes. Common issues to watch for:
- `fig.autofmt_xdate()` can rotate labels too aggressively — adjust if needed
- Stacked bar legend might need repositioning if it overlaps bars
- Box plot outlier fliers can make y-axis range too wide — consider adding `showfliers=False` if extreme outliers compress the main distribution

- [ ] **Step 3: Run final notebook execution**

Run all cells sequentially one final time to confirm clean execution with no errors.
