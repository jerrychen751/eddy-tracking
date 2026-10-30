# Position of an eddy-composite on the two-part time axis of an eddy that crosses the Gulf Stream: 0 to 1 is the time on the side it started on, 1 to 2 the time on the side it ends on
from typing import cast

import pandas as pd

from eddy_tracking.config import resolve_output_dir
from eddy_tracking.preprocess.streamline import compute_signed_distance_km, index_centerlines_by_date
from eddy_tracking.preprocess.tracks import load_track_observations

IDENTITY_COLUMNS = ['polarity', 'track_id']


def measure_distance_past_axis(experiment: str, tracks: pd.DataFrame) -> pd.DataFrame:
    # Daily observations of the given crossing tracks, with across_km the distance from the daily axis toward the side the eddy ends on: positive south of the axis for a cyclone, north of it for an anticyclone
    observations = load_track_observations(experiment).drop(columns=['contour_lon', 'contour_lat']).merge(tracks[IDENTITY_COLUMNS], on=IDENTITY_COLUMNS).sort_values(IDENTITY_COLUMNS + ['date'], ignore_index=True)
    centerlines = index_centerlines_by_date(pd.read_parquet(resolve_output_dir(experiment, 'gulf_stream') / 'streamline.parquet'))
    observations['across_km'] = cast(pd.Series, observations['polarity']).map({'cyclone': -1, 'anticyclone': 1}) * [
        compute_signed_distance_km(centerlines[day.date()].lon, centerlines[day.date()].lat, lon, lat)[0]
        for day, lon, lat in zip(observations['date'], observations['center_lon'], observations['center_lat'])
    ]
    return observations


def add_phase_position(table: pd.DataFrame, tracks: pd.DataFrame, observations: pd.DataFrame, near_axis_km: float) -> pd.DataFrame:
    # The departure is the first day an eddy is farther than near_axis_km past the axis; a crossing eddy that never gets that far has no second part and drops out here
    departure = cast(pd.Series, observations.loc[observations['across_km'].gt(near_axis_km)].groupby(IDENTITY_COLUMNS)['date'].min()).rename('departure_date').reset_index()
    table = table.merge(tracks[IDENTITY_COLUMNS + ['birth_date', 'death_date']], on=IDENTITY_COLUMNS, suffixes=('', '_track')).merge(departure, on=IDENTITY_COLUMNS)
    before = table['date'] < table['departure_date']
    start = table['birth_date'].where(before, table['departure_date'])
    end = table['departure_date'].where(before, table['death_date'])
    # From the first detection to the departure, then from the departure to the last detection; a composite midpoint can fall outside the track, hence the clip
    table['phase_position'] = ((table['date'] - start).dt.days / (end - start).dt.days.clip(lower=1)).clip(0, 1) + ~before
    return table
