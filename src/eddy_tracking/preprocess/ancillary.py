from pathlib import Path

import pandas as pd

from eddy_tracking.preprocess.sss import read_multiple_sss
from eddy_tracking.preprocess.sst import read_multiple_sst


def read_ancillary_grids(sst_dir: Path, sss_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_multiple_sst(sorted(sst_dir.glob("*.nc"))),
        read_multiple_sss(sorted(sss_dir.glob("*.nc4"))),
    )
