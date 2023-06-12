import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

def read_pandas(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_ts_arr(df_data: pd.DataFrame) -> npt.NDArray[Any]:
    return df_data["ts_arr"].to_numpy()

df_data = read_pandas("data/df_elaad_preproc.csv")
ts_arr = get_ts_arr(df_data)
