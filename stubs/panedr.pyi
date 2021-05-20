from pathlib import Path
from typing import Union

from pandas import DataFrame

def edr_to_df(path: Union[str, Path], verbose: bool = False) -> DataFrame: ...
