from pathlib import Path

from pandas import DataFrame

def edr_to_df(path: str | Path, verbose: bool = False) -> DataFrame: ...
