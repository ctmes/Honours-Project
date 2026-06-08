import databento as db
from pathlib import Path

from db_key import get_databento_key

client = db.Historical(key=get_databento_key())

output_dir = Path("data/databento/AMZN/2022")
output_dir.mkdir(parents=True, exist_ok=True)

data = client.timeseries.get_range(
    dataset="XNAS.ITCH",
    symbols=["AMZN"],
    schema="mbo",
    stype_in="raw_symbol",
    start="2022-01-03",
    end="2022-12-30",
    path=output_dir / "AMZN_2022_mbo.dbn.zst",  # compressed DBN format
)
