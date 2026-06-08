import databento as db

from db_key import get_databento_key

client = db.Historical(key=get_databento_key())

params = dict(
    dataset="XNAS.ITCH",
    symbols=["AMZN"],
    schema="mbo",
    stype_in="raw_symbol",
    start="2022-01-03",
    end="2022-12-30",
)

size_bytes = client.metadata.get_billable_size(**params)
print(f"Billable size: {size_bytes / 1e9:.2f} GB")
print(f"Compressed .dbn.zst will be roughly: {size_bytes / 1e9 * 0.15:.2f} GB (est. 15% of billable)")
