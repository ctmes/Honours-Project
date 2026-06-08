import databento as db

from db_key import get_databento_key

client = db.Historical(key=get_databento_key())

cost = client.metadata.get_cost(
    dataset="XNAS.ITCH",
    symbols=["AMZN"],
    schema="mbo",
    stype_in="raw_symbol",
    start="2022-01-03",
    end="2022-01-31",   # start with one month
)
print(cost)
