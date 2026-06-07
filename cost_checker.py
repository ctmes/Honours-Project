import databento as db

client = db.Historical(key="db-nwxHtt6SrqCq8J96J5dhDiXPJ93Ng")

cost = client.metadata.get_cost(
    dataset="XNAS.ITCH",
    symbols=["AMZN"],
    schema="mbo",
    stype_in="raw_symbol",
    start="2022-01-03",
    end="2022-01-31",   # start with one month
)
print(cost)
