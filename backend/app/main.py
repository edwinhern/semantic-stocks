import os
from datetime import datetime, timedelta

from fastapi import FastAPI
from massive import RESTClient
from massive.rest.aggs import Agg

app = FastAPI()

client = RESTClient(
    api_key=os.getenv("MASSIVE_API_KEY"),
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World!"}


@app.get("/api/quote/{ticker}")
async def get_quote(ticker: str) -> list[Agg]:
    quote = client.list_aggs(
        ticker,
        multiplier=1,
        timespan="minute",
        from_=datetime.now() - timedelta(days=1),
        to=datetime.now(),
        limit=5000,
    )
    return quote
