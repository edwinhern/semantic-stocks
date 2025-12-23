"""S&P 500 companies data fetcher.

Fetches the current S&P 500 constituents directly from Wikipedia.
Uses Redis for caching in production (Docker), with local file fallback for development.

Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
"""

from collections import Counter
from dataclasses import asdict, dataclass
from io import StringIO
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import pandas as pd

    from ..services.cache.redis_client import RedisCache
else:
    import pandas as pd  # type: ignore[import-untyped]

# Wikipedia URL for S&P 500 companies table
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# User-Agent header to avoid Wikipedia blocking
HEADERS = {"User-Agent": "SemanticStocks/1.0 (https://github.com/semantic-stocks; contact@example.com)"}


@dataclass
class SP500Company:
    """Represents an S&P 500 company."""

    ticker: str
    company_name: str
    sector: str
    sub_industry: str
    headquarters: str
    date_added: str | None = None
    cik: str | None = None
    founded: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for Redis storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SP500Company":
        """Create from dictionary (from Redis)."""
        return cls(
            ticker=data.get("ticker", ""),
            company_name=data.get("company_name", ""),
            sector=data.get("sector", ""),
            sub_industry=data.get("sub_industry", ""),
            headquarters=data.get("headquarters", ""),
            date_added=data.get("date_added"),
            cik=data.get("cik"),
            founded=data.get("founded"),
        )


def _get_redis_cache() -> "RedisCache | None":
    """Get Redis cache instance, or None if unavailable.

    Returns None silently if Redis is not configured or unreachable.
    This allows the application to work without Redis in development.
    """
    try:
        from ..services.cache.redis_client import get_redis_cache

        cache = get_redis_cache()
        if cache.ping():
            return cache
        return None
    except Exception:
        # Redis not available - this is expected in development
        return None


def _fetch_from_wikipedia() -> "pd.DataFrame":  # type: ignore[no-any-unimported]
    """Fetch the S&P 500 table directly from Wikipedia.

    Uses httpx to fetch the HTML (with proper User-Agent), then
    pandas.read_html() to parse the table.
    The first table on the page contains the current S&P 500 constituents.

    Returns:
        DataFrame with S&P 500 companies
    """
    print("Fetching S&P 500 data from Wikipedia...")

    # Fetch HTML with proper User-Agent to avoid 403 Forbidden
    response = httpx.get(WIKIPEDIA_SP500_URL, headers=HEADERS, timeout=30.0)
    response.raise_for_status()

    # Parse HTML tables
    tables = pd.read_html(StringIO(response.text), header=0)
    # The first table is the S&P 500 constituents
    return tables[0]


def _dataframe_to_companies(df: "pd.DataFrame") -> list[SP500Company]:  # type: ignore[no-any-unimported]
    """Convert DataFrame to list of SP500Company objects."""
    companies: list[SP500Company] = []

    # Use itertuples() instead of iterrows() for better performance
    for row in df.itertuples(index=False):
        ticker = str(getattr(row, "Symbol", "")).strip()
        if not ticker or ticker == "nan":  # Skip empty rows
            continue

        company = SP500Company(
            ticker=ticker,
            company_name=str(getattr(row, "Security", "")).strip(),
            sector=str(getattr(row, "GICS Sector", "")).strip(),
            sub_industry=str(getattr(row, "GICS Sub-Industry", "")).strip(),
            headquarters=str(getattr(row, "Headquarters Location", "")).strip(),
            date_added=_clean_field(getattr(row, "Date added", None)),
            cik=_clean_field(getattr(row, "CIK", None)),
            founded=_clean_field(getattr(row, "Founded", None)),
        )
        companies.append(company)

    return companies


def _clean_field(value: object) -> str | None:
    """Clean a field value, converting to string or None."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned and cleaned.lower() != "nan" else None


def refresh_sp500_cache() -> list[SP500Company]:
    """Force refresh the S&P 500 cache from Wikipedia.

    Fetches fresh data and stores it in Redis (24-hour TTL).

    Returns:
        List of SP500Company objects
    """
    df = _fetch_from_wikipedia()
    companies = _dataframe_to_companies(df)

    # Cache in Redis
    cache = _get_redis_cache()
    if cache:
        cache.set_sp500_list([c.to_dict() for c in companies])
        print(f"Cached {len(companies)} S&P 500 companies to Redis (24h TTL)")
    else:
        print("Redis unavailable - S&P 500 data not cached")

    return companies


def get_sp500_companies(force_refresh: bool = False) -> list[SP500Company]:
    """Get the list of S&P 500 companies.

    Uses Redis cache to avoid fetching from Wikipedia on every call.
    Cache TTL is 24 hours (set in RedisCache).

    Args:
        force_refresh: If True, always fetch fresh data from Wikipedia

    Returns:
        List of SP500Company objects with ticker, name, sector, etc.
    """
    if force_refresh:
        return refresh_sp500_cache()

    # Try to get from Redis cache
    cache = _get_redis_cache()
    if cache:
        cached_data = cache.get_sp500_list()
        if cached_data:
            print("Loading S&P 500 data from Redis cache...")
            return [SP500Company.from_dict(c) for c in cached_data]

    # Cache miss or Redis unavailable - fetch from Wikipedia
    return refresh_sp500_cache()


def get_sp500_tickers() -> list[tuple[str, str, str]]:
    """Get S&P 500 tickers as simple tuples.

    Convenience function for pipeline input.

    Returns:
        List of (ticker, company_name, sector) tuples
    """
    companies = get_sp500_companies()
    return [(c.ticker, c.company_name, c.sector) for c in companies]


# Quick test
if __name__ == "__main__":
    # Load environment variables from .env file
    from dotenv import load_dotenv

    load_dotenv()

    print("Testing S&P 500 data fetcher...")

    # Check Redis connectivity
    cache = _get_redis_cache()
    if cache:
        print("✓ Redis connected")
    else:
        print("✗ Redis not available - will fetch fresh data")

    # Force refresh to get latest data from Wikipedia
    companies = get_sp500_companies(force_refresh=True)
    print(f"\nLoaded {len(companies)} S&P 500 companies")

    print("\nFirst 10 companies:")
    for c in companies[:10]:
        print(f"  {c.ticker:6} | {c.company_name:30} | {c.sector}")

    print("\nSector breakdown:")
    sector_counts = Counter(c.sector for c in companies)
    for sector, count in sector_counts.most_common():
        print(f"  {sector:40} {count:3} companies")

    # Verify cache is working
    if cache:
        print("\nVerifying Redis cache...")
        cached = get_sp500_companies()
        print(f"✓ Retrieved {len(cached)} companies from cache")
