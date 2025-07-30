import os
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from a .env file (optional)
load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("CMC_API_KEY")
if not API_KEY:
    raise RuntimeError("Set your CoinMarketCap API key in the CMC_API_KEY environment variable")

DB_NAME = os.getenv("DB_NAME", "your_db")
DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

BASE_URL = "https://pro-api.coinmarketcap.com/v1"
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY
}

SYMBOLS = ["BTC", "ETH", "ADA", "DOT", "SOL"]
CONVERT = "USD"


def fetch(endpoint, params=None):
    """Helper to perform GET on CoinMarketCap API and return .json()['data']."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()["data"]


def ensure_tables(cur):
    """Create tables if they don't exist."""
    print("Ensuring tables exist...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS crypto_quotes (
      symbol TEXT PRIMARY KEY,
      price_usd DOUBLE PRECISION,
      market_cap DOUBLE PRECISION,
      volume_24h DOUBLE PRECISION,
      pct_change_1h DOUBLE PRECISION,
      pct_change_24h DOUBLE PRECISION,
      pct_change_7d DOUBLE PRECISION,
      last_updated TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS crypto_listings (
      symbol TEXT PRIMARY KEY,
      rank INTEGER,
      circulating_supply DOUBLE PRECISION,
      total_supply DOUBLE PRECISION,
      max_supply DOUBLE PRECISION,
      num_market_pairs INTEGER,
      volume_24h DOUBLE PRECISION,
      last_updated TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS global_metrics (
      metric TEXT PRIMARY KEY,
      value DOUBLE PRECISION,
      last_updated TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS price_conversions (
      base_symbol TEXT,
      target_symbol TEXT,
      rate DOUBLE PRECISION,
      last_updated TIMESTAMP,
      PRIMARY KEY (base_symbol, target_symbol)
    );
    CREATE TABLE IF NOT EXISTS id_map (
      symbol TEXT PRIMARY KEY,
      cmc_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS metadata_info (
      symbol TEXT PRIMARY KEY,
      name TEXT,
      logo_url TEXT,
      description TEXT,
      tags TEXT[],
      date_added DATE
    );
    """)


def upsert_data(conn):
    """Fetch from CMC and upsert into Postgres."""
    cur = conn.cursor()
    now = datetime.utcnow()

    # 1. Quotes
    quotes = fetch("cryptocurrency/quotes/latest", {"symbol": ",".join(SYMBOLS), "convert": CONVERT})
    quote_rows = [
        (
            sym,
            data["quote"][CONVERT]["price"],
            data["quote"][CONVERT]["market_cap"],
            data["quote"][CONVERT]["volume_24h"],
            data["quote"][CONVERT]["percent_change_1h"],
            data["quote"][CONVERT]["percent_change_24h"],
            data["quote"][CONVERT]["percent_change_7d"],
            now
        )
        for sym, data in quotes.items()
    ]
    execute_values(cur, """
        INSERT INTO crypto_quotes
        (symbol, price_usd, market_cap, volume_24h, pct_change_1h, pct_change_24h, pct_change_7d, last_updated)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
          price_usd = EXCLUDED.price_usd,
          market_cap = EXCLUDED.market_cap,
          volume_24h = EXCLUDED.volume_24h,
          pct_change_1h = EXCLUDED.pct_change_1h,
          pct_change_24h = EXCLUDED.pct_change_24h,
          pct_change_7d = EXCLUDED.pct_change_7d,
          last_updated = EXCLUDED.last_updated;
    """, quote_rows)

    # 2. Listings
    listings = fetch("cryptocurrency/listings/latest", {"start": 1, "limit": 5, "convert": CONVERT})
    listing_rows = [
        (
            c["symbol"],
            c["cmc_rank"],
            c["circulating_supply"],
            c["total_supply"],
            c.get("max_supply"),
            c["num_market_pairs"],
            c["quote"][CONVERT]["volume_24h"],
            now
        )
        for c in listings
    ]
    execute_values(cur, """
        INSERT INTO crypto_listings
        (symbol, rank, circulating_supply, total_supply, max_supply, num_market_pairs, volume_24h, last_updated)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
          rank = EXCLUDED.rank,
          circulating_supply = EXCLUDED.circulating_supply,
          total_supply = EXCLUDED.total_supply,
          max_supply = EXCLUDED.max_supply,
          num_market_pairs = EXCLUDED.num_market_pairs,
          volume_24h = EXCLUDED.volume_24h,
          last_updated = EXCLUDED.last_updated;
    """, listing_rows)

    # 3. Global Metrics
    globals_ = fetch("global-metrics/quotes/latest")
    global_rows = [(key, val, now) for key, val in globals_["quote"][CONVERT].items() if key != "last_updated"]
    execute_values(cur, """
        INSERT INTO global_metrics (metric, value, last_updated)
        VALUES %s
        ON CONFLICT (metric) DO UPDATE SET value = EXCLUDED.value, last_updated = EXCLUDED.last_updated;
    """, global_rows)

    # 4. Price Conversion (1 BTC to USD)
    conv = fetch("tools/price-conversion", {"amount": 1, "symbol": "BTC", "convert": CONVERT})
    # Extract the actual price value from the quote dictionary
    conv_row = [("BTC", CONVERT, conv["quote"][CONVERT]["price"], now)]
    execute_values(cur, """
        INSERT INTO price_conversions (base_symbol, target_symbol, rate, last_updated)
        VALUES %s
        ON CONFLICT (base_symbol, target_symbol) DO UPDATE SET
          rate = EXCLUDED.rate, last_updated = EXCLUDED.last_updated;
    """, conv_row)

    # 5. ID Map
    id_map = fetch("cryptocurrency/map", {"symbol": ",".join(SYMBOLS)})
    map_rows = [(c["symbol"], c["id"]) for c in id_map]
    execute_values(cur, """
        INSERT INTO id_map (symbol, cmc_id) VALUES %s
        ON CONFLICT (symbol) DO NOTHING;
    """, map_rows)

    # 6. Metadata Info
    info = fetch("cryptocurrency/info", {"symbol": ",".join(SYMBOLS)})
    meta_rows = [
        (
            sym,
            info[sym]["name"],
            info[sym]["logo"],
            info[sym].get("description", ""),
            info[sym].get("tags", []),
            info[sym].get("date_added", None)
        )
        for sym in SYMBOLS
    ]
    execute_values(cur, """
        INSERT INTO metadata_info
        (symbol, name, logo_url, description, tags, date_added)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
          name = EXCLUDED.name,
          logo_url = EXCLUDED.logo_url,
          description = EXCLUDED.description,
          tags = EXCLUDED.tags,
          date_added = EXCLUDED.date_added;
    """, meta_rows)

    conn.commit()
    cur.close()


def main():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    ensure_tables(cur)
    conn.commit()

    upsert_data(conn)
    conn.close()


if __name__ == "__main__":
    main()