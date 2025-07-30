#!/usr/bin/env python3
"""
Database setup script for Financial Agent.
Creates the necessary tables for financial data if they don't exist.
"""

import os
import asyncpg
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
PG_USER = os.environ.get("PG_USER", "financial_agent_user")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "strong_password_here")
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "financial_agent_db")

DB_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

CREATE_TABLES_SQL = """
-- Quotes table with timestamp column for historical data
CREATE TABLE IF NOT EXISTS crypto_quotes (
  id SERIAL,
  symbol TEXT NOT NULL,
  price_usd DOUBLE PRECISION,
  market_cap DOUBLE PRECISION,
  volume_24h DOUBLE PRECISION,
  pct_change_1h DOUBLE PRECISION,
  pct_change_24h DOUBLE PRECISION,
  pct_change_7d DOUBLE PRECISION,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- Listings table with timestamp column for historical data
CREATE TABLE IF NOT EXISTS crypto_listings (
  id SERIAL,
  symbol TEXT NOT NULL,
  cmc_rank INTEGER,
  circulating_supply DOUBLE PRECISION,
  total_supply DOUBLE PRECISION,
  max_supply DOUBLE PRECISION,
  num_market_pairs INTEGER,
  volume_24h DOUBLE PRECISION,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- Global metrics with timestamp column for historical data
CREATE TABLE IF NOT EXISTS global_metrics (
  id SERIAL,
  metric TEXT NOT NULL,
  value DOUBLE PRECISION,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- Price conversions with timestamp column for historical data
CREATE TABLE IF NOT EXISTS price_conversions (
  id SERIAL,
  base_symbol TEXT NOT NULL,
  target_symbol TEXT NOT NULL,
  rate DOUBLE PRECISION,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- ID mapping table
CREATE TABLE IF NOT EXISTS id_map (
  symbol TEXT PRIMARY KEY,
  cmc_id INTEGER
);

-- Metadata information table
CREATE TABLE IF NOT EXISTS metadata_info (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  logo_url TEXT,
  description TEXT,
  tags TEXT[],
  date_added DATE
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS crypto_quotes_symbol_timestamp_idx ON crypto_quotes(symbol, timestamp);
CREATE INDEX IF NOT EXISTS crypto_listings_symbol_timestamp_idx ON crypto_listings(symbol, timestamp);
CREATE INDEX IF NOT EXISTS global_metrics_metric_timestamp_idx ON global_metrics(metric, timestamp);
CREATE INDEX IF NOT EXISTS price_conversions_symbols_timestamp_idx ON price_conversions(base_symbol, target_symbol, timestamp);

-- Insert some sample data for testing
INSERT INTO crypto_quotes (symbol, price_usd, market_cap, volume_24h, pct_change_1h, pct_change_24h, pct_change_7d)
VALUES 
  ('BTC', 45000.00, 850000000000, 25000000000, 0.5, 2.1, -3.2),
  ('ETH', 3200.00, 380000000000, 15000000000, 0.8, 1.9, -1.5),
  ('SOL', 95.50, 42000000000, 2500000000, 1.2, 4.2, 8.7)
ON CONFLICT DO NOTHING;

INSERT INTO id_map (symbol, cmc_id) VALUES 
  ('BTC', 1),
  ('ETH', 1027),
  ('SOL', 5426)
ON CONFLICT (symbol) DO NOTHING;

INSERT INTO metadata_info (symbol, name, logo_url, description, tags, date_added) VALUES 
  ('BTC', 'Bitcoin', 'https://example.com/btc.png', 'The first cryptocurrency', ARRAY['cryptocurrency', 'digital-currency'], '2009-01-03'),
  ('ETH', 'Ethereum', 'https://example.com/eth.png', 'Smart contract platform', ARRAY['cryptocurrency', 'smart-contracts'], '2015-07-30'),
  ('SOL', 'Solana', 'https://example.com/sol.png', 'High performance blockchain', ARRAY['cryptocurrency', 'defi'], '2020-03-16')
ON CONFLICT (symbol) DO UPDATE SET
  name = EXCLUDED.name,
  logo_url = EXCLUDED.logo_url,
  description = EXCLUDED.description,
  tags = EXCLUDED.tags,
  date_added = EXCLUDED.date_added;
"""

async def setup_database():
    """Setup the database tables."""
    print(f"Setting up database at {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    print(f"User: {PG_USER}")
    
    try:
        # Connect to the database
        conn = await asyncpg.connect(DB_URL)
        print("✅ Connected to database successfully")
        
        # Execute the table creation script
        await conn.execute(CREATE_TABLES_SQL)
        print("✅ Database tables created successfully")
        
        # Test query to verify setup
        result = await conn.fetch("SELECT symbol, price_usd FROM crypto_quotes LIMIT 3;")
        if result:
            print("\n📊 Sample data:")
            for row in result:
                print(f"  {row['symbol']}: ${row['price_usd']:,.2f}")
        
        await conn.close()
        print("✅ Database setup completed")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your database credentials in environment variables")
        print("3. Ensure the database and user exist")
        print("4. Verify network connectivity to the database host")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())