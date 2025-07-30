# scraper_agent

## TODO

1. need fix the unique violation issue

```bash
psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "crypto_quotes_pkey"
DETAIL:  Key (symbol)=(ADA) already exists.
```

2. deprecate the datetime.utcnow

```bash
/Users/wlin/dev/a2a_with_mcp/agents/scraper_agent/prices_scraping.py:94: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  now = datetime.utcnow()
```
