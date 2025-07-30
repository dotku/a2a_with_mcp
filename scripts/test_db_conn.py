import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get database credentials from environment
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

print("=== Database Connection Test ===")
print(f"Host: {DB_HOST}")
print(f"Port: {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}")
print(f"Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'Not set'}")
print("================================")

try:
    # Attempt connection with SSL mode for Supabase
    print("\nAttempting to connect...")
    if "supabase.com" in DB_HOST or "localhost" not in DB_HOST:
        print("suapbase")
        # For Supabase or external hosts, use SSL
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require",
        )
        print("✅ Connected successfully with SSL")
    else:
        print("localhost")
        # For localhost, don't specify sslmode
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        print("✅ Connected successfully (localhost)")

    print("✅ Connection successful!")

    # Create a cursor
    cur = conn.cursor()

    # Test query - get PostgreSQL version
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print(f"\nPostgreSQL version: {version[0]}")

    # List all tables
    cur.execute(
        """
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """
    )

    tables = cur.fetchall()
    if tables:
        print("\nTables in database:")
        for schema, table in tables:
            print(f"  - {schema}.{table}")
    else:
        print("\nNo user tables found in database.")

    # Test creating a simple table (optional)
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS test_connection (
                id SERIAL PRIMARY KEY,
                test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message TEXT
            );
        """
        )
        conn.commit()
        print("\n✅ Successfully created test table")

        # Insert a test record
        cur.execute(
            """
            INSERT INTO test_connection (message) 
            VALUES ('Connection test successful');
        """
        )
        conn.commit()
        print("✅ Successfully inserted test record")

        # Clean up test table
        cur.execute("DROP TABLE test_connection;")
        conn.commit()
        print("✅ Cleaned up test table")

    except psycopg2.Error as e:
        print(f"\n⚠️  Could not create test table (may need permissions): {e}")
        conn.rollback()

    # Close cursor and connection
    cur.close()
    conn.close()
    print("\n✅ Connection closed successfully")

except psycopg2.OperationalError as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nPossible fixes:")
    print("1. Check your database credentials in .env file")
    print("2. Ensure the database server is running")
    print("3. Verify network connectivity to the host")
    print("4. For Supabase, ensure you're using the correct connection string")

except psycopg2.Error as e:
    print(f"\n❌ Database error: {e}")

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
