import os  # Import os for environment variables
from dotenv import load_dotenv

load_dotenv()

# Debug: Print all environment variables
print("=== Environment Variables ===")
for key, value in os.environ.items():
    # Hide sensitive values
    if any(
        sensitive in key.upper() for sensitive in ["PASSWORD", "KEY", "SECRET", "TOKEN"]
    ):
        print(f"{key}: {'*' * 8}")
    else:
        print(f"{key}: {value}")
print("===========================")
