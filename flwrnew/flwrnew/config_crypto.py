import base64
import os

# Fetch encryption key from the environment
encryption_key_encoded = os.getenv("ENCRYPTION_KEY")
if not encryption_key_encoded:
    raise ValueError("ENCRYPTION_KEY environment variable is not set.")

# Decode the encryption key
encryption_key = base64.b64decode(encryption_key_encoded)

# Print for debugging (optional, remove in production)
print(f"Encryption key loaded successfully: {encryption_key}")
