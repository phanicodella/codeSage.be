from cryptography.fernet import Fernet
import os

# File paths
DECODED_FILE = '.env'
ENCODED_FILE = '.enc'
KEY_FILE = 'secret.key'

# Generate a new encryption key
def generate_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        print(f"New encryption key generated and saved to {KEY_FILE}")
    else:
        print(f"Key file {KEY_FILE} already exists.")

# Load the encryption key
def load_key():
    if not os.path.exists(KEY_FILE):
        print(f"Key file {KEY_FILE} not found. Generating a new key...")
        generate_key()
    with open(KEY_FILE, 'rb') as key_file:
        return key_file.read()

# Encode the .env file
def encode_env():
    key = load_key()
    fernet = Fernet(key)

    if not os.path.exists(DECODED_FILE):
        raise FileNotFoundError(f"Decoded file {DECODED_FILE} not found.")

    with open(DECODED_FILE, 'rb') as file:
        data = file.read()

    encoded_data = fernet.encrypt(data)

    with open(ENCODED_FILE, 'wb') as file:
        file.write(encoded_data)

    print(f".env file encoded and saved as {ENCODED_FILE}")

# Decode the encoded file back to .env
def decode_env():
    key = load_key()
    fernet = Fernet(key)

    if not os.path.exists(ENCODED_FILE):
        raise FileNotFoundError(f"Encoded file {ENCODED_FILE} not found.")

    with open(ENCODED_FILE, 'rb') as file:
        encoded_data = file.read()

    decoded_data = fernet.decrypt(encoded_data)

    with open(DECODED_FILE, 'wb') as file:
        file.write(decoded_data)

    print(f"Encoded file decoded and saved as {DECODED_FILE}")

if __name__ == "__main__":
    if not os.path.exists(KEY_FILE):
        print("No encryption key found. Generating one...")
        generate_key()

    if os.path.exists(DECODED_FILE):
        encode_env()
    elif os.path.exists(ENCODED_FILE):
        decode_env()
    else:
        print(f"Neither {DECODED_FILE} nor {ENCODED_FILE} found. Please add one to proceed.")
