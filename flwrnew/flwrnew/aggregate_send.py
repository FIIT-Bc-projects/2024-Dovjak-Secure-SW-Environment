import json
import base64
import requests
import os
#from .config_crypto import encryption_key
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA

from flwrnew.crypto_utils import public_key

# skusit cez postman


# Get the absolute path to the key file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(script_dir, "crypt_keys/public.pem")

# Load the private key
with open(key_path, "rb") as pub_file:
    private_key = RSA.import_key(pub_file.read())


# RSA encryption function
def encrypt_data_with_rsa(data):
    # Generate a random symmetric key for AES encryption
    symmetric_key = get_random_bytes(16)

    # Encrypt the data using AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX)
    ciphertext, tag = aes_cipher.encrypt_and_digest(data)
    encrypted_data = aes_cipher.nonce + ciphertext + tag

    # Encrypt the symmetric key using the public RSA key
    rsa_cipher = PKCS1_OAEP.new(public_key)
    encrypted_key = rsa_cipher.encrypt(symmetric_key)

    # Return the encrypted key and data
    return encrypted_key + encrypted_data


def encode_weights(weights):
    serialized_weights = json.dumps(
        [w.tolist() if hasattr(w, "tolist") else [w] for w in weights]
    ).encode('utf-8')

    # Encrypt the serialized weights with RSA
    encrypted_data = encrypt_data_with_rsa(serialized_weights)
    return encrypted_data


def send_weights_to_server(weights, server_url="http://localhost:8080"):
    try:
        # Prepare the weights
        print("Before encryption")
        #encrypted_weights = encode_weights(weights)
        encrypted_weights = weights
        print(f"Client Encrypted Data Length: {len(encrypted_weights)}")
        print("After encryption")
        # Send POST request with `Content-Type` for binary data
        response = requests.post(
            server_url,
            data=encrypted_weights,  # `encrypted_data` is already encoded as bytes
            headers={"Content-Type": "application/octet-stream"},
            timeout=30  # Timeout in seconds
        )

        print(f"Server responded with status code: {response.status_code}")

        # Check the response and decode if successful
        if response.status_code == 200:
            print(f"Raw response content (first 100 bytes): {response.content[:100]}")
            print(f"Response type: {type(response.content)}, Length: {len(response.content)}")
            return response.content  # Decode server's encrypted response
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error sending data to server: {e}")
        return None


