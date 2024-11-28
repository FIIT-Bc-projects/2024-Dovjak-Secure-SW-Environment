# from Crypto.Cipher import ChaCha20_Poly1305
# from Crypto.Random import get_random_bytes
# import base64
#
#
# # Generate a symmetric encryption key
# def generate_key():
#     return get_random_bytes(32)  # ChaCha20 uses a 256-bit key
#
#
# # urobit public key na server
# # Encrypt data
# def encrypt_data(data, key):
#     cipher = ChaCha20_Poly1305.new(key=key)
#     nonce = cipher.nonce
#     ciphertext, tag = cipher.encrypt_and_digest(data)
#     print(f"[Encrypt] Nonce: {nonce}, Tag: {tag}")
#     return base64.b64encode(nonce + tag + ciphertext)
#
#
# # Decrypt data
# def decrypt_data(encoded_data, key):
#     raw_data = base64.b64decode(encoded_data)
#     nonce, tag, ciphertext = raw_data[:12], raw_data[12:28], raw_data[28:]
#     print(f"[Decrypt] Nonce: {nonce}, Tag: {tag}")
#     print(f"Key length: {len(key)} bytes (should be 32 bytes) key: {key}")
#     print(f"Nonce length: {len(nonce)} bytes (should be 12 bytes)")
#
#     # Ensure they are the correct length
#     if len(key) != 32:
#         raise ValueError("Encryption key must be 32 bytes long.")
#     if len(nonce) != 12:
#         raise ValueError("Nonce must be 12 bytes long.")
#
#     # Proceed to create the cipher
#     cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
#     cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
#     print("ZA cyphher pri crypto_utils")
#     return cipher.decrypt_and_verify(ciphertext, tag)

from Crypto.PublicKey import RSA
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
key_path_priv = os.path.join(script_dir, "crypt_keys/private.pem")
key_path_pub = os.path.join(script_dir, "crypt_keys/public.pem")
# Load the private key for the server
with open(key_path_priv, "rb") as priv_file:
    private_key = RSA.import_key(priv_file.read())

# Load the public key (optional, for testing or sending to clients)
with open(key_path_pub, "rb") as pub_file:
    public_key = RSA.import_key(pub_file.read())
