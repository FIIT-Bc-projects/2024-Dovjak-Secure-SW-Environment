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
