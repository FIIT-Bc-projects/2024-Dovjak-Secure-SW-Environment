from Crypto.PublicKey import RSA


def generate_rsa_key_pair():
    key = RSA.generate(2048)  # Generate a 2048-bit key pair
    private_key = key.export_key()
    public_key = key.publickey().export_key()

    # Save keys to files
    with open("crypt_keys/private.pem", "wb") as priv_file:
        priv_file.write(private_key)
    with open("crypt_keys/public.pem", "wb") as pub_file:
        pub_file.write(public_key)

    print("RSA key pair generated and saved.")


generate_rsa_key_pair()