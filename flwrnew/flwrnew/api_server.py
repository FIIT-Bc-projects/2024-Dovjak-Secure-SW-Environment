import base64
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
from config_crypto import encryption_key
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA


# Load RSA keys
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the private key for decryption
private_key_path = os.path.join(script_dir, "crypt_keys/private.pem")
with open(private_key_path, "rb") as priv_file:
    private_key = RSA.import_key(priv_file.read())

# Load the public key for encryption (optional, to encrypt responses)
public_key_path = os.path.join(script_dir, "crypt_keys/public.pem")
with open(public_key_path, "rb") as pub_file:
    public_key = RSA.import_key(pub_file.read())


# RSA encryption function
def encrypt_data_with_rsa(data):
    """Encrypt data using RSA hybrid encryption."""
    # Generate a random symmetric key for AES encryption
    symmetric_key = os.urandom(16)

    # Encrypt the data using AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX)
    ciphertext, tag = aes_cipher.encrypt_and_digest(data)
    encrypted_data = aes_cipher.nonce + ciphertext + tag

    # Encrypt the symmetric key using RSA
    rsa_cipher = PKCS1_OAEP.new(public_key)
    encrypted_key = rsa_cipher.encrypt(symmetric_key)

    # Combine the encrypted symmetric key and encrypted data
    return encrypted_key + encrypted_data


# RSA decryption function
def decrypt_data_with_rsa(encrypted_message):
    """Decrypt data using RSA hybrid encryption."""
    encrypted_key = encrypted_message[:256]  # First 256 bytes
    encrypted_data = encrypted_message[256:]  # Remaining bytes

    # Decrypt the symmetric key using RSA
    rsa_cipher = PKCS1_OAEP.new(private_key)
    symmetric_key = rsa_cipher.decrypt(encrypted_key)

    # Decrypt the weights with AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX, nonce=encrypted_data[:16])
    plaintext = aes_cipher.decrypt_and_verify(encrypted_data[16:-16], encrypted_data[-16:])
    return plaintext


# Decode weights function
def decode_weights(encrypted_data):
    """Decrypt received data and deserialize weights."""
    # print(f"Server Received Encrypted Data Length: {len(encrypted_data)}")
    decrypted_data = decrypt_data_with_rsa(encrypted_data)
    # print("Server sided decryption of data length is: ", len(decrypted_data))
    weights = [np.array(w) for w in json.loads(decrypted_data.decode('utf-8'))]
    return weights


class AggregationHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            print("Received POST request")
            content_length = int(self.headers['Content-Length'])
            print(f"Content-Length: {content_length}")
            encrypted_data = self.rfile.read(content_length)
            # print("Data received:", encrypted_data[:100])  # Print first 100 bytes
            # print("Length of data is: ", len(encrypted_data))
            client_weights = decode_weights(encrypted_data)
            # print(f"Received weights with shapes: {[w.shape for w in client_weights]}")

            # Aggregate and prepare response
            aggregated_weights = self.aggregate_weights(client_weights)
            serialized_response = json.dumps([w.tolist() for w in aggregated_weights]).encode('utf-8')
            # print("Length of serialized response is: ", len(serialized_response))
            encrypted_response = encrypt_data_with_rsa(serialized_response)
            # print("Length of encrypted serialized response is: ", len(encrypted_response))
            # print(f"Response being sent (first 100 bytes): {encrypted_response[:100]}")
            # print(f"Total length of encrypted response: {len(encrypted_response)}")

            # Send encrypted response to client as bytes
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.end_headers()
            self.wfile.write(encrypted_response)  # Already bytes, so no further encoding needed

        except ValueError as e:
            # Specific catch for MAC check failures
            print(f"[Server Error] MAC check failed or decryption error: {e}")

            self.send_error(400, "Decryption error or MAC check failed.")

        except Exception as e:
            print(f"[Server Error] Error handling POST request: {e}")

            self.send_error(500, "Server error.")

    def check_consistent_shapes(self, client_weights):
        # Ensure that the shapes of client weights are consistent.
        # Group client weights by layer index and ensure consistency
        num_layers = len(client_weights)
        for layer_idx in range(num_layers):
            # Extract the shapes of the current layer's weights from all clients
            layer_shape = client_weights[layer_idx].shape
            # print(f"Checking consistency for layer {layer_idx}, shape: {layer_shape}")
            if layer_shape != client_weights[layer_idx].shape:
                raise ValueError(f"Inconsistent shapes detected for layer {layer_idx}: {layer_shape}")

    def aggregate_weights(self, client_weights):
        
        # Aggregate weights layer-by-layer.
        # Assumes client_weights is a list of model weights from different clients,
        # where each element in the list is a set of layer weights for a particular client.
        
        try:
            # Number of layers (assumes each client has the same number of layers)
            num_layers = len(client_weights)

            aggregated_weights = []
            print("Before aggregation in api server")
            # Aggregate weights for each layer
            for layer_idx in range(num_layers):
                # Extract the weights for this layer from all clients
                layer_weights = np.array([client_weights[layer_idx]])

                # Log the shapes of the layer weights being aggregated
                #print(f"Aggregating layer {layer_idx}, weight shapes: {[w.shape for w in layer_weights]}")

                # Perform weighted average (or simple average in this case)
                layer_aggregated = np.mean(layer_weights, axis=0)

                # Append aggregated layer weights
                aggregated_weights.append(layer_aggregated)
            print("After aggregation in api server")
            return aggregated_weights

        except Exception as e:
            # Log the error
            print(f"Error handling POST request: {e}")

            # Always send a valid JSON response, even on error
            error_response = {
                "error": "An error occurred on the server.",
                "details": str(e)
            }
            self.send_response(500)  # HTTP status for server error
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))  # Encode to bytes


# Set up and run the server
def run_server():
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, AggregationHandler)
    print("Server running on port 8080...")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
