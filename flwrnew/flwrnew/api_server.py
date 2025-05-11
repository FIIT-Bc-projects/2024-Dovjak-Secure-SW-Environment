import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
import datetime
from pymongo import MongoClient
from Crypto.PublicKey import RSA

# Set up MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # MongoDB URI (default: localhost:27017)
db = client.aggregation_db  # Select or create a database
metadata_collection = db.metadata  # Select or create a collection

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


# Decode weights function
def decode_weights(encrypted_data):
    """Decrypt received data and deserialize weights."""
    decrypted_data = encrypted_data  # HELP for debug

    try:
        weights_data = json.loads(decrypted_data.decode("utf-8"))

        return weights_data
    except Exception as e:
        print(f"Error decoding weights: {e}")
        return []


class AggregationHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/aggregation':
            # Handle aggregation-related POST request
            self.handle_aggregation_request()
        elif self.path == '/data-usage':
            # Handle data usage verification request
            self.handle_data_usage_request()
        else:
            self.send_error(404, "Not Found")

    def handle_aggregation_request(self):
        try:
            print("Received POST request")

            # Extract metadata from the custom header
            metadata_header = self.headers.get("Client-Metadata", "[]")
            metadata = json.loads(metadata_header)  # Parse the JSON string into a list of dictionaries

            # Process each client_id and epoch pair
            full_examples = []
            all_clients = []
            all_epochs = []
            all_hashs = []
            for entry in metadata:
                client_id = entry.get("client_id", "unknown_client")
                epoch = entry.get("epoch", 0)
                num_examples = entry.get("num_examples", 0)
                weight_hash = entry.get("weights_hash", 0)
                full_examples.append(num_examples)
                all_epochs.append(epoch)
                all_clients.append(client_id)
                all_hashs.append(weight_hash)

                # Insert metadata into MongoDB
                metadata_document = {
                    "client_id": client_id,
                    "epoch": epoch,
                    "num_examples": num_examples,
                    "weights_hash": weight_hash,
                    "timestamp": datetime.datetime.now()  # Add timestamp for reference
                }

                # Insert the document into the 'metadata' collection
                metadata_collection.insert_one(metadata_document)

                print(f"Inserted metadata for client {client_id} into MongoDB.")

            # Read the binary weights from the body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            client_weights = decode_weights(post_data)

            # Aggregate and prepare response
            aggregated_weights = self.aggregate_weights(client_weights, full_examples)

            serialized_response = json.dumps([w.tolist() for w in aggregated_weights]).encode('utf-8')
            encrypted_response = serialized_response  # help for decoding

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

    def handle_data_usage_request(self):
        """Handle POST request for data usage verification."""
        try:
            print("Received POST request for data usage verification")

            # Parse the incoming JSON payload (client_ids)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))
            client_ids = request_data.get("client_ids", [])

            # Query MongoDB for the metadata of the specified client IDs
            metadata = self.query_metadata(client_ids)

            # Send the queried metadata as a response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metadata).encode('utf-8'))  # Send JSON response

        except Exception as e:
            print(f"[Server Error] Error handling data usage POST request: {e}")
            self.send_error(500, "Server error.")

    def query_metadata(self, client_ids):
        """Query MongoDB for metadata related to the provided client IDs."""
        metadata = []
        for client_id in client_ids:
            client_data = metadata_collection.find({"client_id": client_id})
            for entry in client_data:
                metadata.append({
                    "client_id": entry["client_id"],
                    "epoch": entry["epoch"],
                    "weights_hash": entry["weights_hash"],  # Return the weights data like hash used in that epoch
                })
        return metadata

    def aggregate_weights(self, client_weights, num_examples):
        
        # Aggregate weights layer-by-layer.
        # client_weights is a list of model weights from different clients,
        # where each element in the list is a set of layer weights for a particular client.
        
        try:
            # Step 2: Aggregate the client weights using a weighted average
            total_examples = sum(num_examples)
            num_layers = len(client_weights[0])  # Assuming each client has the same model structure
            # Initialize a list to hold the aggregated weights
            aggregated_weights = [np.zeros_like(client_weights[0][layer_idx]) for layer_idx in range(num_layers)]

            # Perform layer-wise aggregation
            for layer_idx in range(num_layers):
                weighted_sum = np.zeros_like(
                    client_weights[0][layer_idx])  # Initialize weighted sum with the first client's layer shape
                # Sum the weighted client weights for each layer
                count = 0
                for weights, num_example in zip(client_weights, num_examples):
                    if isinstance(weights[layer_idx], np.ndarray):
                        weighted_sum += weights[
                                        layer_idx] * num_example  # Multiply weights by the client's number of examples
                    else:
                        # If weights are not NumPy arrays, convert them
                        weighted_sum += np.array(weights[layer_idx]) * num_example
                    count += 1
                # Normalize by the total number of examples (weighted average)
                aggregated_weights[layer_idx] = weighted_sum / total_examples
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
