import json
import requests
import os
import numpy as np
from Crypto.PublicKey import RSA


# Get the absolute path to the key file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(script_dir, "crypt_keys/public.pem")

# Load the private key
with open(key_path, "rb") as pub_file:
    private_key = RSA.import_key(pub_file.read())


def serialize_parameters(parameters):
    """Serialize model parameters, converting NumPy arrays to lists."""

    # Convert each client's weights (which are lists of NumPy arrays) into a list of lists
    serialized = []
    for client_weights in parameters:
        # For each client, convert its weights (NumPy arrays) to lists
        serialized_client_weights = [weight.tolist() if isinstance(weight, np.ndarray) else weight for weight in
                                     client_weights]
        serialized.append(serialized_client_weights)

    # Now serialize the whole structure (list of lists of lists)
    return json.dumps(serialized).encode("utf-8")  # Convert to JSON and return as bytes


def send_weights_to_server(weights, num_examples, metadata, server_url="http://localhost:8080/aggregation"):
    try:
        # Prepare the metadata by adding num_examples (if not already included)
        for i, meta in enumerate(metadata):
            meta["num_examples"] = num_examples[i]  # Add the corresponding num_examples to each client's metadata

        # Convert metadata to JSON
        metadata_json = json.dumps(metadata)

        encrypted_weights = serialize_parameters(weights)

        # Send POST request with Content-Type for binary data
        response = requests.post(
            server_url,
            headers={
                "Client-Metadata": metadata_json,  # Send the metadata with num_examples included
                "Content-Type": "application/octet-stream"
            },
            data=encrypted_weights,  # Send the encrypted weights
            timeout=30  # Timeout in seconds
        )

        print(f"Server responded with status code: {response.status_code}")

        # Check the response and decode if successful
        if response.status_code == 200:
            return response.content  # Decode server's encrypted response
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error sending data to server: {e}")
        return None


def request_data_usage_from_agg_server(client_ids, server_url="http://localhost:8080/data-usage"):
    """Send a request to the aggregation server to query data usage for specific clients."""

    try:
        # Prepare the payload with client IDs
        payload = {
            "client_ids": client_ids  # Send a list of client IDs
        }

        # Send the POST request to the aggregation server
        response = requests.post(
            server_url,  # Aggregation server URL
            json=payload,  # Send the client IDs as JSON
            timeout=30  # Timeout in seconds
        )

        # Check the server response
        if response.status_code == 200:
            print("Successfully received data usage info from aggregation server.")
            return response.json()  # Return the response (should be the metadata)
        else:
            print(f"Failed to get data usage info. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error sending request to aggregation server: {e}")
        return None
