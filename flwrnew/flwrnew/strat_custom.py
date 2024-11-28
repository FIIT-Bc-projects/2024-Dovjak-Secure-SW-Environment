import base64
import json
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes

from datetime import datetime
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.history import History
from flwrnew.task import load_model
from flwrnew.aggregate_send import send_weights_to_server

# Get current date and time for use in filenames
date_today_raw = datetime.now()
date_today = date_today_raw.strftime("%Y-%m-%d, %H-%M")

# Load the RSA private key for decryption
script_dir = os.path.dirname(os.path.abspath(__file__))
private_key_path = os.path.join(script_dir, "crypt_keys/private.pem")
with open(private_key_path, "rb") as priv_file:
    private_key = RSA.import_key(priv_file.read())

# Load the RSA public key for encryption (shared with clients)
public_key_path = os.path.join(script_dir, "crypt_keys/public.pem")
with open(public_key_path, "rb") as pub_file:
    public_key = RSA.import_key(pub_file.read())


# Utility functions for encryption and decryption
def encrypt_with_rsa(data: bytes) -> bytes:
    """Encrypt data using RSA hybrid encryption."""
    # Generate a random symmetric key for AES encryption
    symmetric_key = get_random_bytes(16)

    # Encrypt the data using AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX)
    ciphertext, tag = aes_cipher.encrypt_and_digest(data)

    # Encrypt the symmetric key using RSA
    rsa_cipher = PKCS1_OAEP.new(public_key)
    encrypted_key = rsa_cipher.encrypt(symmetric_key)

    # Combine encrypted key, AES nonce, ciphertext, and tag
    encrypted_message = encrypted_key + aes_cipher.nonce + ciphertext + tag

    # Log lengths for debugging
    print(f"RSA Encrypted Key Length: {len(encrypted_key)}")
    print(f"AES Nonce Length: {len(aes_cipher.nonce)}")
    print(f"AES Ciphertext Length: {len(ciphertext)}")
    print(f"AES Tag Length: {len(tag)}")
    print(f"Total Encrypted Message Length: {len(encrypted_message)}")

    return encrypted_message


def decrypt_with_rsa(encrypted_message: bytes) -> bytes:
    """Decrypt data using RSA hybrid encryption."""
    # Validate the minimum length of the encrypted message
    if len(encrypted_message) < 256 + 16 + 16:
        raise ValueError(f"Invalid encrypted message length: {len(encrypted_message)}")

    # Extract RSA-encrypted symmetric key and AES-encrypted data
    encrypted_key = encrypted_message[:256]
    encrypted_data = encrypted_message[256:]

    # Decrypt the symmetric key using RSA
    rsa_cipher = PKCS1_OAEP.new(private_key)
    symmetric_key = rsa_cipher.decrypt(encrypted_key)

    # Extract AES components
    nonce = encrypted_data[:16]
    ciphertext = encrypted_data[16:-16]
    tag = encrypted_data[-16:]

    # Log lengths for debugging
    print(f"Nonce Length: {len(nonce)}")
    print(f"Ciphertext Length: {len(ciphertext)}")
    print(f"Tag Length: {len(tag)}")

    # Decrypt the data using AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX, nonce=nonce)
    plaintext = aes_cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext


# Simple class to hold the global model and return weights
class Net:
    def __init__(self):
        pass

    def get_weights(self):
        # Load model weights via an external function from `flwrnew.task`
        return load_model().get_weights()


# Custom federated learning strategy class
class FedCustom(Strategy):
    def __init__(
            self,
            model,
            history: History,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            save_dir: str = "models",
            history_dir: str = "history",

    ) -> None:
        super().__init__()  # Call the parent class initializer

        # Parameters controlling how many clients to use during training/evaluation
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.previous_results = None  # To store previous results
        self.server_key_pair = None
        self.client_public_keys = {}
        self.model = model  # The global model (e.g., Keras model)
        self.save_dir = save_dir  # Directory to save models
        self.history = history  # Flower history object to track results
        self.history_dir = history_dir  # Directory to save history

        # Create the directory to save models if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Create the directory to save history if it does not exist
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

    def __repr__(self) -> str:
        return "FedCustom"  # Representation for this strategy

    def initialize_keys(self):
        """Generate server key pair and clear client public keys."""
        self.server_key_pair = RSA.generate(2048)
        self.client_public_keys = {}

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # Initialize the model and get its weights
        net = Net()
        ndarrays = net.get_weights()
        # Convert NumPy arrays to Flower Parameters format
        return ndarrays_to_parameters(ndarrays)

    def serialize_parameters(self, params):
        """Serialize model parameters."""
        return json.dumps([w.tolist() if isinstance(w, np.ndarray) else [w] for w in params]).encode("utf-8")

    def deserialize_parameters(self, params_bytes):
        """Deserialize model parameters."""
        print("Deserialization going on")
        return [np.array(w) for w in json.loads(params_bytes.decode("utf-8"))]

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Convert Parameters object to NumPy arrays for inspection
        param_arrays = parameters_to_ndarrays(parameters)

        # Ensure keys are initialized for each run
        if self.server_key_pair is None:
            self.initialize_keys()

        # Sample clients based on the available number of clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Configure clients with different learning rates for experimentation
        n_clients = len(clients)
        half_clients = n_clients // 2  # Half clients with a lower learning rate
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []

        # Assign learning rates based on client index
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )

        return fit_configurations  # Return configured clients for training

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and handle missing metrics."""

        # Extract model updates (parameters) and the number of examples from each client
        client_updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Number of layers from the first client
        num_layers = len(client_updates[0][0])

        # Check consistency for each layer across clients
        for layer_idx in range(num_layers):
            # Collect the shapes of the current layer from all clients
            layer_shapes = [params[layer_idx].shape for params, _ in client_updates]

            # Get unique shapes and count occurrences
            unique_shapes = set(layer_shapes)

            # Raise an error if there are inconsistencies
            if len(unique_shapes) != 1:
                inconsistent_clients = [i for i, shape in enumerate(layer_shapes) if shape != layer_shapes[0]]
                raise ValueError(
                    f"Inconsistent shapes detected for layer {layer_idx}: {unique_shapes}. "
                    f"Inconsistent client indices: {inconsistent_clients}. Each client should have "
                    f"matching layer shapes for aggregation."
                )
            else:
                print(f"Layer {layer_idx} shapes are consistent across clients: {unique_shapes}")

        # Perform aggregation layer by layer
        aggregated_params = []
        for layer_idx in range(num_layers):
            # Gather weights and number of examples per client for the current layer
            layer_weights = np.array([params[layer_idx] for params, _ in client_updates])
            num_examples = np.array([num for _, num in client_updates])

            # Ensure num_examples is a 1D array for weighted averaging
            if num_examples.ndim != 1:
                num_examples = num_examples.flatten()

            # Weighted average of the layer weights
            weighted_avg = np.average(layer_weights, axis=0, weights=num_examples)

            # Convert to array if it's a scalar to maintain consistency
            if np.isscalar(weighted_avg):
                weighted_avg = np.array([weighted_avg])

            aggregated_params.append(weighted_avg)

        # Log types of aggregated_params to ensure they're arrays or lists before sending
        # print(f"Aggregated parameter types: {[type(param) for param in aggregated_params]}")

        # Serialize and encrypt aggregated weights
        # print("Length of aggregated parameters is: ", len(aggregated_params))
        serialized_params = self.serialize_parameters(aggregated_params)
        # print("Length of serialized parameters is: ", len(serialized_params))
        encrypted_params = encrypt_with_rsa(serialized_params)

        # print(f"Aggregated parameter types: {[type(param) for param in aggregated_params]}")

        # Send encrypted aggregated weights to the remote server
        try:
            # print("Sending aggregated weights to the remote API server...")
            aggregated_params_from_server = send_weights_to_server(encrypted_params)
            # print(f"Raw response content (first 100 bytes): {aggregated_params_from_server[:100]}")
            # print(f"Response type: {type(aggregated_params_from_server)}, Length: {len(aggregated_params_from_server)}")
            if aggregated_params_from_server is None:
                raise ValueError("Failed to receive aggregated weights from the remote server.")
            print("Aggregated weights successfully sent and returned from the remote server.")

            # Decrypt weights to simulate the return from the server
            # print("Before decryption")
            decrypted_params = decrypt_with_rsa(aggregated_params_from_server)
            # print("Length of decrypted parameters from server is: ", len(decrypted_params))
            # print("After decryption and before deserializing")
            final_params = self.deserialize_parameters(decrypted_params)
            print("Encrypted aggregated weights successfully sent and decrypted from the server")
        except Exception as e:
            print(f"Error while sending to remote server: {e}")
            return None, {}

        # Convert the aggregated weights back to Flower Parameters format
        parameters_aggregated = ndarrays_to_parameters(final_params)

        # Tracking training loss and accuracy from clients
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        # Aggregate losses and accuracies
        for _, fit_res in results:
            # Get loss and accuracy from client results
            loss = fit_res.metrics.get("loss", 0.0)  # Default to 0.0 if not provided
            accuracy = fit_res.metrics.get("accuracy", 0.0)  # Default to 0.0 if not provided

            # Weight metrics by the number of examples for each client
            weighted_loss += fit_res.num_examples * loss
            weighted_accuracy += fit_res.num_examples * accuracy

        # Final weighted average
        if total_examples > 0:
            weighted_loss /= total_examples
            weighted_accuracy /= total_examples
        else:
            weighted_loss = 0.0
            weighted_accuracy = 0.0

        # Log final metrics
        print(f"Round {server_round} - Weighted Training Loss: {weighted_loss}")
        print(f"Round {server_round} - Weighted Training Accuracy: {weighted_accuracy}")

        # Save to history
        self.history.add_loss_centralized(server_round, weighted_loss)
        self.history.add_metrics_distributed_fit(server_round, {"accuracy": weighted_accuracy})

        metrics_aggregated = {"loss": weighted_loss, "accuracy": weighted_accuracy}

        # Update the global model with the aggregated weights
        param_arrays = parameters_to_ndarrays(parameters_aggregated)
        self.model.set_weights(param_arrays)

        # Save the model after each round
        self.save_model(server_round)

        return parameters_aggregated, metrics_aggregated

    def save_model(self, server_round: int):
        """Save the global model to disk after each round."""
        model_path = f"{self.save_dir}/global_model_date_{date_today}_round_{server_round}.keras"
        self.model.save(model_path)  # Save the model to the specified path
        print(f"Model saved to {model_path}")

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs for evaluation
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        # Aggregate evaluation results from clients
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        weighted_loss = 0
        weighted_accuracy = 0

        for _, evaluate_res in results:
            loss = evaluate_res.loss
            accuracy = evaluate_res.metrics.get("accuracy", None)
            weighted_loss += evaluate_res.num_examples * loss
            if accuracy is not None:
                weighted_accuracy += evaluate_res.num_examples * accuracy

        # Calculate weighted average loss and accuracy
        if total_examples > 0:
            weighted_loss /= total_examples
            weighted_accuracy /= total_examples if weighted_accuracy > 0 else None
        else:
            weighted_loss = None
            weighted_accuracy = None

        print(f"Round {server_round} - Eval Loss: {weighted_loss}, Eval Accuracy: {weighted_accuracy}")

        # Save evaluation results to history
        if weighted_loss is not None:
            self.history.add_loss_centralized(server_round, weighted_loss)
        if weighted_accuracy is not None:
            self.history.add_metrics_centralized(server_round, {"accuracy": weighted_accuracy})

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # No global evaluation logic in this strategy.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for fitting."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

