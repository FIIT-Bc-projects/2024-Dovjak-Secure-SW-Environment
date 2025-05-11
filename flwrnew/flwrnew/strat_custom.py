import json
import random
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import time
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
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.history import History
from flwrnew.task import load_model
from flwrnew.aggregate_send import send_weights_to_server, request_data_usage_from_agg_server

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


def encrypt_cl_srv_comms(data: bytes, public_key: RSA.RsaKey) -> bytes:
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(data)


# Utility functions for encryption and decryption
def encrypt_with_rsa(data: bytes) -> bytes:
    """Encrypt data using RSA hybrid encryption."""
    # Timing the ecryption time
    start_time = time.time()

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

    end_time = time.time()
    enc_time = round(end_time - start_time, 4)
    print(f"Encryption took {enc_time} sec.")
    log_time("Encryption", enc_time)

    return encrypted_message


def decrypt_with_rsa(encrypted_message: bytes) -> bytes:
    """Decrypt data using RSA hybrid encryption."""
    # Timing the decryption time
    start_time = time.time()

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

    # Decrypt the data using AES
    aes_cipher = AES.new(symmetric_key, AES.MODE_EAX, nonce=nonce)
    plaintext = aes_cipher.decrypt_and_verify(ciphertext, tag)

    end_time = time.time()
    dec_time = round(end_time - start_time, 4)
    print(f"Decryption took {dec_time} sec.")
    log_time("Decryption", dec_time)

    return plaintext


def log_time(action: str, duration: float):
    """Logging timed processes within custom strategy"""
    log_entry = {
        "action": action,
        "duration_sec": duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    file = "history/timing_log.json"
    if os.path.exists(file):
        with open(file, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(log_entry)

    with open(file, "w") as f1:
        json.dump(logs, f1, indent=4)

    print(f"Logged time for {action}")


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
        self.client_public_keys = {}
        self.data_usage_clients = []  # Array for client ids for requests of data usage

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

    def aggregate_client_weights(self, client_weights, num_examples):
        """Client-side aggregation of model weights using weighted average."""
        aggregated_weights = []

        # Calculate total number of examples across all clients
        total_examples = sum(num_examples)

        # Loop through each layer of the model
        for layer_idx in range(len(client_weights[0])):  # Assuming all clients have the same number of layers
            # Initialize the weighted sum for this layer
            weighted_sum = np.zeros_like(client_weights[0][layer_idx])

            # Sum the weighted layers for each client
            for weights, num_example in zip(client_weights, num_examples):
                weighted_sum += weights[layer_idx] * num_example  # Weighted sum of layer weights

            # Normalize by the total number of examples to get the weighted average
            aggregated_weights.append(weighted_sum / total_examples)

        return aggregated_weights

    def serialize_parameters(self, params):
        """Serialize model parameters."""
        return json.dumps([p.tolist() if isinstance(p, np.ndarray) else p for p in params]).encode("utf-8")

    def serialize_client_update(self, client_update):
        """Serialize list of (weights, num_examples) tuples."""
        serializable = []
        for weights, num_examples in client_update:
            serializable.append({
                "weights": [w.tolist() for w in weights],
                "num_examples": num_examples
            })
        return json.dumps(serializable).encode("utf-8")

    def deserialize_parameters(self, params_bytes):
        """Deserialize model parameters."""
        print("Deserialization going on")
        return [np.array(w) for w in json.loads(params_bytes.decode("utf-8"))]

    def request_client_keys(self, clients):
        """Request public keys from the selected clients."""
        print("Requesting all client keys - start")
        for client in clients:
            # Hypothetical RPC call to the client to get its public key
            client_public_key = client.get_public_key()
            self.client_public_keys[client.cid] = client_public_key
            print("Requesting all client keys - end")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        # Timer for measuring the configuration training time
        start_time = time.time()

        param_arrays = parameters_to_ndarrays(parameters)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_configurations = []
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        half_clients = len(clients) // 2

        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append((client, FitIns(parameters, higher_lr_config)))

        end_time = time.time()
        timed_time = round(end_time - start_time, 4)
        print(f"Training config took {timed_time} sec.")
        log_time("Config_fit", timed_time)

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and handle missing metrics."""

        print(f"Aggregating results for round {server_round}")

        # Timing the start of the aggregation fitting process
        start_time = time.time()

        if not results:
            print("No results received from clients. Skipping aggregation.")
            return None, {}

        # Extract model updates (parameters) and number of examples from each client
        client_weights = []
        num_examples = []
        client_metadata = []
        random_client = random.choice(results)  # Choosing random client for request of data used
        rnd_client_id = random_client[0].cid
        if rnd_client_id not in self.data_usage_clients:
            self.data_usage_clients.append(rnd_client_id)

        for client, fit_res in results:
            if fit_res.parameters is not None:
                try:
                    # Convert Flower Parameters object to NumPy arrays
                    client_weights.append(parameters_to_ndarrays(fit_res.parameters))
                    num_examples.append(fit_res.num_examples)

                    # Prepare metadata for each client
                    client_metadata.append({
                        "client_id": client.cid,
                        "epoch": server_round,
                        "weights_hash": str(hash(str(fit_res.parameters))),
                    })

                except Exception as e:
                    print(f"⚠Failed to convert parameters from client {client.cid}: {e}")

        # Check if we actually received valid updates
        if not client_weights:
            print("No valid client updates received. Skipping aggregation.")
            return None, {}

        print(f"{len(client_weights)} clients provided valid updates.")

        try:
            print("Sending client weights to server for aggregation...")
            # Serialize and send client weights and number of examples to the server for aggregation
            aggregated_weights = send_weights_to_server(client_weights, num_examples, client_metadata)

            if aggregated_weights is None:
                raise ValueError("Failed to receive aggregated weights from the server.")

            # Deserialize the aggregated weights (if needed)
            final_params = self.deserialize_parameters(aggregated_weights)

        except Exception as e:
            print(f"Error while sending to remote server: {e}")
            return None, {}

        # Convert aggregated parameters back to Flower Parameters format
        parameters_aggregated = ndarrays_to_parameters(final_params)

        # Calculate aggregated loss and accuracy
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weighted_loss = sum(
            fit_res.num_examples * fit_res.metrics.get("loss", 0.0) for _, fit_res in results) / total_examples
        weighted_accuracy = sum(
            fit_res.num_examples * fit_res.metrics.get("accuracy", 0.0) for _, fit_res in results) / total_examples

        print(f"Aggregated Metrics - Loss: {weighted_loss:.4f}, Accuracy: {weighted_accuracy:.4f}")

        timer1 = time.time()  # Timing the aggregation

        # Save to history
        self.history.add_loss_centralized(server_round, weighted_loss)
        self.history.add_metrics_distributed_fit(server_round, {"accuracy": weighted_accuracy})

        timer2 = time.time()  # Timing the history saving process

        # Update the global model
        self.model.set_weights(parameters_to_ndarrays(parameters_aggregated))

        timer3 = time.time()  # Timing the global mode updating

        # Save the aggregated model
        self.save_model(server_round)

        timer4 = time.time()  # Timing the end of the process + model saving

        full_process_time = round(timer4 - start_time, 4)
        agg_fit_time = round(timer1 - start_time, 4)
        history_time = round(timer2 - timer1, 4)
        glob_model_time = round(timer3 - timer2, 4)
        model_save_time = round(timer4 - timer3, 4)

        print(f"Times for different actions measured -> Full process ended in {full_process_time} sec,"
              f"\nAggregation fitting ended in {agg_fit_time} sec,\nHistory saving lasted {history_time} sec,"
              f"\nGlobal model updating lasted {glob_model_time} sec,\nModel saving lasted {model_save_time} sec.")
        log_time("Model_save", model_save_time)
        log_time("Glob_model_upd", glob_model_time)
        log_time("History_save", history_time)
        log_time("Agg_fit", agg_fit_time)
        log_time("Agg_fit_full", full_process_time)

        if server_round == 3:
            data_usage = request_data_usage_from_agg_server(self.data_usage_clients)

            #print(data_usage)
            print("\nData Usage Information:")
            print("-" * 50)
            print(f"{'Client ID':<25} {'Epoch':<10} {'Weights Hash':<25}")
            print("-" * 50)

            for entry in data_usage:
                client_id = entry['client_id']
                epoch = entry['epoch']
                weights_hash = entry['weights_hash']
                print(f"{client_id:<25} {epoch:<10} {weights_hash:<25}")

            print("-" * 50)

        return parameters_aggregated, {"loss": weighted_loss, "accuracy": weighted_accuracy}

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

        # Convert Parameters object to NumPy arrays
        param_arrays = parameters_to_ndarrays(parameters)

        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        evaluate_configurations = []

        for client in clients:
            # Create evaluation instructions with unencrypted parameters
            evaluate_ins = EvaluateIns(parameters, {})
            evaluate_configurations.append((client, evaluate_ins))

        return evaluate_configurations

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
