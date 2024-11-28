"""FlwrNew: A Flower / TensorFlow app."""
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwrnew.task import load_data, load_model
import json
import numpy as np


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.key_pair = RSA.generate(2048)
        self.server_public_key = None

    def initialize_keys(self, server_public_key):
        """Initialize the server's public key."""
        self.server_public_key = server_public_key

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}
    # def fit(self, parameters, config):
    #     # Decrypt parameters from the server
    #     rsa_cipher = PKCS1_OAEP.new(self.key_pair)
    #     print("Pred decryptom na fit cliente")
    #     decrypted_params = rsa_cipher.decrypt(parameters)
    #     print("Za decryptom na fit cliente")
    #
    #     # Deserialize the decrypted parameters
    #     model_weights = [np.array(w) for w in json.loads(decrypted_params.decode("utf-8"))]
    #
    #     # Set the global model weights
    #     self.model.set_weights(model_weights)
    #
    #     # Train the model locally
    #     self.model.fit(
    #         self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose
    #     )
    #
    #     # Get the updated model parameters
    #     updated_weights = self.model.get_weights()
    #
    #     # Serialize and encrypt the updated weights with the server's public key
    #     serialized_weights = json.dumps([w.tolist() for w in updated_weights]).encode("utf-8")
    #     rsa_cipher = PKCS1_OAEP.new(self.server_public_key)
    #     encrypted_weights = rsa_cipher.encrypt(serialized_weights)
    #
    #     # Return encrypted weights
    #     return encrypted_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}
    # def evaluate(self, parameters, config):
    #     # Decrypt parameters from the server
    #     rsa_cipher = PKCS1_OAEP.new(self.key_pair)
    #     print("pred encryptom na eval cliente")
    #     print(f"Type of parameters before decryption: {type(parameters)}")
    #     decrypted_params = rsa_cipher.decrypt(parameters)
    #     print("za encryptom na eval cliente")
    #
    #     # Deserialize the decrypted parameters
    #     model_weights = [np.array(w) for w in json.loads(decrypted_params.decode("utf-8"))]
    #
    #     # Set the global model weights
    #     self.model.set_weights(model_weights)
    #
    #     # Evaluate the model on the test data
    #     loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    #     return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
