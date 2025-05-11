"""FlwrNew: A Flower / TensorFlow app."""
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwrnew.task import load_data, load_model


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
        self.private_key = None
        self.public_key = None

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        updated_weights = self.model.get_weights()
        return updated_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Evaluation complete: Loss = {loss}, Accuracy = {accuracy}")
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    client = FlowerClient(net, data, epochs, batch_size, verbose)
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
