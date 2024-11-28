"""FlwrNew: A Flower / TensorFlow app."""

import pickle
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.history import History
from flwrnew.task import load_model
from flwrnew.strat_custom import FedCustom


def save_history(history: History, file_name: str = "training_history.pkl"):
    """Save the Flower history object to a file using pickle."""
    with open(file_name, "wb") as f:
        pickle.dump(history, f)
    print(f"History saved to {file_name}")


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initiating history
    history = History()

    # Define the strategy and pass the model and history object to track metrics
    strategy = FedCustom(model=load_model(), history=history)

    # Define server config
    config = ServerConfig(num_rounds=num_rounds)

    save_history(history, "training_history.pkl")

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

