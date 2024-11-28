# cli.py
import argparse
import subprocess
import flwr as fl
from strat_custom import FedCustom  # Import the FedCustom strategy
from flwr.server.history import History  # History object for logging training process


# Function to start the REST API aggregation server
def start_aggregation_server():
    subprocess.Popen(["python", "aggregate_server.py"])
    print("Started aggregation server...")


# Function to start the Flower server with FedCustom strategy
def start_flower_server():
    # Initialize Flower's history object to track training results
    history = History()

    # Initialize FedCustom strategy with the global model and history
    strategy = FedCustom(
        history=history,
        fraction_fit=0.5,  # You can modify the fraction of clients here
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )

    # Start the Flower server with FedCustom strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},  # Adjust the number of rounds as needed
        strategy=strategy
    )
    print("Started Flower server with FedCustom strategy...")


# Main function for the CLI
def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Flower CLI")
    parser.add_argument("command", type=str, help="Command to run: run, start_server, etc.")

    args = parser.parse_args()

    if args.command == "run":
        # Start both the REST API server and the Flower server with FedCustom strategy
        start_aggregation_server()
        start_flower_server()
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
