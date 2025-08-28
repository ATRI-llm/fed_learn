import numpy as np
from preprocess import preprocess_data, split_clients
from model import create_model
from client import local_train

def federated_average(weights_list):
    return [np.mean(np.array(w), axis=0) for w in zip(*weights_list)]

def main():
    features, labels = preprocess_data('data/loan_applications.csv')
    client_data = split_clients(features, labels)
    input_dim = features.shape[1]
    global_model = create_model(input_dim)
    n_rounds = 5

    for round in range(n_rounds):
        client_weights = []
        for X_client, y_client in client_data:
            weights = local_train(global_model.get_weights(), X_client, y_client, input_dim)
            client_weights.append(weights)
        new_global_weights = federated_average(client_weights)
        global_model.set_weights(new_global_weights)
        loss, acc = global_model.evaluate(features, labels, verbose=0)
        print(f"Round {round+1} - Loss: {loss:.4f}, Acc: {acc:.4f}")

    loss, acc = global_model.evaluate(features, labels, verbose=0)
    print(f"Final Model - Loss: {loss:.4f}, Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
