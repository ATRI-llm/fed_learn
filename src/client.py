from model import create_model

def local_train(global_weights, X, y, input_dim, epochs=10):
    model = create_model(input_dim)
    model.set_weights(global_weights)
    model.fit(X, y, epochs=epochs, batch_size=2, verbose=0)
    return model.get_weights()
