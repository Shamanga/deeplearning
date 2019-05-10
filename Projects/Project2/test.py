from activation_functions.relu import Relu
from activation_functions.tanh import Tanh
from optimizer.stochastic import StochasticGD
from loss.mse import Mse
from data.data_generator import DataGenerator
from container.sequential import Sequential
from layer.linear import Linear
from utils.run_network import RunNetwork

#### Parameters ####
samples = 1000
features = 2
batch = 25
to_shuffle = True
number_of_epochs = 100
#### Parameters ####

def main():
    
    # Data
    train_class = DataGenerator(number_of_examples= samples, number_of_features= features, batch_size = batch, shuffle = to_shuffle)
    test_class = DataGenerator(number_of_examples=samples, number_of_features= features, batch_size = batch, shuffle = to_shuffle)

    train_data, train_target = train_class.generate_data()
    test_data, test_target = test_class.generate_data()

    # Layers
    input_layer = Linear(in_features= features, out_features = 25)
    first_hidden_layer = Linear(in_features = 25, out_features = 25)
    second_hidden_layer = Linear(in_features= 25, out_features = 25)
    output_layer = Linear(in_features= 25, out_features = 2)

    layers = [input_layer, Relu(), first_hidden_layer, Relu(), second_hidden_layer, Relu(), output_layer, Tanh()]

    # Model
    model = Sequential(layers)

    # Optimizer
    optimizer = StochasticGD(parameters_list=model.param(), learning_rate = 0.01)

    # Loss
    loss = Mse()
    
    # Initializing Network class
    rn = RunNetwork(epochs = number_of_epochs, train_class = train_class,\
                    test_class = test_class, model = model, loss = loss, optimizer= optimizer, samples = samples)

    # Training the
    rn.run_network()
    
if __name__=='__main__':
    
    print("Starting Training")
    print("PARAMETERS")
    print("Number of Samples: {} -- Number of features: {} -- Batch size: {} -- Shuffle: {} -- Number of epochs {}"\
          .format(samples, features, batch, to_shuffle, number_of_epochs))
    main()
    print("End Training")