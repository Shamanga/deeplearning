# EE559 Deep learning Project 2 -- Mini Deep Learning Framework

## Description

The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.

## Requirements
The requirements can be installed using:

```
apt-get install pip3-python
```

```
pip3 install torch
```

## Breakdown of the repository

* [activation_functions]: The different implemented activation functions
  * tanh.py
  * sigmoid.py
  * relu.py
  * softmax.py
  * identity.py
  * negative.py
  
* [container]: The containers of the network
  * sequential.py
* [data]: This folde
  * data_generator.py
* [layer]: The layers of the network
  * linear.py
* [loss]: The loss fucntions of the network
  * mse.py
  * cross_entropy.py
* [optimizer]: The optimizer algorithms of the network
  * stochastic.py
* [utils]: The utilities used to train and test the network
  * run_network.py
  
 
## Running the network

```
python test.py
```

## Detailed description

The network built is formed of an input layer with 2 neurons, 3 hidden layers of 25 neurons each, and an output layer with 2 neurons.
The network parameters are:

  * Epochs: 500
  * Shuffle: True
  * Learning rate : 0.01
  * Early stopping: True
  
## Authors
  * Mariam Hakobyan : mariam.hakobyan@epfl.ch
  * Nguyet Minh Nguyen : minh.nguyen@epfl.ch
  * Mazen Fouad A-wali Mahdi : fouad.mazen@epfl.ch
  
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
