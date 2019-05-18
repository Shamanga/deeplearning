# EE559 Deep learning Project 2 -- Mini deep-learning Classificationwith weight sharing and auxiliary losses

## Description

The goal of the project is to compare different architectures, and assess the performance improvement that can be achieved through weight sharing, or using auxiliary losses. For the latter, the training can in particular take advantage of the availability of the classes of the two digits in each pair, beside the Boolean value truly of interest.

## Requirements
The requirements can be installed using:

```
apt-get install pip3-python
```

```
pip3 install torch
```

## Breakdown of the repository

* nets: The different architecures of neural nets and their train test functions
  * NN1.py
  * NN2.py
  * train_test.py
  
* utils: Processing the Mnist dataset
  * MnistPairs
  * utils.py
  
* test.py: 
  
 
## Running the network

```
python test.py
```

## Detailed description

Two networks have been trained for the 2 channel input with the following parameters

  * Epochs: 25
  * Shuffle: True
  * Learning rate : 0.01
  * Early stopping: True
  * Batch size: 16
  * Train data szie: 1000 
  * Test data szie: 1000
  
## Authors
  * Mariam Hakobyan : mariam.hakobyan@epfl.ch
  * Nguyet Minh Nguyen : minh.nguyen@epfl.ch
  * Mazen Fouad A-wali Mahdi : fouad.mazen@epfl.ch
  
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
