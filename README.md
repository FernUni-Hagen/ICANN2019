# ICANN2019
Code complementing the research article "Compute-Efficient Neural Network Architecture Optimization by a Genetic Algorithm", accepted at ICANN 2019

# File Overview
- `ga_config.py`: This module contains the genetic algorithm's configuration, to be edited by the user.
- `ga_estimator.py`: This module contains the custom estimator for TensorFlow to be used as evaluation component in the genetic algorithm. See [TensorFlow Estimators](https://www.tensorflow.org/guide/custom_estimators) for the creation of custom estimators. The code in this file originates from that tutorial and has been heavily modified and extended to serve as an
instrument for the evaluation of nets in the genetic algorithm.
- `ga_genops.py`: This module contains the functions implementing the genetic algorithm's genetic operators.
- `ga_genotype.py`: This module contains the classes implementing the genetic algorithm's genotype.
- `ga_input.py`: This module provides functions that deliver input data for the estimator, function of choice must be specified in the configuration file.
- `ga_main.py`: This module serves as the genetic algorithm's main component, implementing initialization, selection, population update, and logging capabilities.
- `package_list.txt`: This file contains a list of packages installed in the test environment and  exported from anaconda.

# Execution
To run the genetic algorithm:
```
python ga_main.py
```
