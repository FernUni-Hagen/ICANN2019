"""
This module contains the genetic algorithm's configuration, to be edited by the user.
"""


import logging
import tensorflow as tf
from enum import Enum
import ga_input
import os

# Enumerations


class SelectionMode(Enum):
    TOURNAMENT_DUEL = 1
    TOURNAMENT_MANY = 2
    ROULETTE = 3


# Possible activation functions
act_fns = {
    'RELU': tf.nn.relu,
    'SIGMOID': tf.nn.sigmoid,
    'TANH': tf.nn.tanh,
    'ELU': tf.nn.elu
}


# GA Configuration parameters

# Number of iterations performed in total. For unlimited number of iterations set value None.
num_iterations = 2000
# If num_iterations is None, specify number of iterations over which various decays for
# the application of genetic operators are calculated. Must be > 0!
decay_over_iterations = 2000

# Log files and logging levels for both the genetic algorithm and TensorFlow
tf_logfile = "/tmp/tf.log" if os.name != "nt" else "C:\\tmp\\tf.log"
tf_logging_level = logging.DEBUG
ga_logfile = "/tmp/genalg.log"if os.name != "nt" else "C:\\tmp\\genalg.log"
ga_logging_level = logging.INFO

# Files for storing the net dictionary, the current population, and further information
# after termination of the genetic algorithm
net_dict_file = "/tmp/ganetdict.json"if os.name != "nt" else "C:\\tmp\\ganetdict.json"
info_file = "/tmp/gainfo.json"if os.name != "nt" else "C:\\tmp\\gainfo.json"
population_file = "/tmp/gapop.pickle"if os.name != "nt" else "C:\\tmp\\gapop.pickle"

# A checkpoint will be created every checkpoint_after_iterations iterations
checkpoint_after_iterations = 100

# It is possible to continue the genetic algorithm's last run by reading from net dictionary and population files.
continue_from_last_run = False

# Number of nets in population
max_population = 10

# Maximum number of net parameters, nets with more parameters are discarded prior to evaluation.
max_num_params = 5000000

# Dimensions of input and output layer
input_layer_x = 28
input_layer_y = 28
input_channels = 1
output_layer_x = 10
output_layer_y = 1

# Probabilities of genetic operators (must add up to 1.0)
prob_insert_layer = 0.2
prob_delete_layer = 0.2
prob_switch_layers = 0.1
prob_modify_layer = 0.4
prob_crossover = 0.1

# Minimum and maximum values for crossover points
# Only applicable if force_canonical_architecture == False
min_crossover_points = 1
max_crossover_points = 1

# Probabilities for modification of dense layers (must add up to 1.0)
prob_mod_dense_neurons = 1.0
prob_mod_dense_actfn = 0.0

# Probabilities for modification of convolutional layers (must add up to 1.0)
prob_mod_conv_kernel = 0.3
prob_mod_conv_stride = 0.3
prob_mod_conv_filters = 0.4
prob_mod_conv_padding = 0.0
prob_mod_conv_actfn = 0.0

# Probabilities for modification of pooling layers (must add up to 1.0)
prob_mod_pool_kernel = 0.45
prob_mod_pool_stride = 0.45
prob_mod_pool_pooling = 0.1
prob_mod_pool_padding = 0.0

# Probabilities for kernel modifications (must add up to 1.0)
prob_mod_kernel_x = 0.25
prob_mod_kernel_y = 0.25
prob_mod_kernel_both = 0.5

# Probabilities for stride modifications (must add up to 1.0)
prob_mod_stride_x = 0.25
prob_mod_stride_y = 0.25
prob_mod_stride_both = 0.5

# Dense layer modification increment/decrement, decay, and probability of increment
# At the end of the genetic algorithm (or after decay_over_iterations steps), step size will be
# dense_layer_step * dense_layer_step_decay. In between, step size will decrease exponentially.
dense_layer_step = 50
dense_layer_step_decay = 0.1
dense_layer_incr_prob = 0.5

# Convolutional and pooling layer kernel modification increment/decrement, decays, and probability of increment
kernel_mod_step_x = 1
kernel_mod_step_x_decay = 1.0
kernel_mod_step_y = 1
kernel_mod_step_y_decay = 1.0
kernel_mod_incr_prob = 0.5

# Convolutional and pooling layer stride modification increment/decrement , decays, and probability of increment
stride_mod_step_x = 1
stride_mod_step_x_decay = 1.0
stride_mod_step_y = 1
stride_mod_step_y_decay = 1.0
stride_mod_incr_prob = 0.5

# Convolutional layer filter increment/decrement, decay, and probability of increment
filter_mod_step = 20
filter_mod_step_decay = 0.25
filter_mod_incr_prob = 0.5

# Maximum number of layers
max_layers = 10

# Restrictions of hidden layers
# Maximum size of fully connected layer
dense_max_units = 1024
# Maximum dimensions for kernel and stride for convolutional and pooling layers
max_kernel_x = 6
max_kernel_y = 6
max_stride_x = 3
max_stride_y = 3
# Maximum depth of convolutional layer
max_filters = 128

# Default activation function, possible values are listed in dictionary "act_fns" above
default_act_fn = 'RELU'

# Default padding, possible values are 'SAME', 'VALID', or 'RANDOM'
default_padding = 'SAME'

# Default pooling mode, possible values are 'MAX', 'AVERAGE', or 'RANDOM'
default_pooling = 'RANDOM'

# Specify whether nets are restricted to architectures that feature fully connected layers as last hidden layers only
force_canonical_architecture = True

# Number of maximum evaluations for the same architecture if score is above or below threshold, respectively
# This offers the possibility to evaluate promising architectures more often
max_evals_above_threshold = 1
max_evals_below_threshold = 1
eval_threshold = 0.95
# Maximum number of trainings per evaluation, decreasing by 1 per max_trainings_decrease_per_free_params, i.e. a linear
# decay is employed
max_trainings_per_evaluation = 1
max_trainings_decrease_per_free_params = 100000

# Selection mode, possible values are listed in the enumeration above
selection_mode = SelectionMode.ROULETTE
# Number of competing individuals in TOURNAMENT_MANY selection mode, must be > 0 and <= max_population
number_of_contestants = 4
# Probability of selecting the individual with higher fitness in TOURNAMENT_DUEL selection mode
winning_probability = 0.75
# If two architectures both reach the fitness threshold, selection is based on number of free parameters
fitness_threshold_selection = 1.0
# Same as fitness_threshold_selection but for population update
fitness_threshold_pop_update = 1.0


# ANN training parameters

# Directory for models
model_dir = "/tmp/"if os.name != "nt" else "C:\\tmp\\"
# Directory for best current model
best_dir = model_dir + "ga_best_model/"

# Specify data retrieval function
# Adjust dimensions of input and output layer accordingly (above)
data_retrieval_fn = ga_input.load_data_mnist

# Specify optimizer. Possible values are 'GRADIENTDESCENT', 'MOMENTUM', 'RMSPROP', 'ADAGRAD', 'ADAM'.
optimizer = 'MOMENTUM'

# Parameters for optimizers. For details refer to API documentation under the URLs provided.
# For learning rate (applicable to all the optimizers), see below.
# Gradient Descent optimizer: https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
# No additional parameters required
# Momentum optimizer: https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
momentum_momentum = 0.9
momentum_use_nesterov = True
# RMSProp optimizer: https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
rmsprop_decay = 0.9
rmsprop_momentum = 0.0
rmsprop_epsilon = 1e-10
rmsprop_centered = False
# AdaGrad optimizer: https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer
adagrad_initial_accumulator_value = 0.1
# Adam optimizer: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-08

# Optimizer's learning rate, only applicable when learning rate scheduling is disabled.
# Default value for Adam optimizer is 0.001, other optimizers do not feature default learning rates.
learning_rate = 0.1

# Parameters for early stopping
# Net is evaluated every validation_interval number of steps
# Training is stopped, if loss has not decreased over the last max_validations_since_best_loss validations
# or if there has not been a major improvement over the last max_validations_without_major_improvement validations
validation_interval = 5000
max_validations_since_best_loss = 3  # should be > 0. Set to 1 for mandatory improvement
max_validations_without_major_improvement = 3  # should be > 0. Set to 1 for mandatory improvement
major_improvement_threshold = 0.005

# Batch size for training set
train_batch_size = 128
# Maximum number of steps during a single training run
max_training_steps = 100000
# Set use of early stopping
employ_early_stopping = True
# Set use of weight sharing
employ_weight_sharing = True

# Set use of initialization strategies depending on activation function for weights
use_initializer = True

# Set use of learning rate scheduling
use_learning_rate_scheduling = True
initial_learning_rate = 0.1
# Learning rate will decrease exponentially, reaching initial_learning_rate * learning_rate_decay_rate after
# learning_rate_decay_steps steps.
learning_rate_decay_steps = 100000
learning_rate_decay_rate = 1/10

# Set use of batch normalization and momentum
employ_batch_normalization_dense = False
employ_batch_normalization_conv = True
batch_normalization_momentum = 0.99

# Set use of dropout and dropout rate. If both batch normalization and dropout are active, convolutional layers will
# only receive batch normalization while dropout will only be applied to dense layers
employ_dropout_dense = True
employ_dropout_conv = False

dropout_rate = 0.5

