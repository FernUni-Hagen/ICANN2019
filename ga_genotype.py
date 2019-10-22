"""
This module contains the classes implementing the genetic algorithm's genotype.
"""


import ga_config as cf
from enum import Enum
import math


# Class that enumerates all the possible layer types
class ConnectionType(Enum):
    INPUT = 1
    DENSE = 2
    CONVOLUTIONAL = 3
    POOLING = 4


# Class for network layer containing all the information about a layer needed to employ genetic operators and to
# transform genotype to phenotype
class ArtNeurLayer:

    def __init__(self, connection_type, activation_fn=cf.default_act_fn, neurons_x=None, neurons_y=None, kernel_x=None,
                 kernel_y=None, stride_x=None, stride_y=None, filters=None, padding=None, pooling=None):
        self.connection_type = connection_type
        self.neurons_x = neurons_x
        self.neurons_y = neurons_y
        if connection_type == ConnectionType.DENSE:
            self.activation_fn = activation_fn
            self.neurons = neurons_x * neurons_y
        elif connection_type == ConnectionType.CONVOLUTIONAL or connection_type == ConnectionType.POOLING:
            self.kernel_x = kernel_x
            self.kernel_y = kernel_y
            self.stride_x = stride_x
            self.stride_y = stride_y
            self.padding = padding
            if connection_type == ConnectionType.CONVOLUTIONAL:
                self.activation_fn = activation_fn
                self.filters = filters
            if connection_type == ConnectionType.POOLING:
                self.pooling = pooling


# Class for artificial neural network containing the layer list as well as additional information regarding the net
# and methods for modifying the net's architecture
class ArtNeurNet:

    def __init__(self, id_string, input_layer, output_layer):
        # id_string comprises information about the net's architecture, not unique!
        self.id_string = id_string
        self.layers = [input_layer, output_layer]
        self.score = 0.0
        # net_parameters is the amount of the net's free parameters in terms of layers' weights and biases
        self.net_parameters = self.calculate_net_params()
        # free_parameters includes net_parameters as well as free parameters e.g. from batch normalization
        # Value is calculated by the estimator during evaluation
        self.free_parameters = 0
        # Record iteration number of evaluation
        self.iteration_evaluated = 0
        # Weights and bias values for best evaluation
        self.weights = None
        # Record position of last change in net architecture for reuse of weights during evaluation
        self.pos_last_change = 0

    def insert_layer(self, layer, position):
        success = False
        # Input and output layer must keep their positions
        if position in range(1, len(self.layers)):
            # Insert layer
            self.layers.insert(position, layer)
            success = True
            # For pooling layer, check if layer has any effect
            if layer.connection_type == ConnectionType.POOLING:
                success = self.check_pooling_layer(layer)
            # Update layer dimensions and net parameters
            self.check_net(position)
        return success

    def delete_layer_by_position(self, position):
        # Input and output layer cannot be deleted
        if position in range(1, len(self.layers) - 1):
            # Retrieve layer and delete
            layer_to_delete = self.layers[position]
            self.layers.remove(layer_to_delete)
            # Update layer dimensions and net parameters
            self.check_net(position)
            return True
        return False

    def switch_layers_by_position(self, position_1, position_2):
        # Input and output layer have to remain in their respective positions
        if position_1 in range(1, len(self.layers) - 1) and position_2 in range(1, len(self.layers) - 1):
            self.layers[position_1], self.layers[position_2] = self.layers[position_2], self.layers[position_1]
            # Update layer dimensions and net parameters
            if position_1 < position_2:
                self.check_net(position_1)
            else:
                self.check_net(position_2)
            return True
        return False

    def __set_neurons(self, layer, units):
        # Output layer cannot be modified
        if layer.connection_type == ConnectionType.DENSE and units > 0 and\
                layer is not self.layers[len(self.layers) - 1]:
            # Dense layer has one dimension only
            layer.neurons = units
            layer.neurons_x = units
            layer.neurons_y = 1
            return True
        return False

    def set_neurons_by_position(self, position, units):
        # Input and output layer cannot be modified
        if position in range(1, len(self.layers) - 1):
            layer_to_modify = self.layers[position]
            if self.__set_neurons(layer_to_modify, units):
                # Update layer dimensions and net parameters
                self.check_net(position)
                return True
        return False

    def set_kernel_x_by_position(self, position, value):
        # Input and output layer cannot be modified
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if (layer.connection_type == ConnectionType.CONVOLUTIONAL or
                    layer.connection_type == ConnectionType.POOLING) and value > 0:
                layer.kernel_x = value
                if layer.connection_type == ConnectionType.POOLING:
                    # For pooling layer, check if layer has any effect
                    self.check_pooling_layer(layer)
                # Update layer dimensions and net parameters
                self.check_net(position)
                return True
        return False

    def set_kernel_y_by_position(self, position, value):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if (layer.connection_type == ConnectionType.CONVOLUTIONAL or
                    layer.connection_type == ConnectionType.POOLING) and value > 0:
                layer.kernel_y = value
                if layer.connection_type == ConnectionType.POOLING:
                    self.check_pooling_layer(layer)
                self.check_net(position)
                return True
        return False

    def set_stride_x_by_position(self, position, value):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if (layer.connection_type == ConnectionType.CONVOLUTIONAL or
                    layer.connection_type == ConnectionType.POOLING) and value > 0:
                layer.stride_x = value
                if layer.connection_type == ConnectionType.POOLING:
                    self.check_pooling_layer(layer)
                self.check_net(position)
                return True
        return False

    def set_stride_y_by_position(self, position, value):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if (layer.connection_type == ConnectionType.CONVOLUTIONAL or
                    layer.connection_type == ConnectionType.POOLING) and value > 0:
                layer.stride_y = value
                if layer.connection_type == ConnectionType.POOLING:
                    self.check_pooling_layer(layer)
                self.check_net(position)
                return True
        return False

    def set_padding_by_position(self, position, padding):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if ((layer.connection_type == ConnectionType.CONVOLUTIONAL or layer.connection_type ==
                 ConnectionType.POOLING)
                    and (padding == 'VALID' or padding == 'SAME')):
                layer.padding = padding
                self.check_net(position)
                return True
        return False

    def set_activation_function_by_position(self, position, act_fn):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if (layer.connection_type == ConnectionType.CONVOLUTIONAL or
                    layer.connection_type == ConnectionType.DENSE):
                layer.activation_fn = act_fn
                self.pos_last_change = position
                return True
        return False

    def set_filters_by_position(self, position, filters):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if layer.connection_type == ConnectionType.CONVOLUTIONAL and filters > 0:
                layer.filters = filters
                self.pos_last_change = position
                # Update net parameters
                self.net_parameters = self.calculate_net_params()
                return True
        return False

    def set_pooling_by_position(self, position, pooling):
        if position in range(1, len(self.layers) - 1):
            layer = self.layers[position]
            if layer.connection_type == ConnectionType.POOLING and (pooling == 'MAX' or pooling == 'AVERAGE'):
                layer.pooling = pooling
                self.check_net(position)
                return True
        return False

    def change_architecture(self, new_layers):
        # Ascertain correct type
        for layer in new_layers:
            if not isinstance(layer, ArtNeurLayer):
                return False
        # There has to be an input layer
        if new_layers[0].connection_type is not ConnectionType.INPUT:
            return False
        self.layers = new_layers
        # Update layer dimensions and net parameters
        self.check_net(1)
        return True

    def check_net(self, from_position):
        self.pos_last_change = from_position
        position = from_position
        while position != len(self.layers) - 1:
            # Compute dimensions of convolutional and pooling layers
            self.compute_layer_dimensions(position)
            # Ensure that receptive fields' sizes are adequate
            # Returns position of modification, from which computation of layer dimensions has to be initiated in the
            # next iteration
            position = self.check_receptive_fields(position)
        # Calculate number of net parameters
        self.net_parameters = self.calculate_net_params()

    def check_receptive_fields(self, from_position):
        # Various evolutional changes might result in a net which would cause errors when trained.
        # It has to be ensured that kernel sizes of convolutional and pooling layers do not exceed
        # previous layer's dimensions (at least when no padding is applied).
        # For this to be guaranteed, kernel sizes are adapted to previous layer's dimensions if necessary,
        # for all layers. Aside from this technical requirement, kernel size should be kept rather small
        # due to computational efficiency considerations.
        for i in range(from_position, len(self.layers)):
            layer = self.layers[i]
            # Ensure that convolution and pooling kernel dimensions do not exceed dimensions of previous layer
            # If necessary, adjust kernel size
            if layer.connection_type == ConnectionType.CONVOLUTIONAL or layer.connection_type == ConnectionType.POOLING:
                previous_layer = self.layers[i - 1]
                # If kernel size has to be changed, return immediately after modification to recompute dimensions of
                #  succeeding layers
                if layer.kernel_x > previous_layer.neurons_x:
                    layer.kernel_x = previous_layer.neurons_x
                    return i
                if layer.kernel_y > previous_layer.neurons_y:
                    layer.kernel_y = previous_layer.neurons_y
                    return i
        # No changes necessary, return maximum position value
        return len(self.layers) - 1

    def check_pooling_layer(self, layer):
        # If pooling kernel and stride is 1 in all dimensions, layer can be eliminated
        if layer.connection_type == ConnectionType.POOLING and \
                layer.kernel_x == layer.kernel_y == layer.stride_x == layer.stride_y == 1:
            self.layers.remove(layer)
            return False
        return True

    def compute_layer_dimensions(self, from_position):
        for i in range(from_position, len(self.layers)):
            layer = self.layers[i]
            if layer.connection_type == ConnectionType.CONVOLUTIONAL or layer.connection_type == ConnectionType.POOLING:
                # Compute layer dimensions
                input_x = self.layers[i - 1].neurons_x
                input_y = self.layers[i - 1].neurons_y
                # Layer dimensions also depend on padding
                if layer.padding == 'SAME':
                    layer.neurons_x = int(math.ceil(input_x / layer.stride_x))
                    layer.neurons_y = int(math.ceil(input_y / layer.stride_y))
                elif layer.padding == 'VALID':
                    layer.neurons_x = (input_x - layer.kernel_x) // layer.stride_x + 1
                    layer.neurons_y = (input_y - layer.kernel_y) // layer.stride_y + 1

    def calculate_net_params(self):
        net_params = 0
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            previous_layer = self.layers[i - 1]
            depth_multiplier = 1
            # Check if previous layer contains multiple feature maps
            if previous_layer.connection_type == ConnectionType.CONVOLUTIONAL:
                depth_multiplier = previous_layer.filters
            # Depth of pooling layer depends on preceding layer(s)
            elif previous_layer.connection_type == ConnectionType.POOLING:
                # Find out depth of pooling layer
                further_layer = self.layers[i - 2]
                offset = 3
                # Find first preceding non-pooling layer
                while further_layer.connection_type == ConnectionType.POOLING:
                    further_layer = self.layers[i - offset]
                    offset += 1
                if further_layer.connection_type == ConnectionType.CONVOLUTIONAL:
                    depth_multiplier = further_layer.filters
            # Dense and convolutional layers add to net's parameters
            if layer.connection_type == ConnectionType.CONVOLUTIONAL:
                # Calculate number of connection weights
                params = layer.kernel_x * layer.kernel_y * layer.filters * depth_multiplier
                # Add biases
                params += layer.filters
                net_params += params
            if layer.connection_type == ConnectionType.DENSE:
                # Calculate number of neurons in previous layer
                num_neurons = previous_layer.neurons_x * previous_layer.neurons_y * depth_multiplier
                # Calculate sum of weights and biases
                params = num_neurons * layer.neurons + layer.neurons
                net_params += params
        return net_params
