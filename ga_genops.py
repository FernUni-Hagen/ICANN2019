"""
This module contains the functions implementing the genetic algorithm's genetic operators.
"""


import ga_genotype as gt
import ga_config as cf
import random
import numpy as np
import math


def insert_layer(net):
    layer_inserted = False
    num_layers = len(net.layers)
    # Check if maximum number of layers has been reached
    if num_layers < cf.max_layers:
        # Create layer
        # Randomly select connection type (no second input layer possible)
        connection_type = gt.ConnectionType(random.randint(2, 4))
        layer = None
        minpos = 1
        maxpos = num_layers - 1
        # Check whether architecture restrictions apply
        if cf.force_canonical_architecture:
            # Determine position of first fully connected layer
            pos_first_dense = 0
            for i in range(1, len(net.layers)):
                if net.layers[i].connection_type == gt.ConnectionType.DENSE:
                    pos_first_dense = i
                    break
            # Set range of feasible positions accordingly
            if connection_type == gt.ConnectionType.DENSE:
                minpos = pos_first_dense
            else:
                maxpos = pos_first_dense
        # Determine new layer's position
        position = random.randint(minpos, maxpos)
        # Create new layer
        if connection_type == gt.ConnectionType.DENSE:
            neurons = random.randint(1, cf.dense_max_units)
            layer = gt.ArtNeurLayer(connection_type=connection_type, neurons_x=neurons, neurons_y=1)
        elif connection_type == gt.ConnectionType.CONVOLUTIONAL:
            # Kernel size should not exceed previous layer's size
            kernel_x_upper_limit = min(cf.max_kernel_x, net.layers[position - 1].neurons_x)
            kernel_x = random.randint(1, kernel_x_upper_limit)
            kernel_y_upper_limit = min(cf.max_kernel_y, net.layers[position - 1].neurons_y)
            kernel_y = random.randint(1, kernel_y_upper_limit)
            stride_x = random.randint(1, cf.max_stride_x)
            stride_y = random.randint(1, cf.max_stride_y)
            filters = random.randint(1, cf.max_filters)
            if cf.default_padding == 'RANDOM':
                if random.random() < 0.5:
                    padding = 'VALID'
                else:
                    padding = 'SAME'
            else:
                padding = cf.default_padding
            layer = gt.ArtNeurLayer(connection_type=connection_type, kernel_x=kernel_x, kernel_y=kernel_y,
                                    stride_x=stride_x, stride_y=stride_y, filters=filters, padding=padding)
        elif connection_type == gt.ConnectionType.POOLING:
            # Kernel size should not exceed previous layer's size
            kernel_x_upper_limit = min(cf.max_kernel_x, net.layers[position - 1].neurons_x)
            kernel_x = random.randint(1, kernel_x_upper_limit)
            kernel_y_upper_limit = min(cf.max_kernel_y, net.layers[position - 1].neurons_y)
            kernel_y = random.randint(1, kernel_y_upper_limit)
            stride_x = random.randint(1, cf.max_stride_x)
            stride_y = random.randint(1, cf.max_stride_y)
            if cf.default_padding == 'RANDOM':
                if random.random() < 0.5:
                    padding = 'VALID'
                else:
                    padding = 'SAME'
            else:
                padding = cf.default_padding
            if cf.default_pooling == 'RANDOM':
                if random.random() < 0.5:
                    pooling = 'MAX'
                else:
                    pooling = 'AVERAGE'
            else:
                pooling = cf.default_pooling
            layer = gt.ArtNeurLayer(connection_type=connection_type, kernel_x=kernel_x, kernel_y=kernel_y,
                                    stride_x=stride_x, stride_y=stride_y, padding=padding, pooling=pooling)
        # Insert layer
        layer_inserted = net.insert_layer(layer, position)
    return layer_inserted


def delete_layer(net):
    # Check if there are hidden layers
    if len(net.layers) > 2:
        # Randomly select layer to be deleted
        position = random.randint(1, len(net.layers) - 1)
        return net.delete_layer_by_position(position)
    return False


def switch_layers(net):
    # Randomly select two layers and switch positions
    number_of_layers = len(net.layers)
    # Only hidden layers can be switched, so there must be at least 4 layers in total
    if number_of_layers >= 4:
        minpos = 1
        maxpos = number_of_layers - 1
        # Check whether architecture restrictions apply
        if cf.force_canonical_architecture:
            # Determine position of first fully connected layer
            pos_first_dense = 0
            for i in range(1, len(net.layers)):
                if net.layers[i].connection_type == gt.ConnectionType.DENSE:
                    pos_first_dense = i
                    break
            # Determine in which part of the net the switch will occur
            conv_pool_possible = False
            dense_possible = False
            # There have to be at least two layers of respective type for a switch to be possible
            if pos_first_dense >= 3:
                conv_pool_possible = True
            if pos_first_dense <= number_of_layers - 3:
                dense_possible = True
            if conv_pool_possible and dense_possible:
                # Both options possible, randomly choose one
                if random.random() < 0.5:
                    minpos = pos_first_dense
                else:
                    maxpos = pos_first_dense - 1
            elif conv_pool_possible:
                # Switch has to occur between convolutional and/or pooling layers
                maxpos = pos_first_dense - 1
            elif dense_possible:
                # Switch has to occur between fully connected layers
                minpos = pos_first_dense
            else:
                # No switch possible
                return False
        # Determine positions for layer switch
        position_1 = random.randint(minpos, maxpos)
        position_2 = random.randint(minpos, maxpos)
        while position_1 == position_2:
            position_2 = random.randint(minpos, maxpos)
        return net.switch_layers_by_position(position_1, position_2)
    return False


def modify_layer(net, iteration):
    # Only hidden layers can be modified
    if len(net.layers) > 2:
        # Randomly select layer for modification
        position = random.randint(1, len(net.layers) - 1)
        layer = net.layers[position]
        mod_success = False
        # Possible modifications depend on connection type
        if layer.connection_type == gt.ConnectionType.DENSE:
            # For dense layers, the number of neurons and the activation function can be modified
            # Determine what modification is to be applied based on probability distribution given in configuration file
            probability_distribution = [cf.prob_mod_dense_neurons, cf.prob_mod_dense_actfn]
            cases = list(x for x in range(2))
            case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
            if case == 0:
                mod_success = modify_neurons(net, position, layer, iteration)
            elif case == 1:
                mod_success = modify_act_fn(net, position, layer)
        elif layer.connection_type == gt.ConnectionType.CONVOLUTIONAL:
            # Possible modifications for convolutional layers are:
            # kernel size, stride, number of filters, padding type, activation function
            # Determine what modification is to be applied based on probability distribution given in configuration file
            probability_distribution = [cf.prob_mod_conv_kernel, cf.prob_mod_conv_stride, cf.prob_mod_conv_filters,
                                        cf.prob_mod_conv_padding, cf.prob_mod_conv_actfn]
            cases = list(x for x in range(5))
            case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
            if case == 0:
                mod_success = modify_kernel(net, position, layer, iteration)
            elif case == 1:
                mod_success = modify_stride(net, position, layer, iteration)
            elif case == 2:
                mod_success = modify_filters(net, position, layer, iteration)
            elif case == 3:
                mod_success = modify_padding(net, position, layer)
            elif case == 4:
                mod_success = modify_act_fn(net, position, layer)
        elif layer.connection_type == gt.ConnectionType.POOLING:
            # Possible modifications for convolutional layers are: kernel size, stride, number of filters, padding type
            # Determine what modification is to be applied based on probability distribution given in configuration file
            probability_distribution = [cf.prob_mod_pool_kernel, cf.prob_mod_pool_stride, cf.prob_mod_pool_pooling,
                                        cf.prob_mod_pool_padding]
            cases = list(x for x in range(4))
            case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
            if case == 0:
                mod_success = modify_kernel(net, position, layer, iteration)
            elif case == 1:
                mod_success = modify_stride(net, position, layer, iteration)
            elif case == 2:
                mod_success = modify_pooling(net, position, layer)
            elif case == 3:
                mod_success = modify_padding(net, position, layer)
        return mod_success
    return False


def modify_neurons(net, position, layer, iteration):
    mod_success = False
    # Determine step size
    # Step size decreases exponentially during the course of the genetic algorithm according to parameters given in
    # configuration file
    step_size = get_step_size(cf.dense_layer_step, cf.dense_layer_step_decay, iteration)
    # Randomly augment or shrink layer
    if random.random() < cf.dense_layer_incr_prob:
        # Check if new layer size is permitted
        if layer.neurons + step_size <= cf.dense_max_units:
            # Increment number of neurons
            mod_success = net.set_neurons_by_position(position, layer.neurons + step_size)
    else:
        if layer.neurons - step_size > 0:
            # Decrement number of neurons
            mod_success = net.set_neurons_by_position(position, layer.neurons - step_size)
    return mod_success


def modify_kernel(net, position, layer, iteration):
    # Kernel can be modified in X or Y dimension or both
    # Determine what modification is to be applied based on probability distribution given in configuration file
    mod_success = False
    # Determine step size
    # Step size decreases exponentially during the course of the genetic algorithm according to parameters given in
    # configuration file
    step_size_x = get_step_size(cf.kernel_mod_step_x, cf.kernel_mod_step_x_decay, iteration)
    step_size_y = get_step_size(cf.kernel_mod_step_y, cf.kernel_mod_step_y_decay, iteration)
    probability_distribution = [cf.prob_mod_kernel_x, cf.prob_mod_kernel_y, cf.prob_mod_kernel_both]
    cases = list(x for x in range(3))
    case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
    if case == 0:
        # Modify X dimension only
        # Randomly augment or shrink kernel in X dimension
        if random.random() < cf.kernel_mod_incr_prob:
            # Check if new kernel size is permitted
            if layer.kernel_x + step_size_x <= cf.max_kernel_x:
                # Increment kernel size
                mod_success = net.set_kernel_x_by_position(position, layer.kernel_x + step_size_x)
        else:
            if layer.kernel_x - step_size_x > 0:
                # Decrement kernel size
                mod_success = net.set_kernel_x_by_position(position, layer.kernel_x - step_size_x)
    elif case == 1:
        # Modify Y dimension only
        # Randomly augment or shrink kernel in Y dimension
        if random.random() < cf.kernel_mod_incr_prob:
            # Check if new kernel size is permitted
            if layer.kernel_y + step_size_y <= cf.max_kernel_y:
                # Increment kernel size
                mod_success = net.set_kernel_y_by_position(position, layer.kernel_y + step_size_y)
        else:
            if layer.kernel_y - step_size_y > 0:
                # Decrement kernel size
                mod_success = net.set_kernel_y_by_position(position, layer.kernel_y - step_size_y)
    elif case == 2:
        # Modify both X and Y dimensions
        # Randomly augment or shrink kernel
        if random.random() < cf.kernel_mod_incr_prob:
            # Increase kernel size
            new_kernel_x = layer.kernel_x + step_size_x
            new_kernel_y = layer.kernel_y + step_size_y
            # Check if new kernel size is permitted
            if new_kernel_x <= cf.max_kernel_x and new_kernel_y <= cf.max_kernel_y:
                # Increment kernel size
                mod_success_x = net.set_kernel_x_by_position(position, new_kernel_x)
                mod_success_y = net.set_kernel_y_by_position(position, new_kernel_y)
                if mod_success_x and mod_success_y:
                    mod_success = True
                else:
                    # Modification could not be applied for both dimensions, abort and revise
                    if mod_success_x:
                        net.set_kernel_x_by_position(position, layer.kernel_x - step_size_x)
                    if mod_success_y:
                        net.set_kernel_y_by_position(position, layer.kernel_y - step_size_y)
                    mod_success = False
        else:
            # Decrease kernel size
            new_kernel_x = layer.kernel_x - step_size_x
            new_kernel_y = layer.kernel_y - step_size_y
            # Check if new kernel size is permitted
            if new_kernel_x > 0 and new_kernel_y > 0:
                # Decrement kernel size
                mod_success_x = net.set_kernel_x_by_position(position, new_kernel_x)
                mod_success_y = net.set_kernel_y_by_position(position, new_kernel_y)
                if mod_success_x and mod_success_y:
                    mod_success = True
                else:
                    # Modification could not be applied for both dimensions, abort and revise
                    if mod_success_x:
                        net.set_kernel_x_by_position(position, layer.kernel_x + step_size_x)
                    if mod_success_y:
                        net.set_kernel_y_by_position(position, layer.kernel_y + step_size_y)
                    mod_success = False
    return mod_success


def modify_stride(net, position, layer, iteration):
    # Stride can be modified in X or Y dimension or both
    # Determine what modification is to be applied based on probability distribution given in configuration file
    mod_success = False
    # Determine step size
    # Step size decreases exponentially during the course of the genetic algorithm according to parameters given in
    # configuration file
    step_size_x = get_step_size(cf.stride_mod_step_x, cf.stride_mod_step_x_decay, iteration)
    step_size_y = get_step_size(cf.stride_mod_step_y, cf.stride_mod_step_y_decay, iteration)
    probability_distribution = [cf.prob_mod_stride_x, cf.prob_mod_stride_y, cf.prob_mod_stride_both]
    cases = list(x for x in range(3))
    case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
    if case == 0:
        # Modify X dimension only
        # Randomly augment or shrink stride in X dimension
        if random.random() < cf.stride_mod_incr_prob:
            # Check if new stride is permitted
            if layer.stride_x + step_size_x <= cf.max_stride_x:
                # Increment stride
                mod_success = net.set_stride_x_by_position(position, layer.stride_x + step_size_x)
        else:
            if layer.stride_x - step_size_x > 0:
                # Decrement stride
                mod_success = net.set_stride_x_by_position(position, layer.stride_x - step_size_x)
    elif case == 1:
        # Modify Y dimension only
        # Randomly augment or shrink stride in Y dimension
        if random.random() < cf.stride_mod_incr_prob:
            # Check if new stride is permitted
            if layer.stride_y + step_size_y <= cf.max_stride_y:
                # Increment stride
                mod_success = net.set_stride_y_by_position(position, layer.stride_y + step_size_y)
        else:
            if layer.stride_y - step_size_y > 0:
                # Decrement stride
                mod_success = net.set_stride_y_by_position(position, layer.stride_y - step_size_y)
    elif case == 2:
        # Modify both X and Y dimensions
        # Randomly augment or shrink stride
        if random.random() < cf.stride_mod_incr_prob:
            # Increase stride
            new_stride_x = layer.stride_x + step_size_x
            new_stride_y = layer.stride_y + step_size_y
            # Check if new stride is permitted
            if new_stride_x <= cf.max_stride_x and new_stride_y <= cf.max_stride_y:
                # Increment stride
                mod_success_x = net.set_stride_x_by_position(position, new_stride_x)
                mod_success_y = net.set_stride_y_by_position(position, new_stride_y)
                if mod_success_x and mod_success_y:
                    mod_success = True
                else:
                    # Modification could not be applied for both dimensions, abort and revise
                    if mod_success_x:
                        net.set_stride_x_by_position(position, layer.stride_x - step_size_x)
                    if mod_success_y:
                        net.set_stride_y_by_position(position, layer.stride_y - step_size_y)
                    mod_success = False
        else:
            # Decrease stride
            new_stride_x = layer.stride_x - step_size_x
            new_stride_y = layer.stride_y - step_size_y
            # Check if new stride is permitted
            if new_stride_x > 0 and new_stride_y > 0:
                # Decrement stride
                mod_success_x = net.set_stride_x_by_position(position, new_stride_x)
                mod_success_y = net.set_stride_y_by_position(position, new_stride_y)
                if mod_success_x and mod_success_y:
                    mod_success = True
                else:
                    # Modification could not be applied for both dimensions, abort and revise
                    if mod_success_x:
                        net.set_stride_x_by_position(position, layer.stride_x + step_size_x)
                    if mod_success_y:
                        net.set_stride_y_by_position(position, layer.stride_y + step_size_y)
                    mod_success = False
    return mod_success


def modify_filters(net, position, layer, iteration):
    mod_success = False
    # Determine step size
    # Step size decreases exponentially during the course of the genetic algorithm according to parameters given in
    # configuration file
    step_size = get_step_size(cf.filter_mod_step, cf.filter_mod_step_decay, iteration)
    # Number of filters can be incremented or decremented
    if random.random() < cf.filter_mod_incr_prob:
        # Check if new number of filters is permitted
        if layer.filters + step_size <= cf.max_filters:
            # Increment number of filters
            mod_success = net.set_filters_by_position(position, layer.filters + step_size)
    else:
        if layer.filters + step_size > 0:
            # Decrement number of filters
            mod_success = net.set_filters_by_position(position, layer.filters - step_size)
    return mod_success


def modify_padding(net, position, layer):
    # Switch padding mode
    if layer.padding == 'VALID':
        net.set_padding_by_position(position, 'SAME')
    else:
        net.set_padding_by_position(position, 'VALID')
    return True


def modify_pooling(net, position, layer):
    # Switch pooling mode
    if layer.pooling == 'MAX':
        net.set_pooling_by_position(position, 'AVERAGE')
    else:
        net.set_pooling_by_position(position, 'MAX')
    return True


def modify_act_fn(net, position, layer):
    # Determine current activation function
    act_fn = layer.activation_fn
    # Randomly choose new activation function
    new_act_fn = random.choice(list(cf.act_fns.keys()))
    while act_fn is new_act_fn:
        new_act_fn = random.choice(list(cf.act_fns.keys()))
    return net.set_activation_function_by_position(position, new_act_fn)


def crossover(parent_one, parent_two):
    # Check whether architecture restrictions apply
    if cf.force_canonical_architecture:
        # Ensure that order of layer types prevails
        # Employ one-point crossover
        # Determine position of first fully connected layer for both parents
        parent_one_pos_first_dense = len(parent_one.layers) - 1
        for i in range(1, len(parent_one.layers)):
            if parent_one.layers[i].connection_type == gt.ConnectionType.DENSE:
                parent_one_pos_first_dense = i
                break
        parent_two_pos_first_dense = len(parent_two.layers)
        for i in range(1, len(parent_two.layers)):
            if parent_two.layers[i].connection_type == gt.ConnectionType.DENSE:
                parent_two_pos_first_dense = i
                break
        # In case parent one contains dense layers only, take input layer from parent one
        # In case parent two contains no dense layer, crossover is not possible
        if parent_two_pos_first_dense > len(parent_two.layers) - 1:
            return False
        cutoff_parent_one = random.randint(1, parent_one_pos_first_dense)
        cutoff_parent_two = random.randint(parent_two_pos_first_dense, len(parent_two.layers) - 1)
        # Ensure new architecture meets depth restriction. This is possible in any constellation
        while cutoff_parent_one + cutoff_parent_two - 1 > cf.max_layers:
            cutoff_parent_one = random.randint(1, parent_one_pos_first_dense)
            cutoff_parent_two = random.randint(parent_two_pos_first_dense, len(parent_two.layers) - 1)
        # Create new architecture
        new_architecture = parent_one.layers[:cutoff_parent_one]
        new_architecture.extend(parent_two.layers[cutoff_parent_two:])
    else:
        # Determine number of crossover points
        min_crossover_points = cf.min_crossover_points
        max_crossover_points = min(cf.max_crossover_points, len(parent_one.layers) - 1, len(parent_two.layers) - 1)
        if max_crossover_points < min_crossover_points:
            return False
        num_crossover_points = random.randint(min_crossover_points, max_crossover_points)
        # Determine which parent has more layers
        if len(parent_one.layers) < len(parent_two.layers):
            num_layers = len(parent_one.layers)
        else:
            num_layers = len(parent_two.layers)
        # Randomly determine crossover points
        crossover_indices = random.sample(range(1, num_layers), num_crossover_points)
        # Create new architecture
        new_architecture = parent_one.layers[:crossover_indices[0]]
        # Alternate layer sequences from both parents
        for i in range(1, num_crossover_points):
            if i % 2 == 0:
                new_architecture.extend(parent_one.layers[crossover_indices[i - 1]:crossover_indices[i]])
            else:
                new_architecture.extend(parent_two.layers[crossover_indices[i - 1]:crossover_indices[i]])
        if num_crossover_points % 2 == 0:
            new_architecture.extend(parent_two.layers[crossover_indices[num_crossover_points - 1]:])
        else:
            new_architecture.extend(parent_one.layers[crossover_indices[num_crossover_points - 1]:])
    # Submit new architecture to net
    return parent_one.change_architecture(new_architecture)


def get_step_size(step, decay, iteration):
    # Step size decreases exponentially during the course of the genetic algorithm
    if cf.num_iterations is not None:
        decay_factor = decay ** (iteration / cf.num_iterations)
    elif cf.decay_over_iterations > 0:
        decay_factor = decay ** (iteration / cf.decay_over_iterations)
    else:
        decay_factor = decay
    return math.ceil(step * decay_factor)
