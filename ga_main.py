"""
This module serves as the genetic algorithm's main component, implementing initialization, selection, population update,
and logging capabilities.
"""


import numpy as np
import ga_config as cf
import ga_genotype as gt
import ga_genops as go
import random
import ga_estimator as est
import copy
import sys
import logging
import json
import pickle
import signal


# Termination request can be submitted via SIGINT
termination_requested = False


def initialize_population(net_scores):
    # Get logger
    logger = logging.getLogger(__name__)
    # Create initial net with no hidden layers
    nets = []
    logger.info("Creating initial net...")
    input_layer = gt.ArtNeurLayer(connection_type=gt.ConnectionType.INPUT,
                                  neurons_x=cf.input_layer_x, neurons_y=cf.input_layer_y)
    output_layer = gt.ArtNeurLayer(connection_type=gt.ConnectionType.DENSE,
                                   neurons_x=cf.output_layer_x, neurons_y=cf.output_layer_y)
    new_net = gt.ArtNeurNet("init", input_layer, output_layer)
    # Evaluate initial net
    new_net.score, new_net.free_parameters, best_score, best_params = evaluate_net(new_net, 0, net_scores, 0.0,
                                                                                   sys.maxsize)
    # Add initial net to population
    nets.append(new_net)
    logger.info("Initial net created.")
    return nets, best_score, best_params


def evolve_net(net, nets, iteration):
    logger = logging.getLogger(__name__)
    logger.debug("Evolving net...")
    # Type of evolution is determined by probability distribution given in configuration file
    probability_distribution = [cf.prob_insert_layer, cf.prob_delete_layer, cf.prob_switch_layers, cf.prob_modify_layer,
                                cf.prob_crossover]
    cases = list(x for x in range(5))
    evolved = False
    # Ensure that net evolves
    while not evolved:
        case = np.random.choice(a=cases, size=1, p=probability_distribution).tolist().pop()
        if case == 0:
            if go.insert_layer(net):
                logger.info("Layer inserted.")
                evolved = True
        elif case == 1:
            if go.delete_layer(net):
                evolved = True
                logger.info("Layer deleted.")
        elif case == 2:
            if go.switch_layers(net):
                evolved = True
                logger.info("Layers switched.")
        elif case == 3:
            if go.modify_layer(net, iteration):
                evolved = True
                logger.info("Layer modified.")
        elif case == 4:
            second_net = copy.deepcopy(select_net(nets))
            if go.crossover(net, second_net):
                evolved = True
                logger.info("Crossover applied.")


def select_net(nets):
    logger = logging.getLogger(__name__)
    logger.debug("Selecting net...")
    # Selection mode depends on configuration
    if cf.selection_mode == cf.SelectionMode.TOURNAMENT_DUEL:
        return select_net_tournament_duel(nets)
    elif cf.selection_mode == cf.SelectionMode.TOURNAMENT_MANY:
        return select_net_tournament_many(nets)
    elif cf.selection_mode == cf.SelectionMode.ROULETTE:
        return select_net_roulette_wheel(nets)


def select_net_roulette_wheel(nets):
    # Roulette Wheel Selection
    logger = logging.getLogger(__name__)
    logger.debug("Selecting net (Roulette) ...")
    # Determine whether selection will be based on score or number of free parameters
    # That depends on whether all the nets' scores exceed the selection threshold
    worst_score = 1.0
    for net in nets:
        if net.score < worst_score:
            worst_score = net.score
    if worst_score < cf.fitness_threshold_selection:
        # Roulette Wheel Selection based on score values
        logger.debug("%s < %s -> Score-based selection...", worst_score, cf.fitness_threshold_selection)
        # Determine sum of fitness values
        sum_of_scores = 0.0
        for net in nets:
            sum_of_scores += net.score
        # Create probability distribution for selection process
        probability_distribution = []
        for i in range(len(nets)):
            probability_distribution.append(nets[i].score / sum_of_scores)
        # Select net based on probability distribution
        selected_net = np.random.choice(a=nets, size=1, p=probability_distribution).tolist().pop()
        return selected_net
    else:
        # Roulette Wheel Selection based on number of free parameters
        logger.debug("%s >= %s -> Parameter-based selection...", worst_score, cf.fitness_threshold_selection)
        # Determine sum of free parameter values
        sum_of_free_parameters = 0
        for net in nets:
            sum_of_free_parameters += net.free_parameters
        # Create probability distribution for selection process
        probability_distribution = []
        for i in range(len(nets)):
            probability_distribution.append(nets[i].free_parameters / sum_of_free_parameters)
        # Select net based on probability distribution
        selected_net = np.random.choice(a=nets, size=1, p=probability_distribution).tolist().pop()
        return selected_net


def select_net_tournament_many(nets):
    # Tournament with many contestants
    logger = logging.getLogger(__name__)
    logger.debug("Selecting net (Tournament) ...")
    # Randomly select contestants
    contestants = np.random.choice(a=nets, size=min(cf.number_of_contestants, len(nets)), replace=False)
    # If population contains one net only, return net
    if len(contestants) == 1:
        logger.debug("Only one net in population, returning net...")
        return contestants[0]
    contestants = contestants.tolist()
    logger.debug("%s contestants encountered.", len(contestants))
    # Examine contestants, collect contestants exceeding fitness threshold and single best net regarding score
    fit_contestants = []
    best_net = None
    highest_score = 0.0
    for contestant in contestants:
        if contestant.score > highest_score:
            highest_score = contestant.score
            best_net = contestant
            logger.debug("Higher score encountered: %s", highest_score)
        else:
            logger.debug("Lower score encountered: %s", contestant.score)
        if contestant.score > cf.fitness_threshold_selection:
            logger.debug("Score above threshold, adding net to fit contestants...")
            fit_contestants.append(contestant)
        else:
            logger.debug("Score below threshold.")
    logger.debug("Best net score: %s", highest_score)
    # Return single best net
    if len(fit_contestants) < 2:
        return best_net
    # Otherwise, selection is based on number of free parameters
    least_free_parameters = sys.maxsize
    logger.debug("Selection based on number of free parameters...")
    best_net = None
    for fit_contestant in fit_contestants:
        if fit_contestant.free_parameters < least_free_parameters:
            least_free_parameters = fit_contestant.free_parameters
            best_net = fit_contestant
    return best_net


def select_net_tournament_duel(nets):
    # Duel Tournament selection
    logger = logging.getLogger(__name__)
    logger.debug("Selecting net (Duel) ...")
    # If population contains one net only, return net
    if len(nets) == 1:
        logger.debug("Only one net in population, returning net...")
        return nets[0]
    # Randomly select 2 nets from population
    net_1 = random.choice(nets)
    net_2 = random.choice(nets)
    # Ensure that there are two different nets selected
    while net_1 is net_2:
        net_2 = random.choice(nets)
    logger.debug("Net 1 score: %s", net_1.score)
    logger.debug("Net 2 score: %s", net_2.score)
    # In case both scores are above selection threshold, selection is based on number of free parameters
    if net_1.score > cf.fitness_threshold_selection and net_2.score > cf.fitness_threshold_selection:
        if net_1.free_parameters < net_2.free_parameters:
            better_net = net_1
            worse_net = net_2
            logger.debug("Net 1 has less free parameters.")
        else:
            better_net = net_2
            worse_net = net_1
            logger.debug("Net 2 has less free parameters.")
    # If not, determine better net via score
    else:
        if net_1.score > net_2.score:
            better_net = net_1
            worse_net = net_2
            logger.debug("Net 1 has higher score.")
        else:
            better_net = net_2
            worse_net = net_1
            logger.debug("Net 2 has higher score.")
    # Select better net with probability chosen in configuration file
    if random.random() < cf.winning_probability:
        logger.debug("Better net selected.")
        return better_net
    else:
        logger.debug("Worse net selected.")
        return worse_net


def evaluate_net(net, iteration, net_scores, best_current_score, best_current_params):
    # Get logger
    logger = logging.getLogger(__name__)
    # Retrieve ID string and store in net object
    net_id = get_net_id_string(net)
    logger.debug("Evaluating net %s...", net_id)
    net.id_string = net_id
    # Determine whether evaluation shall be carried out
    old_score = 0.0
    old_params = 0
    times_evaluated = 0
    to_be_evaluated = True
    # Look up ID
    if net_id in net_scores:
        old_score, old_params, times_evaluated = net_scores[net_id]
        logger.debug("Net has been trained before. Times trained: %s", times_evaluated)
        if old_score == 1.0:
            to_be_evaluated = False
        elif old_score >= cf.eval_threshold and times_evaluated >= cf.max_evals_above_threshold:
            to_be_evaluated = False
        elif old_score < cf.eval_threshold and times_evaluated >= cf.max_evals_below_threshold:
            to_be_evaluated = False
    else:
        logger.debug("Net has not been trained before.")
    # Initiate evaluation if necessary
    if to_be_evaluated:
        logger.info("Commencing training and evaluation...")
        # Train and evaluate net with custom estimator
        fitness, free_params = est.eval_net(net, iteration, best_current_score, best_current_params)
        # Check whether best values are to be updated
        best_score = best_current_score
        best_params = best_current_params
        if fitness > best_current_score or (fitness == best_current_score and free_params < best_current_params):
            best_score = fitness
            best_params = free_params
        # Update values in net dictionary
        if fitness > old_score:
            logger.debug("Fitness improved.")
            net.iteration_evaluated = iteration
            net_scores[net_id] = (fitness, free_params, times_evaluated + 1)
        else:
            logger.debug("Fitness not improved.")
            net_scores[net_id] = (old_score, old_params, times_evaluated + 1)
        logger.debug("Evaluation terminated.")
        return fitness, free_params, best_score, best_params
    logger.info("No further evaluation carried out.")
    return old_score, old_params, best_current_score, best_current_params


def update_population(nets, new_net):
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("Updating population...")
    # If the population contains a net with the same architecture, update fitness value
    old_net = None
    for net in nets:
        if net.id_string == new_net.id_string:
            old_net = net
    if old_net is not None:
        logger.debug("Architecture found in population.")
        if new_net.score > old_net.score:
            logger.debug("Better result from recent training, updating...")
            nets.remove(old_net)
            nets.append(new_net)
    elif len(nets) < cf.max_population:
        # If population is not at maximum and architecture is not contained in current population, add net
        logger.debug("Population not at maximum, adding net...")
        nets.append(new_net)
    # In any other case, compare net to nets in current population
    else:
        # Evaluate nets by score first
        logger.debug("Score new net: %s", new_net.score)
        lowest_score = new_net.score
        worst_nets = [new_net]
        # Find net(s) with lowest score
        for net in nets:
            if net.score < lowest_score:
                lowest_score = net.score
                worst_nets.clear()
                worst_nets.append(net)
            elif net.score == lowest_score:
                worst_nets.append(net)
        logger.debug("Worst net score: %s", lowest_score)
        # If lowest score exceeds threshold, update solely based on number of free parameters
        if lowest_score > cf.fitness_threshold_pop_update:
            logger.debug("Update will be based on number of free parameters entirely...")
            # Find (one of the) net(s) with most free parameters
            most_free_parameters = 0
            worst_net = None
            for net in nets:
                if net.free_parameters > most_free_parameters:
                    most_free_parameters = net.free_parameters
                    worst_net = net
        # If multiple nets have equal score (but below threshold), find (one of the) net(s) with most free parameters
        elif len(worst_nets) > 1:
            logger.debug("Choosing between multiple nets with lowest score...")
            most_free_parameters = 0
            worst_net = None
            for net in worst_nets:
                if net.free_parameters > most_free_parameters:
                    most_free_parameters = net.free_parameters
                    worst_net = net
        # If there is a single net with lowest score below threshold, eliminate net from population
        else:
            logger.debug("Single net with lowest score below threshold identified.")
            worst_net = worst_nets.pop()
        # Worst net is removed and new net added
        if worst_net is not new_net:
            logger.debug("Adding new net...")
            nets.remove(worst_net)
            nets.append(new_net)
        else:
            logger.debug("New net discarded.")


def get_net_id_string(net):
    # Generate ID string to keep track of architecture evaluations
    # The ID string is not unique to a net but to a net's architecture, i.e. two different nets with equal architecture
    # share the same ID string
    id_string = ""
    for i in range(1, len(net.layers)):
        layer = net.layers[i]
        if layer.connection_type == gt.ConnectionType.DENSE:
            id_string += "D" + "_" + str(layer.neurons) + "_" + layer.activation_fn
        elif layer.connection_type == gt.ConnectionType.CONVOLUTIONAL:
            id_string += "C" + "_" + str(layer.kernel_x) + "_" + str(layer.kernel_y) + "_" + str(layer.stride_x) + "_"\
                         + str(layer.stride_y) + "_" + str(layer.padding) + "_" + str(layer.filters) + "_"\
                         + layer.activation_fn
        elif layer.connection_type == gt.ConnectionType.POOLING:
            id_string += "P" + "_" + str(layer.kernel_x) + "_" + str(layer.kernel_y) + "_" + str(layer.stride_x) + "_" \
                         + str(layer.stride_y) + "_" + str(layer.padding) + "_" + str(layer.pooling)
        if i != len(net.layers) - 1:
            id_string += "__"
    return id_string


def retrieve_best_nets(nets):
    # Searches for the best nets in the population, to be presented after termination of the genetic algorithm
    # Returns a list of nets with highest score and equally low number of free parameters and a list of nets with the
    # lowest number of parameters regardless of score
    # Find nets with highest score first
    highest_score = 0.0
    top_nets = []
    for net in nets:
        if net.score > highest_score:
            highest_score = net.score
            top_nets.clear()
            top_nets.append(net)
        elif net.score == highest_score:
            top_nets.append(net)
    # Find nets with highest score and as few free parameters as possible
    least_free_parameters = sys.maxsize
    best_nets = []
    for net in top_nets:
        if net.free_parameters < least_free_parameters:
            least_free_parameters = net.free_parameters
            best_nets.clear()
            best_nets.append(net)
        elif net.free_parameters == least_free_parameters:
            best_nets.append(net)
    least_free_parameters = sys.maxsize
    smallest_nets = []
    for net in nets:
        if net.free_parameters < least_free_parameters:
            least_free_parameters = net.free_parameters
            smallest_nets.clear()
            smallest_nets.append(net)
        elif net.free_parameters == least_free_parameters:
            smallest_nets.append(net)
    return best_nets, smallest_nets


def print_net(net, logger):
    # Print information about net to log file
    logger.info("--------------------------------")
    logger.info("Layers: %s", len(net.layers))
    for i in range(0, len(net.layers)):
        layer = net.layers[i]
        logger.info("Layer %s: Type: %s", i, layer.connection_type)
        if layer.connection_type == gt.ConnectionType.INPUT:
            logger.info("Neurons: X: %s, Y: %s", layer.neurons_x, layer.neurons_y)
        elif layer.connection_type == gt.ConnectionType.DENSE:
            logger.info("act fn: %s, neurons: %s", layer.activation_fn, layer.neurons)
        elif layer.connection_type == gt.ConnectionType.CONVOLUTIONAL \
                or layer.connection_type == gt.ConnectionType.POOLING:
            logger.info("Neurons: X: %s, Y: %s; Kernel: %s, %s; Stride: %s, %s; Padding: %s", layer.neurons_x,
                        layer.neurons_y, layer.kernel_x, layer.kernel_y, layer.stride_x, layer.stride_y, layer.padding)
            if layer.connection_type == gt.ConnectionType.CONVOLUTIONAL:
                logger.info("Filters: %s, act fn: %s", layer.filters, layer.activation_fn)
            if layer.connection_type == gt.ConnectionType.POOLING:
                logger.info("Pooling mode: %s", layer.pooling)
    logger.info("Score: %s, Free parameters: %s, Iteration: %s", net.score, net.free_parameters,
                net.iteration_evaluated)
    logger.info("Net parameters: %s", net.net_parameters)
    logger.info("--------------------------------")


def print_nets(nets, logger):
    # Print information about nets to log file
    for i in range(len(nets)):
        logger.info("Net: %s", i)
        print_net(nets[i], logger)


def do_evolution_step(nets, iteration, net_scores, best_current_score, best_current_params):
    # Perform an iteration of the genetic algorithm
    logger = logging.getLogger(__name__)
    logger.info("Commencing iteration %s...", iteration)
    print("Commencing iteration", iteration)
    # Select net and create copy for evolution
    net = copy.deepcopy(select_net(nets))
    logger.info("Net selected: %s", get_net_id_string(net))
    # Perform evolution
    evolve_net(net, nets, iteration)
    logger.info("Evolved net: %s", get_net_id_string(net))
    best_score = best_current_score
    best_params = best_current_params
    if cf.max_num_params is None or net.net_parameters < cf.max_num_params:
        # Net does not exceed maximum number of parameters specified in configuration file, initiate evaluation
        net.score, net.free_parameters, best_score, best_params = evaluate_net(net, iteration, net_scores,
                                                                               best_current_score, best_current_params)
        logger.info("Score: %s, Free parameters: %s, Iteration: %s", net.score, net.free_parameters,
                    net.iteration_evaluated)
        logger.info("Net parameters: %s", net.net_parameters)
        # Perform population update
        update_population(nets, net)
    else:
        logger.info("Number of free parameters too large! Net discarded.")
    logger.info("--------------------------------")
    logger.debug("Iteration %s terminated.", iteration)
    return best_score, best_params


def create_checkpoint(best_current_params, best_current_score, current_iterations, ga_info, net_scores, nets,
                      previous_iterations, to_be_terminated):
    logger = logging.getLogger(__name__)
    logger.info("Creating checkpoint...")
    # Store important information
    ga_info['previous_iterations'] = previous_iterations + current_iterations
    ga_info['best_score'] = best_current_score
    ga_info['best_params'] = best_current_params
    # Log information and save current state of algorithm
    save_algorithm_state(nets, net_scores, ga_info, to_be_terminated)


def save_algorithm_state(nets, net_scores, ga_info, to_be_terminated):
    # Before termination, the results of the genetic algorithm's run must be logged and the current state preserved
    logger = logging.getLogger(__name__)
    # Log information
    logger.info("Different architectures evaluated: %s", len(net_scores))
    # Compute number of total evaluations
    evaluations = 0
    for k, v in net_scores.items():
        logger.info("%s: %s", k, v)
        fitness, free_params, times_evaluated = v
        evaluations += times_evaluated
    logger.info("Total fitness evaluations: %s", evaluations)
    # Include current population only if termination imminent
    if to_be_terminated:
        logger.info("Population at termination:")
        print_nets(nets, logger)
    # Present best nets and smallest nets in current population
    best_nets, smallest_nets = retrieve_best_nets(nets)
    logger.info("Best net(s):")
    for net in best_nets:
        print_net(net, logger)
    logger.info("Net(s) with least number of free parameters:")
    for net in smallest_nets:
        print_net(net, logger)
    # Prepare additional information to be saved
    ga_info['num_architectures'] = len(net_scores)
    ga_info['num_evaluations'] = evaluations
    logger.info("Saving current state of algorithm...")
    # Save net dictionary to file
    try:
        with open(cf.net_dict_file, 'w') as ndfp:
            json.dump(net_scores, ndfp)
        # Save further information to file
        with open(cf.info_file, 'w') as ifp:
            json.dump(ga_info, ifp)
        # Save population to file
        with open(cf.population_file, 'wb') as pfp:
            pickle.dump(nets, pfp)
    except IOError:
        logger.error("Files could not be saved!")
    logger.info("Done.")


def handle_sigint(signal_number, stack_frame):
    # On SIGINT, algorithm shall terminate in a controlled fashion
    # Termination request is stored to be handled at termination of the current iteration
    global termination_requested
    termination_requested = True


def main():
    # Register handler for SIGINT
    signal.signal(signal.SIGINT, handle_sigint)

    # Provide logging capabilities
    logger = logging.getLogger(__name__)
    logger.setLevel(cf.ga_logging_level)

    # Append to previous log file if continuation requested
    if cf.continue_from_last_run:
        write_mode = 'a+'
    else:
        write_mode = 'w'

    fh = logging.FileHandler(filename=cf.ga_logfile, mode=write_mode)
    fh.setLevel(cf.ga_logging_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False

    # Provide file for TensorFlow log
    tflogger = logging.getLogger('tensorflow')
    tflogger.setLevel(cf.tf_logging_level)
    tffh = logging.FileHandler(filename=cf.tf_logfile, mode=write_mode)
    tffh.setLevel(cf.tf_logging_level)
    tflogger.addHandler(tffh)
    tflogger.propagate = False

    # Prevent tensorflow logger from logging to console
    for handler in tflogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            tflogger.removeHandler(handler)

    logger.info("Input dimensions: %s, %s", cf.input_layer_x, cf.input_layer_y)
    logger.info("Output dimensions: %s, %s", cf.output_layer_x, cf.output_layer_y)

    # If applicable, continue from previous run of genetic algorithm
    previous_iterations = 0
    # Best score achieved so far, to be sent to the estimator in order to decide on whether to store the current model
    # Number of free parameters of net which achieved best_current_score
    if cf.continue_from_last_run:
        # Restore state from prior run
        logger.info("Continuing from last run, loading net dictionary and population...")
        try:
            with open(cf.net_dict_file, 'r') as ndfp:
                net_scores = json.load(ndfp)
            with open(cf.info_file, 'r') as ifp:
                ga_info = json.load(ifp)
            with open(cf.population_file, 'rb') as pfp:
                nets = pickle.load(pfp)
            previous_iterations = ga_info['previous_iterations']
            best_current_score = ga_info['best_score']
            best_current_params = ga_info['best_params']
        except IOError:
            logger.error("Saved files could not be opened, running genetic algorithm from scratch!")
            net_scores = {}
            ga_info = {}
            nets, best_current_score, best_current_params = initialize_population(net_scores)
    else:
        logger.info("Running from scratch, no nets imported...")
        # Create data structures, keep track of all nets trained so far via net score dictionary
        net_scores = {}
        ga_info = {}
        nets, best_current_score, best_current_params = initialize_population(net_scores)

    # Execute genetic algorithm, either until termination is requested via SIGINT or for a fix number of iterations
    current_iterations = 0
    if cf.num_iterations is None:
        while True:
            best_current_score, best_current_params = do_evolution_step(nets,
                                                                        current_iterations + 1 + previous_iterations,
                                                                        net_scores,
                                                                        best_current_score,
                                                                        best_current_params)
            current_iterations += 1
            if current_iterations % cf.checkpoint_after_iterations == 0 and current_iterations != cf.num_iterations:
                create_checkpoint(best_current_params, best_current_score, current_iterations, ga_info, net_scores,
                                  nets, previous_iterations, False)
            if termination_requested:
                logger.info("Termination requested...")
                break
    else:
        for _ in range(cf.num_iterations - previous_iterations):
            best_current_score, best_current_params = do_evolution_step(nets,
                                                                        current_iterations + 1 + previous_iterations,
                                                                        net_scores,
                                                                        best_current_score,
                                                                        best_current_params)
            current_iterations += 1
            if current_iterations % cf.checkpoint_after_iterations == 0 and current_iterations != cf.num_iterations:
                create_checkpoint(best_current_params, best_current_score, current_iterations, ga_info, net_scores,
                                  nets, previous_iterations, False)
            if termination_requested:
                logger.info("Termination requested...")
                break

    # Log information and save current state of algorithm
    create_checkpoint(best_current_params, best_current_score, current_iterations, ga_info, net_scores,
                      nets, previous_iterations, True)


if __name__ == "__main__":
    main()
