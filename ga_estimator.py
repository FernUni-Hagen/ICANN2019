"""
This module contains the custom estimator for TensorFlow to be used as evaluation component in the genetic algorithm.

See https://www.tensorflow.org/guide/custom_estimators for the creation of custom estimators.
The code in this file originates from that tutorial and has been heavily modified and extended to serve as an
instrument for the evaluation of nets in the genetic algorithm.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import ga_config as cf
import ga_genotype as gt
import shutil
import os
import pickle
import logging
from tensorflow.python.training.basic_session_run_hooks import NanLossDuringTrainingError


# Records global step at which checkpoint with lowest loss has been encountered
best_checkpoint_at_step = 0
# Record whether input functions have been created yet
input_functions_created = False
train_input_fn = None
validate_input_fn = None
eval_input_fn = None
# Provide logger for error logging
logger = None
# List of layer objects
layer_obs = [None] * cf.max_layers
# Summary writer for logging loss and accuracy on validation set
# summary_writer = None
gl_step = 0


# ValidationListener implements early stopping functionality. To be used as listener for CheckpointSaverHook.
class ValidationListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn, model_dir):
        self._estimator = estimator
        self._input_fn = input_fn
        self._model_dir = model_dir
        self._last_loss = None
        self._best_loss = None
        self._validations_since_best_loss = 0
        self._validations_since_major_improvement = 0

    def after_save(self, session, global_step_value):
        # Evaluate net on validation set
        eval_result = self._estimator.evaluate(input_fn=self._input_fn)
        current_loss = eval_result['loss']
        current_accuracy = eval_result['accuracy']
        summary_writer = tf.summary.FileWriterCache.get(os.path.join(self._model_dir, 'validation'))
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=current_loss)
        summary.value.add(tag='accuracy', simple_value=current_accuracy)
        summary_writer.add_summary(summary, global_step_value)
        # summary_writer.close()
        
        # Update global step value
        global gl_step
        gl_step = global_step_value
        
        # Check "validations since best loss" criterion
        if self._best_loss is None:
            self._best_loss = current_loss
            set_best_checkpoint_at_step(global_step_value)
        elif current_loss > self._best_loss:
            self._validations_since_best_loss += 1
            if self._validations_since_best_loss >= cf.max_validations_since_best_loss:
                # Request termination of training
                return True
        else:
            self._best_loss = current_loss
            set_best_checkpoint_at_step(global_step_value)
            self._validations_since_best_loss = 0

        # Check "validations without major improvement" criterion
        if self._last_loss is None:
            pass
        elif current_loss > self._last_loss - cf.major_improvement_threshold:
            self._validations_since_major_improvement += 1
            if self._validations_since_major_improvement >= cf.max_validations_without_major_improvement:
                # Request termination of training
                return True
        else:
            self._validations_since_major_improvement = 0
        self._last_loss = current_loss

    def reset_loss(self):
        # Reset values prior to next training round
        self._best_loss = None
        self._last_loss = None
        self._validations_since_best_loss = 0
        self._validations_since_major_improvement = 0
        set_best_checkpoint_at_step(0)


def set_best_checkpoint_at_step(global_step_value):
    # Stores global step value for which checkpoint with lowest loss has been created
    # Used to restore model state prior to evaluation
    global best_checkpoint_at_step
    best_checkpoint_at_step = global_step_value


def get_initializer(layer):
    # Provide kernel initializer depending on activation function
    if not cf.use_initializer:
        return None
    elif layer.activation_fn == 'RELU' or layer.activation_fn == 'ELU':
        return tf.contrib.layers.variance_scaling_initializer()
    elif layer.activation_fn == 'SIGMOID':
        return tf.contrib.layers.xavier_initializer()
    else:
        return None


def apply_batch_normalization(layer, training):
    # Provide batch normalization to layer
    return tf.layers.batch_normalization(layer, training=training, momentum=cf.batch_normalization_momentum)


def apply_activation_function(net, layer):
    # Provide activation function for layer
    act_fn = cf.act_fns[layer.activation_fn]
    return act_fn(net)


def apply_dropout(layer, training):
    # Provide dropout to layer
    return tf.layers.dropout(layer, rate=cf.dropout_rate, training=training)


def create_model(features, labels, mode, params):
    # Model function, to be used by estimator
    # Determine if running in training mode
    training = mode == tf.estimator.ModeKeys.TRAIN
    
    # Create summary writer
    # global summary_writer
    # summary_writer = tf.summary.FileWriter(params['train_dir'])
    
    # Create net from layers, according to genotype
    # Retrieve genotype
    network = params['network']
    # Create net
    logits = create_net(features, network, params['n_classes'], training)

    # Compute predictions, again depending on number of classes
    if params['n_classes'] > 2:
        predicted_classes = tf.argmax(logits, 1)
        probabilities = tf.nn.softmax(logits)
    else:
        predicted_classes = tf.round(x=logits)
        probabilities = tf.identity(logits)

    # For prediction mode, specify predictions and return estimator spec. Not used in genetic algorithm.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': probabilities,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss, again depending on number of classes
    if params['n_classes'] > 2:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    else:
        labels = tf.expand_dims(labels, axis=1)
        loss = tf.losses.log_loss(labels=labels, predictions=logits)

    # Compute accuracy
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    # Create variable for free parameters, to be read from function eval_net below
    profiling_options = tf.profiler.ProfileOptionBuilder(
        tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()).with_empty_output().build()
    param_stats = tf.profiler.profile(tf.get_default_graph(), options=profiling_options)
    tf.Variable(param_stats.total_parameters, trainable=False, name='free_params')

    # Prepare metrics
    metrics = {'accuracy': accuracy}
    # Ensure availability of accuracy metric in TensorBoard
    tf.summary.scalar('accuracy', accuracy[1])

    # For evaluation mode, return respective estimator spec
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # For training mode, prepare training op
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Determine whether learning schedule is to be employed
    if not cf.use_learning_rate_scheduling:
        learning_rate = cf.learning_rate
    else:
        # Exponential decay is applied, cf. Geron (2017), p. 306
        initial_learning_rate = cf.initial_learning_rate
        decay_steps = cf.learning_rate_decay_steps
        decay_rate = cf.learning_rate_decay_rate
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    # Create optimizer
    if cf.optimizer == 'MOMENTUM':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cf.momentum_momentum,
                                               use_nesterov=cf.momentum_use_nesterov)
    elif cf.optimizer == 'RMSPROP':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=cf.rmsprop_decay,
                                              momentum=cf.rmsprop_momentum, epsilon=cf.rmsprop_epsilon,
                                              centered=cf.rmsprop_centered)
    elif cf.optimizer == 'ADAGRAD':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                              initial_accumulator_value=cf.adagrad_initial_accumulator_value)
    elif cf.optimizer == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=cf.adam_beta1, beta2=cf.adam_beta2,
                                           epsilon=cf.adam_epsilon)
    else:
        # Standard optimizer is Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # Create train op, for batch normalization add dependency for update ops to train op
    if cf.employ_batch_normalization_conv or cf.employ_batch_normalization_dense:
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Return estimator spec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_net(features, network, n_classes, training):
    # Creates a network of TensorFlow layers according to the architecture stored in the genotype
    # Create input layer
    net = tf.reshape(features["x"], [-1, cf.input_layer_x, cf.input_layer_y, cf.input_channels])
    # Create hidden layers
    for i in range(1, len(network.layers) - 1):
        layer = network.layers[i]
        layer_type = layer.connection_type
        if layer_type == gt.ConnectionType.DENSE:
            # Dense layer with number of neurons specified by neurons attribute of layer
            # Reshape tensor if necessary
            if not network.layers[i - 1].connection_type == gt.ConnectionType.DENSE:
                net = tf.layers.flatten(net)
            # Create layer
            if i < network.pos_last_change and network.weights[i] is not None:
                # Reuse weights
                kernel = network.weights[i][0]
                bias = network.weights[i][1]
                kernel_initializer = tf.constant_initializer(kernel)
                bias_initializer = tf.constant_initializer(bias)
                net = tf.layers.dense(net, units=layer.neurons, activation=None, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name=str(i))
            else:
                # Do not reuse weights but kernel initializer instead
                initializer = get_initializer(layer)
                net = tf.layers.dense(net, units=layer.neurons, activation=None, kernel_initializer=initializer,
                                  name=str(i))
            # Check for application of dropout or batch normalization. For dense layers, dropout takes precedence over
            # batch normalization.
            if cf.employ_dropout_dense:
                net = apply_activation_function(net, layer)
                net = apply_dropout(net, training)
            elif cf.employ_batch_normalization_dense:
                net = apply_batch_normalization(net, training)
                net = apply_activation_function(net, layer)
            else:
                net = apply_activation_function(net, layer)
        elif layer_type == gt.ConnectionType.CONVOLUTIONAL:
            # Convolutional layer
            # Reshape tensor if necessary
            if network.layers[i - 1].connection_type == gt.ConnectionType.DENSE:
                # Dense layer is considered as 1D layer
                net = tf.reshape(net, [-1, network.layers[i - 1].neurons, 1, 1])
            # Create layer
            if i < network.pos_last_change and network.weights[i] is not None:
                # Reuse weights
                kernel = network.weights[i][0]
                bias = network.weights[i][1]
                kernel_initializer = tf.constant_initializer(kernel)
                bias_initializer = tf.constant_initializer(bias)
                net = tf.layers.conv2d(inputs=net, filters=layer.filters, kernel_size=[layer.kernel_x, layer.kernel_y],
                                       padding=layer.padding, activation=None, strides=(layer.stride_x, layer.stride_y),
                                       bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,
                                       name=str(i))
            else:
                # Do not reuse weights but kernel initializer instead
                initializer = get_initializer(layer)
                net = tf.layers.conv2d(inputs=net, filters=layer.filters, kernel_size=[layer.kernel_x, layer.kernel_y],
                                       padding=layer.padding, activation=None, strides=(layer.stride_x, layer.stride_y),
                                       kernel_initializer=initializer, name=str(i))
            # Check for application of dropout or batch normalization. For convolutional layers, batch normalization
            # takes precedence over dropout.
            if cf.employ_batch_normalization_conv:
                net = apply_batch_normalization(net, training)
                net = apply_activation_function(net, layer)
            elif cf.employ_dropout_conv:
                net = apply_activation_function(net, layer)
                net = apply_dropout(net, training)
            else:
                net = apply_activation_function(net, layer)
        elif layer_type == gt.ConnectionType.POOLING:
            # Pooling layer
            # Reshape tensor if necessary
            if network.layers[i - 1].connection_type == gt.ConnectionType.DENSE:
                net = tf.reshape(net, [-1, network.layers[i - 1].neurons, 1, 1])
            # Determine kind of pooling layer and create layer
            if layer.pooling == 'MAX':
                net = tf.layers.max_pooling2d(inputs=net, pool_size=[layer.kernel_x, layer.kernel_y],
                                              padding=layer.padding, strides=(layer.stride_x, layer.stride_y),
                                              name=str(i))
            else:
                # Create average pooling layer
                net = tf.layers.average_pooling2d(inputs=net, pool_size=[layer.kernel_x, layer.kernel_y],
                                                  padding=layer.padding, strides=(layer.stride_x, layer.stride_y),
                                                  name=str(i))
    # In case last hidden layer is input, convolutional or pooling layer, reshape
    if not network.layers[len(network.layers) - 2].connection_type == gt.ConnectionType.DENSE:
        net = tf.layers.flatten(net)
    # Add final layer depending on number of classes
    if n_classes > 2:
        logits = tf.layers.dense(net, n_classes, activation=None, name=str(len(network.layers)-1))
    else:
        logits = tf.layers.dense(inputs=net, units=1, activation=tf.nn.sigmoid, name=str(len(network.layers)-1))
    return logits


def create_input_functions():
    # Retrieve data
    (train_x, train_y), (validate_x, validate_y), (test_x, test_y) = cf.data_retrieval_fn()

    # Create input functions
    global train_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=cf.train_batch_size,
        num_epochs=None,
        shuffle=True
    )
    if cf.employ_early_stopping:
        global validate_input_fn
        validate_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": validate_x},
            y=validate_y,
            num_epochs=1,
            shuffle=False
        )
    global eval_input_fn
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False
    )
    global input_functions_created
    input_functions_created = True


def save_best_model(model_dir, net, step, iteration):
    try:
        # Remove old best model
        if os.path.isdir(cf.best_dir):
            shutil.rmtree(cf.best_dir, ignore_errors=True)
        # Copy current model
        shutil.copytree(model_dir, cf.best_dir)
        # Record best checkpoint
        best_checkpoint_file = open(cf.best_dir + "best_checkpoint.txt", 'w+')
        best_checkpoint_file.write(str(step))
        best_checkpoint_file.close()
        # Record iteration number
        iteration_file = open(cf.best_dir + "iteration.txt", 'w+')
        iteration_file.write(str(iteration))
        iteration_file.close()
        with open(cf.best_dir + "best_net.pickle", 'wb') as nfp:
            pickle.dump(net, nfp)
    except (IOError, OSError, shutil.Error):
        print("Best model could not be saved!")
        logger.error("Best model could not be saved!")


def eval_net(net, iteration, best_current_score, best_current_params):
    # Build directory name
    train_dir = cf.model_dir + "ga_train_" + str(iteration)

    # Log errors to log file of genetic algorithm
    global logger
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(cf.ga_logging_level)
        fh = logging.FileHandler(filename=cf.ga_logfile, mode='a+')
        fh.setLevel(cf.ga_logging_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = True

    # Create input functions if not yet created
    if not input_functions_created:
        create_input_functions()

    # Determine number of classes
    if cf.output_layer_x * cf.output_layer_y == 1:
        classes = 2
    else:
        classes = cf.output_layer_x * cf.output_layer_y

    # Ensure that sufficiently many checkpoints are kept
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=cf.max_validations_since_best_loss + 1)

    # Create estimator
    classifier = tf.estimator.Estimator(
        model_fn=create_model,
        model_dir=train_dir,
        config=run_config,
        params={
            # Number of classes
            'n_classes': classes,
            # The ANN
            'network': net,
            # Training directory
            'train_dir': train_dir
        })

    hooks = []
    # Create hook for early stopping. Use validation set data as input.
    if cf.employ_early_stopping:
        validation_listener = ValidationListener(estimator=classifier, model_dir=train_dir, input_fn=validate_input_fn)
        early_stopping_hook = tf.train.CheckpointSaverHook(checkpoint_dir=train_dir, save_steps=cf.validation_interval,
                                                           listeners=[validation_listener])
        hooks.append(early_stopping_hook)

    # Number of training rounds is based on net's number of free parameters
    # (excluding free parameters for batch normalization)
    num_trainings = max(1, cf.max_trainings_per_evaluation - net.net_parameters
                        // cf.max_trainings_decrease_per_free_params)

    # Train and evaluate net
    max_accuracy = 0.0
    free_params = 0

    for _ in range(num_trainings):
        if cf.employ_early_stopping:
            # Reset saved loss in validation listener
            validation_listener.reset_loss()
            # Create validation summary writer
            tf.summary.FileWriter(os.path.join(train_dir, 'validation'))
        # Create evaluation summary writer
        tf.summary.FileWriter(os.path.join(train_dir, 'evaluation'))
        # Train net
        try:
            classifier.train(input_fn=train_input_fn, max_steps=cf.max_training_steps, hooks=hooks)
        except tf.train.NanLossDuringTrainingError:
            logger.error("NaN loss during training.")
            weights = [None] * len(net.layers)
            net.weights = weights
            free_params = classifier.get_variable_value('free_params')
            # Close summary writer
            # summary_writer.close()
            try:
                shutil.rmtree(train_dir, ignore_errors=True)
            except (OSError, shutil.Error) as e:
                print("Could not delete model directory!")
                logger.error("Could not delete model directory!")
            return np.float(0.0), np.int(free_params)

        # Evaluate net
        path_to_best_checkpoint = None
        if cf.employ_early_stopping:
            path_to_best_checkpoint = train_dir + "/model.ckpt-" + str(best_checkpoint_at_step)
        eval_result = classifier.evaluate(input_fn=eval_input_fn, checkpoint_path=path_to_best_checkpoint)
        if cf.employ_early_stopping:
            global_step_value = gl_step
        else:
            global_step_value = cf.max_training_steps
        summary_writer = tf.summary.FileWriterCache.get(os.path.join(train_dir, 'evaluation'))
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=eval_result['loss'])
        summary.value.add(tag='accuracy', simple_value=eval_result['accuracy'])
        summary_writer.add_summary(summary, global_step_value)
        summary_writer.close()
        # Close summary writer in training directory
        summary_writer = tf.summary.FileWriterCache.get(train_dir)
        summary_writer.close()
        # Close summary writer in validation and TensorFlow evaluation directories
        if cf.employ_early_stopping:
            summary_writer = tf.summary.FileWriterCache.get(os.path.join(train_dir, 'validation'))
            summary_writer.close()
        summary_writer = tf.summary.FileWriterCache.get(os.path.join(train_dir, 'eval'))
        summary_writer.close()
        if eval_result['accuracy'] > max_accuracy:
            max_accuracy = eval_result['accuracy']
            free_params = classifier.get_variable_value('free_params')
            # Extract weights and save in genotype
            weights = [None] * len(net.layers)
            if cf.employ_weight_sharing:
                for i in range(1, len(net.layers)):
                    if net.layers[i].connection_type == gt.ConnectionType.DENSE or\
                            net.layers[i].connection_type == gt.ConnectionType.CONVOLUTIONAL:
                        bias = classifier.get_variable_value(str(i) + '/bias')
                        kernel = classifier.get_variable_value(str(i) + '/kernel')
                        layer_weights = [kernel, bias]
                        weights[i] = layer_weights
            net.weights = weights

        # Check whether model is to be kept
        if max_accuracy > best_current_score or (max_accuracy == best_current_score
                                                 and free_params < best_current_params):
            best_current_score = max_accuracy
            best_current_params = free_params
            save_best_model(train_dir, net, best_checkpoint_at_step, iteration)

        # Delete model directory
        try:
            shutil.rmtree(train_dir, ignore_errors=True)
        except (OSError, shutil.Error) as e:
            print("Could not delete model directory!")
            logger.error("Could not delete model directory!")

        # Maximal accuracy does not necessitate further training runs
        if eval_result['accuracy'] == 1.0:
            break

    # print("Last change:", net.pos_last_change)
    # print("Weights:", net.weights)
    # Net score is equivalent to classification accuracy
    return np.float(max_accuracy), np.int(free_params)
