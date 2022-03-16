"""
This module provides functions that deliver input data for the estimator, function of choice must be specified in the
configuration file.
"""

import numpy as np
import tensorflow as tf
import os


def load_data_mnist():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_features = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    validate_features = mnist.validation.images
    validate_labels = np.asarray(mnist.validation.labels, dtype=np.int32)
    test_features = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return (train_features, train_labels), (validate_features, validate_labels), (test_features, test_labels)
   
