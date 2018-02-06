# https://cntk.ai/pythondocs/CNTK_101_LogisticRegression.html#Introduction
# ------Function------
# A cancer hospital has provided data and wants us to determine 
# if a patient has a fatal malignant cancer vs. a benign growth
# features: age and tumor size

# Import the relevant componets
from __future__ import print_function
import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils # pytest is a python lib that can be pipped.
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix the random so that LR exmaples are repeatable.

# Define the network
input_dim = 2
num_output_classes = 2

# Ensure that we always get the same results
np.random.seed(0)

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
	# Create synthetic data using Numpy.
	Y = np.random.randint(size=(sample_size,1), low=0, high=num_classes)
	
	# Make sure that the data is separable
	X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)

	# Specify the data type to match the input variable used later in the tutorial
	# (default type is double)
	X = X.astype(np.float32)

	# convert class 0 into the vector "1 0 0"
	# class 1 into the vector "0 1 0", ...
	class_ind = [Y==class_number for class_number in range(num_classes)]
	Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
	return X,Y

mysamplesize = 32
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)
print (features, labels)
