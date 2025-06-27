"""
Biological Object Classification Neural Network
==============================================

This script implements a Convolutional Neural Network (CNN) for classifying 
microscopic aquatic organisms from image vignettes. The model is designed to 
distinguish between four categories: cladocera, copepod, junk, and rotifer.

Architecture Overview:
- Input: 50x50 grayscale images
- 3 Convolutional blocks with progressive feature extraction
- 3 Fully connected layers for classification
- Output: 4-class probability distribution

The model uses TensorFlow 1.x syntax and includes regularization techniques
like Local Response Normalization (LRN) and Dropout to prevent overfitting.

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import argparse
from imutils import paths
from tools.nn.soft_max import SoftMax
from tools.io.batch_generator import BatchGenerator
from tools.preprocessing.image_preprocessing import ImagePreprocessor
from tools.metrics.entropy import Entropy
from tools.metrics.least_confidence import LeastConfidence
from tools.metrics.margin_sampling import MarginSampling
from tqdm import tqdm

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, default='./vignettes', help="path to vignettes")
ap.add_argument('-model', type=str, default='./model', help='path to classification model')
ap.add_argument('-batch_size', default=32, type=int, help='set batch_size; default = 32')
ap.add_argument('-input_size', default=50, type=int, help='image input size; default = 50')
ap.add_argument('-input_depth', default=1, type=int, help='image greyscale (1) or RGB (3); default 3')
ap.add_argument("-output", type=str, default='./classification', help="path to classification output")
args = vars(ap.parse_args())

# =============================================================================
# INITIALIZATION AND SETUP
# =============================================================================
print('[Progress update]: starting vignettes classification...')

# Disable eager execution to use TensorFlow 1.x style placeholders
tf.compat.v1.disable_eager_execution()

# Start interactive TensorFlow session for dynamic graph execution
sess = tf.compat.v1.InteractiveSession()

# Extract configuration parameters
batchSize = args["batch_size"]      # Number of images processed simultaneously
imageSize = args["input_size"]      # Input image dimensions (default: 50x50)
imageDepth = args["input_depth"]    # Number of color channels (1=grayscale, 3=RGB)

# Define classification categories (aquatic organisms)
categories = ['cladocera', 'copepod', 'junk', 'rotifer']
nr_categories = len(categories)

# Initialize utility classes for data processing and evaluation
print('[Progress update]: initializing batch generator...', flush=True, end='')
gen = BatchGenerator()              # Handles batch creation for training/inference
pre = ImagePreprocessor()           # Preprocesses input images
en = Entropy()                      # Calculates prediction entropy for uncertainty
lc = LeastConfidence()              # Least confidence uncertainty metric
ms = MarginSampling()               # Margin sampling uncertainty metric
print('DONE')

# =============================================================================
# NEURAL NETWORK ARCHITECTURE CONSTRUCTION
# =============================================================================
print('[Progress update]: construct classification model...', flush=True, end='')

# Define TensorFlow placeholders for dynamic input
x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, imageSize, imageSize])  # Input images
y_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, nr_categories])          # Output predictions
keep_prob = tf.compat.v1.placeholder(tf.float32)                                    # Dropout probability

# Local Response Normalization (LRN) parameters
# LRN normalizes activations across adjacent feature maps to prevent overfitting
radius = 2        # Number of adjacent feature maps to consider (total window = 2*radius + 1)
alpha = 2e-05     # Scaling factor for normalization strength
beta = 0.75       # Exponent for normalization (controls sharpness)
bias = 1.0        # Additive term to prevent division by zero

# Reshape input to add channel dimension required for 2D convolutions
# Input: [batch_size, height, width] -> [batch_size, height, width, channels]
x_image = tf.reshape(x_input, [-1, imageSize, imageSize, imageDepth])

# =============================================================================
# CONVOLUTIONAL BLOCK 1: INITIAL FEATURE DETECTION
# =============================================================================
# Purpose: Extract low-level features like edges, corners, and basic shapes
# Input: [batch, 50, 50, 1] -> Output: [batch, 12, 12, 32]

# Conv1: 5x5 filters with 32 output channels, stride 2 for dimensionality reduction
conv1_filters = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.05), name="conv1W")
conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]), name="conv1b")

# 2D Convolution operation
# Stride [1, 2, 2, 1]: stride 2 in spatial dimensions (height, width)
# Padding 'SAME': output size = input size / stride = 50/2 = 25
conv1_in = tf.compat.v1.nn.conv2d(x_image, conv1_filters, [1, 2, 2, 1], padding='SAME')

# Add bias terms to each feature map and reshape for proper tensor dimensions
conv1_out = tf.reshape(tf.nn.bias_add(conv1_in, conv1_bias), [-1] + conv1_in.get_shape().as_list()[1:])

# ReLU activation: f(x) = max(0, x)
# Introduces non-linearity and removes negative activations
conv1_relu = tf.compat.v1.nn.relu(conv1_out)

# Local Response Normalization (LRN)
# Normalizes activations across adjacent feature maps to prevent overfitting
# Mathematical formula: b_{x,y}^i = a_{x,y}^i / (bias + α * Σ_{j=max(0,i-n)}^{min(N-1,i+n)} (a_{x,y}^j)²)^β
conv1_norm = tf.compat.v1.nn.local_response_normalization(conv1_relu, depth_radius=radius, alpha=alpha, beta=beta,
                                                          bias=bias)

# MaxPooling: 3x3 kernel with stride 2
# Purpose: Translation invariance and further dimensionality reduction
# Padding 'VALID': no padding, output size = (input_size - kernel_size) / stride + 1 = (25-3)/2 + 1 = 12
conv1_maxpool = tf.compat.v1.nn.max_pool2d(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# =============================================================================
# CONVOLUTIONAL BLOCK 2: INTERMEDIATE FEATURE LEARNING
# =============================================================================
# Purpose: Combine low-level features into mid-level patterns
# Input: [batch, 12, 12, 32] -> Output: [batch, 5, 5, 64]

# Conv2: 5x5 filters with 64 output channels, stride 1 (no spatial reduction)
conv2_filters = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.05), name="conv2W")
conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]), name="conv2b")

# 2D Convolution: combines 32 input features into 64 more complex features
conv2_in = tf.compat.v1.nn.conv2d(conv1_maxpool, conv2_filters, [1, 1, 1, 1], padding='SAME')
conv2_out = tf.reshape(tf.nn.bias_add(conv2_in, conv2_bias), [-1] + conv2_in.get_shape().as_list()[1:])
conv2_relu = tf.compat.v1.nn.relu(conv2_out)

# Apply LRN and MaxPooling (same parameters as Block 1)
conv2_norm = tf.compat.v1.nn.local_response_normalization(conv2_relu, depth_radius=radius, alpha=alpha, beta=beta,
                                                          bias=bias)
conv2_maxpool = tf.compat.v1.nn.max_pool2d(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# =============================================================================
# CONVOLUTIONAL BLOCK 3: DEEP FEATURE EXTRACTION
# =============================================================================
# Purpose: Extract high-level, complex features through multiple 3x3 convolutions
# Input: [batch, 5, 5, 64] -> Output: [batch, 2, 2, 96]

# Conv3: First 3x3 convolution, increases channels from 64 to 96
conv3_filters = tf.Variable(tf.random.truncated_normal([3, 3, 64, 96], stddev=0.05), name="conv3W")
conv3_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv3b")
conv3_in = tf.compat.v1.nn.conv2d(conv2_maxpool, conv3_filters, [1, 1, 1, 1], padding='SAME')
conv3_out = tf.reshape(tf.nn.bias_add(conv3_in, conv3_bias), [-1] + conv3_in.get_shape().as_list()[1:])
conv3_relu = tf.compat.v1.nn.relu(conv3_out)

# Conv4: Second 3x3 convolution, maintains 96 channels
# Purpose: Further refine and process the 96 feature maps
conv4_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv4W")
conv4_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv4b")
conv4_in = tf.compat.v1.nn.conv2d(conv3_relu, conv4_filters, [1, 1, 1, 1], padding='SAME')
conv4_out = tf.reshape(tf.nn.bias_add(conv4_in, conv4_bias), [-1] + conv4_in.get_shape().as_list()[1:])
conv4_relu = tf.compat.v1.nn.relu(conv4_out)

# Conv5: Third 3x3 convolution, maintains 96 channels
# Purpose: Final convolutional processing before classification
conv5_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv5W")
conv5_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv5b")
conv5_in = tf.compat.v1.nn.conv2d(conv4_relu, conv5_filters, [1, 1, 1, 1], padding='SAME')
conv5_out = tf.reshape(tf.nn.bias_add(conv5_in, conv5_bias), [-1] + conv5_in.get_shape().as_list()[1:])
conv5_relu = tf.compat.v1.nn.relu(conv5_out)

# Final MaxPooling: 2x2 kernel with stride 2
# Reduces spatial dimensions from 5x5 to 2x2
conv5_maxpool = tf.compat.v1.nn.max_pool2d(conv5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# =============================================================================
# FULLY CONNECTED LAYERS: CLASSIFICATION
# =============================================================================
# Purpose: Transform spatial features into class probabilities
# Input: [batch, 2, 2, 96] = [batch, 384] -> Output: [batch, 4]

# FC6: Bottleneck layer - reduces dimensions from 384 to 76
# This creates a compressed representation of the features
fc6_nodes = tf.Variable(tf.random.truncated_normal([384, 76], stddev=0.1), name="fc6")
fc6_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc6")

# Flatten the convolutional output and apply fully connected layer
# np.prod(conv5_maxpool.get_shape()[1:]) = 2 * 2 * 96 = 384
fc6_relu = tf.compat.v1.nn.relu_layer(tf.reshape(conv5_maxpool, [-1, int(np.prod(conv5_maxpool.get_shape()[1:]))]),
                                      fc6_nodes, fc6_bias)

# Dropout regularization: randomly sets 50% of activations to zero during training
# Prevents overfitting by forcing the network to not rely on specific neurons
fc6_drop = tf.compat.v1.nn.dropout(fc6_relu, rate=1 - keep_prob)

# FC7: Hidden layer - maintains 76 dimensions
# Further processes the compressed features
fc7_nodes = tf.Variable(tf.random.truncated_normal([76, 76], stddev=0.1), name="fc7")
fc7_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc7")
fc7_relu = tf.compat.v1.nn.relu_layer(fc6_drop, fc7_nodes, fc7_bias)
fc7_drop = tf.compat.v1.nn.dropout(fc7_relu, rate=1 - keep_prob)

# FC8: Output layer - maps to 4 class logits
# Linear activation (no ReLU) to allow for proper softmax application
fc8_nodes = tf.Variable(tf.random.truncated_normal([76, nr_categories], stddev=0.1), name="fc8")
fc8_bias = tf.Variable(tf.constant(0.1, shape=[nr_categories]), name="fc8")
y_pred = tf.compat.v1.nn.xw_plus_b(fc7_drop, fc8_nodes, fc8_bias)

print('DONE')

# =============================================================================
# MODEL RESTORATION
# =============================================================================
# Load pre-trained weights from checkpoint file
print('[Progress update]: restoring classification model...', flush=True, end='')
sess.run(tf.compat.v1.global_variables_initializer())  # Initialize all variables
saver = tf.compat.v1.train.Saver()                    # Create saver for loading weights
saver.restore(sess, os.path.sep.join([args['model'], "model.ckpt"]))  # Load trained weights
print('DONE')

# =============================================================================
# INFERENCE LOOP: CLASSIFY IMAGES
# =============================================================================
# Process images organized by project/date/time/location hierarchy

# Get input directory and list all project folders
directory = args['input']
projects = os.listdir(directory)

# Loop through each project
for project in projects:
    projectPath = os.path.sep.join([directory, project])
    dates = os.listdir(projectPath)

    # Loop through each date
    for date in dates:
        datePath = os.path.sep.join([projectPath, date])
        times = os.listdir(datePath)

        # Loop through each time
        for time in times:
            timePath = os.path.sep.join([datePath, time])
            locations = os.listdir(timePath)

            # Loop through each location to perform classification
            for location in locations:
                locationPath = os.path.sep.join([timePath, location])

                # Find all image files in the current location
                print('[Progress update]: listing all vignettes...', flush=True, end='')
                imagePaths = list(paths.list_images(locationPath))
                print('DONE')
                
                # Check if any images were found
                if len(imagePaths) == 0:
                    print('[WARNING]: No vignettes where found. Please check the input path, or set input path with -input')
                    sys.exit()

                # =============================================================================
                # IMAGE PREPROCESSING
                # =============================================================================
                # Preprocess images: resize, normalize, and extract orientation information
                angles, images, image_mean = pre.image_preprocessing(imagePaths, imageSize)

                # =============================================================================
                # BATCH PREDICTION
                # =============================================================================
                # Initialize prediction array and batch processing
                predictions = np.array([], dtype=np.float32).reshape(0, nr_categories)
                epoch_step = 0
                print('[Progress update]: start vignettes classification...')
                
                # Process images in batches for memory efficiency
                for i in tqdm(range(int(np.ceil(len(imagePaths) / batchSize))), desc='[Progress update]: classifying'):
                    # Generate batch of preprocessed images
                    batch_x = gen.batch_generator(images=images, images_mean=image_mean, nr_images=len(imagePaths),
                                                  batch_size=batchSize, index=epoch_step)
                    epoch_step += batchSize
                    
                    # Run forward pass through the neural network
                    # keep_prob=1.0 disables dropout during inference
                    pred = sess.run(y_pred, feed_dict={x_input: batch_x, keep_prob: 1})
                    predictions = np.concatenate((predictions, pred), axis=0)
                print('[Progress update]: start vignettes classification...DONE')

                # =============================================================================
                # POST-PROCESSING AND UNCERTAINTY CALCULATION
                # =============================================================================
                # Apply softmax to convert logits to probabilities
                predictions = SoftMax().soft_max(predictions)

                # Calculate uncertainty metrics for active learning
                # These metrics help identify which samples are most uncertain
                _, max_y = lc.least_confidence(predictions)      # Least confidence uncertainty
                _, diff = ms.margin_sampling(predictions)        # Margin sampling uncertainty  
                _, ent = en.entropy(predictions)                 # Entropy-based uncertainty

                # Extract final predictions
                probabilities = np.amax(predictions, axis=1).tolist()  # Highest probability per sample
                cat_predictions = np.argmax(predictions, axis=1).tolist()  # Predicted class indices

                # =============================================================================
                # OUTPUT GENERATION
                # =============================================================================
                print('[Progress update]: creating pickle output...', flush=True, end='')

                # Create output directory structure
                outputPath = os.path.sep.join([args['output'], project, date, time])
                if not os.path.exists(outputPath): 
                    os.makedirs(outputPath)

                # Generate output filename
                filename = f'{project}_{date}_{location}_classification.pkl'

                # Compile results into structured format
                out_ = []
                for i, image_path in enumerate(imagePaths):
                    # Create dictionary for each classified image
                    out_.append({
                        'paths': image_path,                    # Original image path
                        'y_predicted': cat_predictions[i],      # Predicted class index
                        'full_y_predicted': predictions[i, :],  # Full probability distribution
                        'HC_lc': max_y[i],                      # Least confidence uncertainty
                        'HC_ms': diff[i],                       # Margin sampling uncertainty
                        'HC_en': ent[i],                        # Entropy uncertainty
                        'orientation': angles[i]                # Image orientation information
                    })

                # Create final output structure
                out = {
                    'classes': categories,    # Class names for reference
                    'samples': out_           # Classification results for each image
                }

                # Save results to pickle file
                with open(os.path.sep.join([outputPath, filename]), 'wb') as handle:
                    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('DONE')

print('[Progress update]: vignettes classification is DONE')
