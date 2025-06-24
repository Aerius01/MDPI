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

ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, default='./vignettes', help="path to vignettes")
ap.add_argument('-model', type=str, default='./model', help='path to classification model')
ap.add_argument('-batch_size', default=32, type=int, help='set batch_size; default = 32')
ap.add_argument('-input_size', default=50, type=int, help='image input size; default = 50')
ap.add_argument('-input_depth', default=1, type=int, help='image greyscale (1) or RGB (3); default 3')
ap.add_argument("-output", type=str, default='./classification', help="path to classification output")
args = vars(ap.parse_args())

# progress udpate
print('[Progress update]: starting vignettes classification...')

# start interactive tensorflow session
sess = tf.compat.v1.InteractiveSession()

# settings for classification
batchSize = args["batch_size"]
imageSize = args["input_size"]
imageDepth = args["input_depth"]

# classification categories
categories = ['cladocera', 'copepod', 'junk', 'rotifer']
nr_categories = len(categories)

# initialize the batch generator and image prepocessor
print('[Progress update]: initializing batch generator...', flush=True, end='')
gen = BatchGenerator()
pre = ImagePreprocessor()
en = Entropy()
lc = LeastConfidence()
ms = MarginSampling()
print('DONE')

# initialize classification model
print('[Progress update]: construct classification model...', flush=True, end='')
# initialize input and prediction placeholders
x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, imageSize, imageSize])
y_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, nr_categories])
keep_prob = tf.compat.v1.placeholder(tf.float32)

#
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0
# input shape
x_image = tf.reshape(x_input, [-1, imageSize, imageSize, imageDepth])

# convolutional block 1 CONV + RELU + NORM + MAXPOOL
conv1_filters = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.05), name="conv1W")
conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]), name="conv1b")
conv1_in = tf.compat.v1.nn.conv2d(x_image, conv1_filters, [1, 2, 2, 1], padding='SAME')
conv1_out = tf.reshape(tf.nn.bias_add(conv1_in, conv1_bias), [-1] + conv1_in.get_shape().as_list()[1:])
conv1_relu = tf.compat.v1.nn.relu(conv1_out)
conv1_norm = tf.compat.v1.nn.local_response_normalization(conv1_relu, depth_radius=radius, alpha=alpha, beta=beta,
                                                          bias=bias)
conv1_maxpool = tf.compat.v1.nn.max_pool2d(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# convolutional block 2 CONV + RELU + NORM + MAXPOOL
conv2_filters = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.05), name="conv2W")
conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]), name="conv2b")
conv2_in = tf.compat.v1.nn.conv2d(conv1_maxpool, conv2_filters, [1, 1, 1, 1], padding='SAME')
conv2_out = tf.reshape(tf.nn.bias_add(conv2_in, conv2_bias), [-1] + conv2_in.get_shape().as_list()[1:])
conv2_relu = tf.compat.v1.nn.relu(conv2_out)
conv2_norm = tf.compat.v1.nn.local_response_normalization(conv2_relu, depth_radius=radius, alpha=alpha, beta=beta,
                                                          bias=bias)
conv2_maxpool = tf.compat.v1.nn.max_pool2d(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# convolutional block 3 CONV + RELU + CONV + RELU + CONV + RELU + MAXPOOL
conv3_filters = tf.Variable(tf.random.truncated_normal([3, 3, 64, 96], stddev=0.05), name="conv3W")
conv3_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv3b")
conv3_in = tf.compat.v1.nn.conv2d(conv2_maxpool, conv3_filters, [1, 1, 1, 1], padding='SAME')
conv3_out = tf.reshape(tf.nn.bias_add(conv3_in, conv3_bias), [-1] + conv3_in.get_shape().as_list()[1:])
conv3_relu = tf.compat.v1.nn.relu(conv3_out)

conv4_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv4W")
conv4_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv4b")
conv4_in = tf.compat.v1.nn.conv2d(conv3_relu, conv4_filters, [1, 1, 1, 1], padding='SAME')
conv4_out = tf.reshape(tf.nn.bias_add(conv4_in, conv4_bias), [-1] + conv4_in.get_shape().as_list()[1:])
conv4_relu = tf.compat.v1.nn.relu(conv4_out)

conv5_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv5W")
conv5_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv5b")
conv5_in = tf.compat.v1.nn.conv2d(conv4_relu, conv5_filters, [1, 1, 1, 1], padding='SAME')
conv5_out = tf.reshape(tf.nn.bias_add(conv5_in, conv5_bias), [-1] + conv5_in.get_shape().as_list()[1:])
conv5_relu = tf.compat.v1.nn.relu(conv5_out)
conv5_maxpool = tf.compat.v1.nn.max_pool2d(conv5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# fully connected layers 2048 -> 76 -> 76 -> y_pred
fc6_nodes = tf.Variable(tf.random.truncated_normal([384, 76], stddev=0.1), name="fc6")
fc6_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc6")
fc6_relu = tf.compat.v1.nn.relu_layer(tf.reshape(conv5_maxpool, [-1, int(np.prod(conv5_maxpool.get_shape()[1:]))]),
                                      fc6_nodes, fc6_bias)
fc6_drop = tf.compat.v1.nn.dropout(fc6_relu, rate=1 - keep_prob)

fc7_nodes = tf.Variable(tf.random.truncated_normal([76, 76], stddev=0.1), name="fc7")
fc7_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc7")
fc7_relu = tf.compat.v1.nn.relu_layer(fc6_drop, fc7_nodes, fc7_bias)
fc7_drop = tf.compat.v1.nn.dropout(fc7_relu, rate=1 - keep_prob)

fc8_nodes = tf.Variable(tf.random.truncated_normal([76, nr_categories], stddev=0.1), name="fc8")
fc8_bias = tf.Variable(tf.constant(0.1, shape=[nr_categories]), name="fc8")
y_pred = tf.compat.v1.nn.xw_plus_b(fc7_drop, fc8_nodes, fc8_bias)

# finished construction
print('DONE')

# restore trained model
print('[Progress update]: restoring classification model...', flush=True, end='')
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()
saver.restore(sess, os.path.sep.join([args['model'], "model.ckpt"]))
print('DONE')

# We loop over the project, date and do classification per location
# list project folders
directory = args['input']
projects = os.listdir(directory)

# loop over the projects
for project in projects:
    # set project path
    projectPath = os.path.sep.join([directory, project])

    # list dates
    dates = os.listdir(projectPath)

    # loop over the dates
    for date in dates:
        # set date path
        datePath = os.path.sep.join([projectPath, date])

        # list times
        times = os.listdir(datePath)

        for time in times:
            # set time path
            timePath = os.path.sep.join([datePath, time])

            # list locations
            locations = os.listdir(timePath)

            # loop over locations to do classifications
            for location in locations:
                # set location path
                locationPath = os.path.sep.join([timePath, location])

                # list all images to classify
                print('[Progress update]: listing all vignettes...', flush=True, end='')
                imagePaths = list(paths.list_images(locationPath))
                print('DONE')
                # in case no images are found
                if len(imagePaths) is 0:
                    print('[WARNING]: No vignettes where found. Please check the input path, or set input path with -input')
                    sys.exit()

                # preprocessing images
                angles, images, image_mean = pre.image_preprocessing(imagePaths, imageSize)

                # predicting images
                predictions = np.array([], dtype=np.float32).reshape(0, nr_categories)
                epoch_step = 0
                print('[Progress update]: start vignettes classification...')
                for i in tqdm(range(int(np.ceil(len(imagePaths) / batchSize))), desc='[Progress update]: classifying'):
                    batch_x = gen.batch_generator(images=images, images_mean=image_mean, nr_images=len(imagePaths),
                                                  batch_size=batchSize, index=epoch_step)
                    epoch_step += batchSize
                    pred = sess.run(y_pred, feed_dict={x_input: batch_x, keep_prob: 1})
                    predictions = np.concatenate((predictions, pred), axis=0)
                print('[Progress update]: start vignettes classification...DONE')

                # find prediction
                predictions = SoftMax().soft_max(predictions)

                # metrics
                _, max_y = lc.least_confidence(predictions)
                _, diff = ms.margin_sampling(predictions)
                _, ent = en.entropy(predictions)

                # selecting and storing highest probability of each array
                # transforming the array into a list to enable indexing
                probabilities = np.amax(predictions, axis=1).tolist()
                cat_predictions = np.argmax(predictions, axis=1).tolist()

                # create pickle output
                print('[Progress update]: creating pickle output...', flush=True, end='')

                # create output path
                outputPath = os.path.sep.join([args['output'], project, date, time])
                if not os.path.exists(outputPath): os.makedirs(outputPath)

                # set filename
                filename = f'{project}_{date}_{location}_classification.pkl'

                # store results in dict.
                out_ = []
                for i, image_path in enumerate(imagePaths):
                    # create output dict for pickle
                    out_.append({'paths': image_path,
                                 'y_predicted': cat_predictions[i],
                                 'full_y_predicted': predictions[i, :],
                                 'HC_lc': max_y[i],
                                 'HC_ms': diff[i],
                                 'HC_en': ent[i],
                                 'orientation': angles[i]})

                out = {'classes': categories,
                       'samples': out_}

                with open(os.path.sep.join([outputPath, filename]), 'wb') as handle:
                    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('DONE')
print('[Progress update]: vignettes classification is DONE')
