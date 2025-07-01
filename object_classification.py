"""
Biological Object Classification Neural Network
==============================================

CNN for classifying microscopic aquatic organisms from 50x50 grayscale images.
Categories: cladocera, copepod, junk, rotifer.

Architecture: 3 conv blocks -> 3 FC layers -> 4-class output
Uses TensorFlow 1.x syntax with LRN and Dropout regularization.
"""

import os
import sys
import pickle
import numpy as np
from imutils import paths
from tools.nn.soft_max import SoftMax
from tools.io.batch_generator import BatchGenerator
from tools.preprocessing.image_preprocessing import ImagePreprocessor
from tools.metrics.entropy import Entropy
from tools.metrics.least_confidence import LeastConfidence
from tools.metrics.margin_sampling import MarginSampling
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Suppress TensorFlow logging messages - MUST be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@dataclass
class ClassificationConfig:
    """Configuration for the classification pipeline."""
    input_path: str
    model_path: str
    output_path: str = './classification'
    batch_size: int = 32
    input_size: int = 50
    input_depth: int = 1
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ['cladocera', 'copepod', 'junk', 'rotifer']


class CNNArchitecture:
    """Handles CNN model construction with TensorFlow 1.x syntax."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.nr_categories = len(config.categories)
        
    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Constructs the CNN architecture and returns placeholders."""
        # Define placeholders
        x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.config.input_size, self.config.input_size])
        y_pred_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nr_categories])
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        
        # Build network
        y_pred = self._build_network(x_input, keep_prob)
        
        return x_input, keep_prob, y_pred
    
    def _build_network(self, x_input: tf.Tensor, keep_prob: tf.Tensor) -> tf.Tensor:
        """Builds the complete neural network architecture."""
        # LRN parameters
        lrn_params = {'depth_radius': 2, 'alpha': 2e-05, 'beta': 0.75, 'bias': 1.0}
        
        # Reshape input
        x_image = tf.reshape(x_input, [-1, self.config.input_size, self.config.input_size, self.config.input_depth])
        
        # Convolutional blocks
        conv1_out = self._conv_block_1(x_image, lrn_params)
        conv2_out = self._conv_block_2(conv1_out, lrn_params)
        conv3_out = self._conv_block_3(conv2_out)
        
        # Fully connected layers
        y_pred = self._fc_layers(conv3_out, keep_prob)
        
        return y_pred
    
    def _conv_block_1(self, x_image: tf.Tensor, lrn_params: Dict) -> tf.Tensor:
        """First convolutional block: [batch, 50, 50, 1] -> [batch, 12, 12, 32]"""
        conv1_filters = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.05), name="conv1W")
        conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]), name="conv1b")
        conv1_in = tf.compat.v1.nn.conv2d(x_image, conv1_filters, [1, 2, 2, 1], padding='SAME')
        conv1_out = tf.reshape(tf.nn.bias_add(conv1_in, conv1_bias), [-1] + conv1_in.get_shape().as_list()[1:])
        conv1_relu = tf.compat.v1.nn.relu(conv1_out)
        conv1_norm = tf.compat.v1.nn.local_response_normalization(conv1_relu, **lrn_params)
        return tf.compat.v1.nn.max_pool2d(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _conv_block_2(self, conv1_out: tf.Tensor, lrn_params: Dict) -> tf.Tensor:
        """Second convolutional block: [batch, 12, 12, 32] -> [batch, 5, 5, 64]"""
        conv2_filters = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.05), name="conv2W")
        conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]), name="conv2b")
        conv2_in = tf.compat.v1.nn.conv2d(conv1_out, conv2_filters, [1, 1, 1, 1], padding='SAME')
        conv2_out = tf.reshape(tf.nn.bias_add(conv2_in, conv2_bias), [-1] + conv2_in.get_shape().as_list()[1:])
        conv2_relu = tf.compat.v1.nn.relu(conv2_out)
        conv2_norm = tf.compat.v1.nn.local_response_normalization(conv2_relu, **lrn_params)
        return tf.compat.v1.nn.max_pool2d(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _conv_block_3(self, conv2_out: tf.Tensor) -> tf.Tensor:
        """Third convolutional block: [batch, 5, 5, 64] -> [batch, 2, 2, 96]"""
        # Conv3
        conv3_filters = tf.Variable(tf.random.truncated_normal([3, 3, 64, 96], stddev=0.05), name="conv3W")
        conv3_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv3b")
        conv3_in = tf.compat.v1.nn.conv2d(conv2_out, conv3_filters, [1, 1, 1, 1], padding='SAME')
        conv3_out = tf.reshape(tf.nn.bias_add(conv3_in, conv3_bias), [-1] + conv3_in.get_shape().as_list()[1:])
        conv3_relu = tf.compat.v1.nn.relu(conv3_out)
        
        # Conv4
        conv4_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv4W")
        conv4_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv4b")
        conv4_in = tf.compat.v1.nn.conv2d(conv3_relu, conv4_filters, [1, 1, 1, 1], padding='SAME')
        conv4_out = tf.reshape(tf.nn.bias_add(conv4_in, conv4_bias), [-1] + conv4_in.get_shape().as_list()[1:])
        conv4_relu = tf.compat.v1.nn.relu(conv4_out)
        
        # Conv5
        conv5_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv5W")
        conv5_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv5b")
        conv5_in = tf.compat.v1.nn.conv2d(conv4_relu, conv5_filters, [1, 1, 1, 1], padding='SAME')
        conv5_out = tf.reshape(tf.nn.bias_add(conv5_in, conv5_bias), [-1] + conv5_in.get_shape().as_list()[1:])
        conv5_relu = tf.compat.v1.nn.relu(conv5_out)
        
        return tf.compat.v1.nn.max_pool2d(conv5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _fc_layers(self, conv_out: tf.Tensor, keep_prob: tf.Tensor) -> tf.Tensor:
        """Fully connected layers: [batch, 2, 2, 96] -> [batch, 4]"""
        # FC6
        fc6_nodes = tf.Variable(tf.random.truncated_normal([384, 76], stddev=0.1), name="fc6")
        fc6_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc6")
        fc6_relu = tf.compat.v1.nn.relu_layer(
            tf.reshape(conv_out, [-1, int(np.prod(conv_out.get_shape()[1:]))]), 
            fc6_nodes, fc6_bias
        )
        fc6_drop = tf.compat.v1.nn.dropout(fc6_relu, rate=1 - keep_prob)
        
        # FC7
        fc7_nodes = tf.Variable(tf.random.truncated_normal([76, 76], stddev=0.1), name="fc7")
        fc7_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc7")
        fc7_relu = tf.compat.v1.nn.relu_layer(fc6_drop, fc7_nodes, fc7_bias)
        fc7_drop = tf.compat.v1.nn.dropout(fc7_relu, rate=1 - keep_prob)
        
        # FC8
        fc8_nodes = tf.Variable(tf.random.truncated_normal([76, self.nr_categories], stddev=0.1), name="fc8")
        fc8_bias = tf.Variable(tf.constant(0.1, shape=[self.nr_categories]), name="fc8")
        return tf.compat.v1.nn.xw_plus_b(fc7_drop, fc8_nodes, fc8_bias)


class InferenceEngine:
    """Handles model inference and prediction pipeline."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self._initialize_components()
        self._setup_model()
    
    def _initialize_components(self):
        """Initialize utility classes for processing and evaluation."""
        print('[CLASSIFICATION]: initializing batch generator...', flush=True, end='')
        self.gen = BatchGenerator()
        self.pre = ImagePreprocessor()
        self.en = Entropy()
        self.lc = LeastConfidence()
        self.ms = MarginSampling()
        self.softmax = SoftMax()
        print('DONE')
    
    def _setup_model(self):
        """Setup TensorFlow session and model."""
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.InteractiveSession()
        
        print('[CLASSIFICATION]: construct classification model...', flush=True, end='')
        cnn = CNNArchitecture(self.config)
        self.x_input, self.keep_prob, self.y_pred = cnn.build_model()
        print('DONE')
        
        print('[CLASSIFICATION]: restoring classification model...', flush=True, end='')
        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.sep.join([self.config.model_path, "model.ckpt"]))
        print('DONE')
    
    def process_location(self, location_path: str) -> Dict[str, Any]:
        """Process all images in a location and return classification results."""
        print('[CLASSIFICATION]: listing all vignettes...', flush=True, end='')
        image_paths = list(paths.list_images(location_path))
        print('DONE')
        
        if len(image_paths) == 0:
            print('[WARNING]: No vignettes where found. Please check the input path, or set input path with -input')
            sys.exit()
        
        # Preprocess images
        angles, images, image_mean = self.pre.image_preprocessing(image_paths, self.config.input_size)
        
        # Run batch predictions
        predictions = self._run_batch_predictions(images, image_mean, len(image_paths))
        
        # Post-process predictions
        return self._post_process_predictions(predictions, image_paths, angles)
    
    def _run_batch_predictions(self, images: np.ndarray, image_mean: float, num_images: int) -> np.ndarray:
        """Run batch predictions on preprocessed images."""
        predictions = np.array([], dtype=np.float32).reshape(0, len(self.config.categories))
        epoch_step = 0
        
        print('[CLASSIFICATION]: start vignettes classification...')
        for i in tqdm(range(int(np.ceil(num_images / self.config.batch_size))), desc='[CLASSIFICATION]'):
            batch_x = self.gen.batch_generator(
                images=images, images_mean=image_mean, nr_images=num_images,
                batch_size=self.config.batch_size, index=epoch_step
            )
            epoch_step += self.config.batch_size
            
            pred = self.sess.run(self.y_pred, feed_dict={self.x_input: batch_x, self.keep_prob: 1})
            predictions = np.concatenate((predictions, pred), axis=0)
        
        print('[CLASSIFICATION]: start vignettes classification...DONE')
        return predictions
    
    def _post_process_predictions(self, predictions: np.ndarray, image_paths: List[str], angles: List[float]) -> Dict[str, Any]:
        """Post-process predictions and calculate uncertainty metrics."""
        predictions = self.softmax.soft_max(predictions)
        
        # Calculate uncertainty metrics
        _, max_y = self.lc.least_confidence(predictions)
        _, diff = self.ms.margin_sampling(predictions)
        _, ent = self.en.entropy(predictions)
        
        probabilities = np.amax(predictions, axis=1).tolist()
        cat_predictions = np.argmax(predictions, axis=1).tolist()
        
        # Format output
        samples = []
        for i, image_path in enumerate(image_paths):
            samples.append({
                'paths': image_path,
                'y_predicted': cat_predictions[i],
                'full_y_predicted': predictions[i, :],
                'HC_lc': max_y[i],
                'HC_ms': diff[i],
                'HC_en': ent[i],
                'orientation': angles[i]
            })
        
        return {
            'classes': self.config.categories,
            'samples': samples
        }


class OutputHandler:
    """Handles output file generation."""
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str, filename: str):
        """Save classification results to pickle file."""
        print('[CLASSIFICATION]: creating pickle output...', flush=True, end='')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with open(os.path.sep.join([output_path, filename]), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('DONE')


class DirectoryProcessor:
    """Handles directory traversal and processing coordination."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.inference_engine = InferenceEngine(config)
        self.output_handler = OutputHandler()
    
    def process_all_directories(self):
        """Process all directories in the input path hierarchy."""
        projects = os.listdir(self.config.input_path)
        
        # Count total locations first
        total_locations = 0
        for project in projects:
            project_path = os.path.sep.join([self.config.input_path, project])
            dates = os.listdir(project_path)
            for date in dates:
                date_path = os.path.sep.join([project_path, date])
                times = os.listdir(date_path)
                for time in times:
                    time_path = os.path.sep.join([date_path, time])
                    locations = os.listdir(time_path)
                    total_locations += len(locations)
        
        print(f'[CLASSIFICATION]: Found {total_locations} locations to process')
        
        location_count = 0
        for project in projects:
            project_path = os.path.sep.join([self.config.input_path, project])
            dates = os.listdir(project_path)
            
            for date in dates:
                date_path = os.path.sep.join([project_path, date])
                times = os.listdir(date_path)
                
                for time in times:
                    time_path = os.path.sep.join([date_path, time])
                    locations = os.listdir(time_path)
                    
                    for location in locations:
                        location_count += 1
                        location_path = os.path.sep.join([time_path, location])
                        full_path = f"{project}/{date}/{time}/{location}"
                        print(f'\n[CLASSIFICATION]: Processing location {location_count}/{total_locations}: {full_path}')
                        self._process_single_location(project, date, time, location, location_path)
    
    def _process_single_location(self, project: str, date: str, time: str, location: str, location_path: str):
        """Process a single location directory."""
        results = self.inference_engine.process_location(location_path)
        
        output_path = os.path.sep.join([self.config.output_path, project, date, time])
        filename = f'{project}_{date}_{location}_classification.pkl'
        
        self.output_handler.save_results(results, output_path, filename)


def classify_objects(input_path, model_path, batch_size=32, input_size=50, input_depth=1, output_path='./classification'):
    """Main classification function maintaining original interface."""
    print('\n[CLASSIFICATION]: starting vignettes classification...')
    
    config = ClassificationConfig(
        input_path=input_path,
        model_path=model_path,
        batch_size=batch_size,
        input_size=input_size,
        input_depth=input_depth,
        output_path=output_path
    )
    
    processor = DirectoryProcessor(config)
    processor.process_all_directories()
    
    print('\n[CLASSIFICATION]: vignettes classification is DONE')
