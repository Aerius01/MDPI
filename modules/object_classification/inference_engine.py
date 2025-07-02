import os
import numpy as np
from tools.nn.soft_max import SoftMax
from tools.io.batch_generator import BatchGenerator
from tools.preprocessing.image_preprocessing import ImagePreprocessor
from tools.metrics.entropy import Entropy
from tools.metrics.least_confidence import LeastConfidence
from tools.metrics.margin_sampling import MarginSampling
from tqdm import tqdm
from typing import List, Dict, Any
from .config import ClassificationConfig
from .architecture import CNNArchitecture

# Suppress TensorFlow logging messages - MUST be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class InferenceEngine:
    """Handles model inference and prediction pipeline."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.sess = None
        self._initialize_components()
        self._setup_model()
    
    def _initialize_components(self):
        """Initialize utility classes for processing and evaluation."""
        print('[CLASSIFICATION]: Initializing batch generator...')
        self.gen = BatchGenerator()
        self.pre = ImagePreprocessor()
        self.en = Entropy()
        self.lc = LeastConfidence()
        self.ms = MarginSampling()
        self.softmax = SoftMax()
    
    def _setup_model(self):
        """Setup TensorFlow session and model."""
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()  # Use Session instead of InteractiveSession
        
        print('[CLASSIFICATION]: Constructing classification model...')
        cnn = CNNArchitecture(self.config)
        self.x_input, self.keep_prob, self.y_pred = cnn.build_model()
        
        print('[CLASSIFICATION]: Restoring classification model...')
        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.sep.join([self.config.model_path, "model.ckpt"]))
    
    def process_location(self, image_paths: List[str]) -> Dict[str, Any]:
        """Process all images in the provided list and return classification results."""
        
        if len(image_paths) == 0:
            raise ValueError("No images provided in image_paths list")
        
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
        
        print('[CLASSIFICATION]: Starting vignettes classification...')
        print(f"[CLASSIFICATION]: Processing {num_images} images in {int(np.ceil(num_images / self.config.batch_size))} batches")
        for i in tqdm(range(int(np.ceil(num_images / self.config.batch_size))), desc='[CLASSIFICATION]'):
            batch_x = self.gen.batch_generator(
                images=images, images_mean=image_mean, nr_images=num_images,
                batch_size=self.config.batch_size, index=epoch_step
            )
            epoch_step += self.config.batch_size
            
            pred = self.sess.run(self.y_pred, feed_dict={self.x_input: batch_x, self.keep_prob: 1})
            predictions = np.concatenate((predictions, pred), axis=0)
        
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
    
    def close(self):
        """Close the TensorFlow session to free resources."""
        if self.sess is not None:
            self.sess.close()
            self.sess = None 