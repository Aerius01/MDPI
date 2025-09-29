import os
import numpy as np
from typing import List, Dict, Any

# Custom modules located within the object_classification module
from .architecture import _build_model
from .classification_data import ClassificationData

# Custom modules located externally from the object_classification module
from tools.nn.soft_max import soft_max
from tools.io.batch_generator import batch_generator
from tools.preprocessing.image_preprocessing import image_preprocessing
from tools.metrics.entropy import entropy
from tools.metrics.least_confidence import least_confidence
from tools.metrics.margin_sampling import margin_sampling

# Suppress TensorFlow logging messages - MUST be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

try:
    from absl import logging
    logging.set_verbosity(logging.ERROR)
except ImportError:
    pass

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class _InferenceEngine:
    """Handles model inference and prediction pipeline."""
    
    def __init__(self, validated_args: ClassificationData):
        self.validated_args = validated_args
        self.session = None

        # Reset the default graph and close the session
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        if tf.compat.v1.get_default_session() is not None:
            tf.compat.v1.get_default_session().close()
    
    def setup_model(self):
        """Setup TensorFlow session and model."""
        self.session = tf.compat.v1.Session()
        
        print('[CLASSIFICATION]: Constructing classification model...')
        self.x_input, self.keep_prob, self.y_pred = _build_model(self.validated_args.input_size, self.validated_args.input_depth, len(self.validated_args.categories))
        
        print('[CLASSIFICATION]: Restoring classification model...')
        self.session.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.session, os.path.sep.join([self.validated_args.model_path, "model.ckpt"]))
    
    def process_location(self) -> Dict[str, Any]:
        """Process all images in the provided list and return classification results."""
        
        # Preprocess images
        angles, images, image_mean = image_preprocessing(self.validated_args.vignette_paths, self.validated_args.input_size)
        
        # Run batch predictions
        predictions = self._run_batch_predictions(images, image_mean, len(self.validated_args.vignette_paths))
        
        # Post-process predictions
        return self._post_process_predictions(predictions, self.validated_args.vignette_paths, angles)
    
    def _run_batch_predictions(self, images: np.ndarray, image_mean: float, num_images: int) -> np.ndarray:
        """Run batch predictions on preprocessed images."""
        predictions = np.array([], dtype=np.float32).reshape(0, len(self.validated_args.categories))
        epoch_step = 0
        num_batches = int(np.ceil(num_images / self.validated_args.batch_size))
        
        print('[CLASSIFICATION]: Starting vignettes classification...')
        print(f"[CLASSIFICATION]: Processing {num_images} images in {num_batches} batches")
        
        for i in range(num_batches):
            batch_x = batch_generator(
                images=images, images_mean=image_mean, nr_images=num_images,
                batch_size=self.validated_args.batch_size, index=epoch_step
            )
            epoch_step += self.validated_args.batch_size
            
            pred = self.session.run(self.y_pred, feed_dict={self.x_input: batch_x, self.keep_prob: 1})
            predictions = np.concatenate((predictions, pred), axis=0)

            progress = min(epoch_step, num_images)
            print(f"[PROGRESS] {progress}/{num_images}")
        
        # Hardcode a final, completed progress bar to bypass async issues
        bar = f"[{'#' * 40}]"
        total_str = str(num_images)
        progress_bar = f"[CLASSIFICATION]: {bar} 100% | {total_str}/{total_str}"
        print(progress_bar, flush=True)

        return predictions
    
    def _post_process_predictions(self, predictions: np.ndarray, vignette_paths: List[str], angles: List[float]) -> Dict[str, Any]:
        """Post-process predictions and calculate uncertainty metrics."""
        predictions = soft_max(predictions)
        
        # Calculate uncertainty metrics
        _, max_y = least_confidence(predictions)
        _, diff = margin_sampling(predictions)
        _, ent = entropy(predictions)
        
        cat_predictions = np.argmax(predictions, axis=1).tolist()
        
        # Format output
        samples = []
        for i, image_path in enumerate(vignette_paths):
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
            'classes': self.validated_args.categories,
            'samples': samples
        }
    
    def close(self):
        """Close the TensorFlow session to free resources."""
        if self.session is not None:
            self.session.close()
            self.session = None 