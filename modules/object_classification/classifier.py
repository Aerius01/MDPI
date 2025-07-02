import os
from typing import List

# Suppress TensorFlow logging messages - MUST be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .config import ClassificationConfig
from .processor import SingleLocationProcessor

def classify_objects(image_group: List[str], output_path: str, model_path: str, 
                    batch_size: int = 32, input_size: int = 50, input_depth: int = 1):
    """Main classification function for processing a single image group.
    
    Args:
        image_group (List[str]): List of paths to individual image files to classify
        output_path (str): Directory where classification results will be saved
        model_path (str): Path to the trained model checkpoint
        batch_size (int): Batch size for processing images
        input_size (int): Size of input images (width=height)
        input_depth (int): Number of channels in input images (1 for grayscale)
    
    Returns:
        dict: Classification results containing predictions and uncertainty metrics
    """
    print('[CLASSIFICATION]: Starting vignettes classification...')
    
    # Reset TensorFlow state to ensure clean slate between calls
    tf.compat.v1.reset_default_graph()
    if tf.compat.v1.get_default_session() is not None:
        tf.compat.v1.get_default_session().close()
    
    config = ClassificationConfig(
        image_group=image_group,
        model_path=model_path,
        batch_size=batch_size,
        input_size=input_size,
        input_depth=input_depth,
        output_path=output_path
    )
    
    processor = SingleLocationProcessor(config)
    processor.process_location()
    
    print('[CLASSIFICATION]: Vignettes classification completed successfully!') 