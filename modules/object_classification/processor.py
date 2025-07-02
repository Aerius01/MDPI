import os
from .config import ClassificationConfig
from .inference_engine import InferenceEngine
from .output_handler import OutputHandler

class SingleLocationProcessor:
    """Handles processing of a single location directory."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.inference_engine = InferenceEngine(config)
        self.output_handler = OutputHandler()
    
    def process_location(self):
        """Process the single location specified in the config."""
        print(f'[CLASSIFICATION]: Processing {len(self.config.image_group)} images')
        
        # Check if image group has any paths
        if not self.config.image_group:
            raise ValueError("Image group is empty - no images to process")
        
        # Check if all image paths exist
        for image_path in self.config.image_group:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
        
        try:
            # Process the images
            results = self.inference_engine.process_location(self.config.image_group)
            
            # Determine output path and filename - get last 4 directories and filename
            path_parts = self.config.image_group[0].split(os.path.sep)
            project, date, time, location = path_parts[-5:-1]  # Last 4 directories
            filename = f'{project}_{date}_{time}_{location}_classification.pkl'
            
            self.output_handler.save_results(results, self.config.output_path, filename)
            
            return results
        finally:
            # Always close the session to free resources
            self.inference_engine.close() 