import os
import pickle
from typing import Dict, Any

class OutputHandler:
    """Handles output file generation."""
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str, filename: str):
        """Save classification results to pickle file."""
        print('[CLASSIFICATION]: Creating pickle output...')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with open(os.path.sep.join([output_path, filename]), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL) 