import os
import pickle
import csv
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

    @staticmethod
    def save_csv_results(results: Dict[str, Any], output_path: str, filename: str):
        """Save classification results to a CSV file."""
        print('[CLASSIFICATION]: Creating CSV output...')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fieldnames = ['path', 'prediction', 'label']
        
        with open(os.path.join(output_path, filename), 'w', newline='') as fd:
            dict_writer = csv.DictWriter(fd, fieldnames=fieldnames)
            dict_writer.writeheader()
            
            if not results.get('samples'):
                print("no data to export available!")
                return

            classes = results.get('classes', [])
            for sample in results['samples']:
                prediction_class_name = ''
                if 'y_predicted' in sample and classes:
                    try:
                        prediction_class_name = classes[sample['y_predicted']]
                    except IndexError:
                        prediction_class_name = 'unknown'

                row = {
                    'path': sample.get('paths', ''),
                    'prediction': prediction_class_name,
                    'label': prediction_class_name
                }
                dict_writer.writerow(row) 