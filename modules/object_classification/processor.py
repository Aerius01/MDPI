from pathlib import Path
from typing import Dict, Any
import pickle
import pandas as pd

class _ClassificationProcessor:
    def save_results(self, results: Dict[str, Any], output_path: Path, filename: str):
        """Save classification results to pickle file."""
        print('[CLASSIFICATION]: Creating pickle output...')
        
        with open(output_path / filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def assemble_classification_dataframe(self, results: Dict[str, Any], left_join_key: str) -> pd.DataFrame:
        """
        Prepares a DataFrame from the classification results.
        """
        if not results.get('samples'):
            return pd.DataFrame()

        new_data = []
        classes = results.get('classes', [])
        for sample in results['samples']:
            prediction_class_name = ''
            if 'y_predicted' in sample and classes:
                try:
                    prediction_class_name = classes[sample['y_predicted']]
                except IndexError:
                    prediction_class_name = 'unknown'

            path = sample.get('paths', '')
            file_name = Path(path).name

            # Hardcoded prediction = label columns as per Christian Dilewski's compromise with Jens Nejstgaard
            row = {
                left_join_key: file_name,
                'prediction': prediction_class_name,
                'label': prediction_class_name
            }
            new_data.append(row)

        return pd.DataFrame(new_data)

    def left_join_dataframes(self, new_df: pd.DataFrame, existing_df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Joins new classification results with an existing dataframe.
        It drops old 'prediction' and 'label' columns from the existing dataframe
        and performs a left join.
        """
        # If the prediction and label columns already exist, drop them.
        if 'prediction' in existing_df.columns:
            existing_df = existing_df.drop(columns=['prediction'])
        if 'label' in existing_df.columns:
            existing_df = existing_df.drop(columns=['label'])

        # Perform a left join to add the classification results, updating rows in new_df with data from existing_df.
        updated_df = pd.merge(new_df, existing_df, on=key, how='left')
        return updated_df