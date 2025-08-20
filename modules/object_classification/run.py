from pathlib import Path
from typing import Dict, Any
import pickle
import os
import pandas as pd
from .inference_engine import InferenceEngine
from .classification_data import ClassificationData

def save_results(results: Dict[str, Any], output_path: Path, filename: str):
    """Save classification results to pickle file."""
    print('[CLASSIFICATION]: Creating pickle output...')
    
    with open(output_path / filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def assemble_classification_dataframe(results: Dict[str, Any], left_join_key: str) -> pd.DataFrame:
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

def left_join_dataframes(new_df: pd.DataFrame, existing_df: pd.DataFrame, key: str) -> pd.DataFrame:
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

def save_csv_results(merged_df: pd.DataFrame, output_path: str, filename: str):
    """Save classification results to a CSV file, updating if file exists."""
    print('[CLASSIFICATION]: Creating CSV output...')
    
    file_path = os.path.join(output_path, filename)
    merged_df.to_csv(file_path, index=False)
    print(f"[CLASSIFICATION]: Updated existing CSV at {file_path}")

def run_classification(data: ClassificationData):
    """Main classification function for processing a single image group."""
    print('[CLASSIFICATION]: Starting vignettes classification...')
    
    # Initialize inference engine, build the model, and restore it from the checkpoint file.
    inference_engine = InferenceEngine(data)
    inference_engine.setup_model()

    try:
        results = inference_engine.process_location()
        
        # Save results to a .pkl file for use with the LabelChecker
        save_results(results, data.output_path, data.pkl_filename)

        # Assemble the classification dataframe and merge the detection dataframe into it, before saving as a .csv file
        classification_df = assemble_classification_dataframe(results, data.left_join_key)
        merged_df = left_join_dataframes(classification_df, data.detection_df, key=data.left_join_key)
        save_csv_results(merged_df, data.output_path, data.csv_filename)
        
    finally:
        inference_engine.close()
    
    print('[CLASSIFICATION]: Vignettes classification completed successfully!') 