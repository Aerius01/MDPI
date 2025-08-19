#!/usr/bin/env python3
"""
Command line interface for object classification module.
Usage: python -m modules.object_classification [options]
"""

import argparse
import os
import sys
from pathlib import Path
from .inference_engine import InferenceEngine
from typing import Dict, Any
import pickle
import pandas as pd
from .validated_arguments import ValidatedArguments, validate_arguments
from modules.common.constants import CONSTANTS

# Destructure constants
DEFAULT_BATCH_SIZE = CONSTANTS.CLASSIFICATION_BATCH_SIZE
DEFAULT_INPUT_SIZE = CONSTANTS.CLASSIFICATION_INPUT_SIZE
DEFAULT_INPUT_DEPTH = CONSTANTS.CLASSIFICATION_INPUT_DEPTH

# Hardcoded constants
PKL_FILENAME = 'object_data.pkl'
LEFT_JOIN_KEY = 'FileName'

def save_results(results: Dict[str, Any], output_path: Path, filename: str):
    """Save classification results to pickle file."""
    print('[CLASSIFICATION]: Creating pickle output...')
    
    with open(output_path / filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def assemble_classification_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
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
            LEFT_JOIN_KEY: file_name,
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

def main(validated_args: ValidatedArguments):
    # Run classification
    """Main classification function for processing a single image group."""
    print('[CLASSIFICATION]: Starting vignettes classification...')
    
    # Initialize inference engine, build the model, and restore it from the checkpoint file.
    inference_engine = InferenceEngine(validated_args)
    inference_engine.setup_model()

    try:
        results = inference_engine.process_location()
        
        # Save results to a .pkl file for use with the LabelChecker
        save_results(results, validated_args.output_path, PKL_FILENAME)

        # Assemble the classification dataframe and merge the detection dataframe into it, before saving as a .csv file
        classification_df = assemble_classification_dataframe(results)
        merged_df = left_join_dataframes(classification_df, validated_args.detection_df, key=LEFT_JOIN_KEY)
        save_csv_results(merged_df, validated_args.output_path, validated_args.csv_filename)
        
    finally:
        inference_engine.close()
    
    print('[CLASSIFICATION]: Vignettes classification completed successfully!') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classify objects in vignette images using a trained CNN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model --batch-size 64
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model -o ./output --input-size 64
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing vignette images to classify. If following the default output path, this should be <output_directory>/project/date/cycle/location/vignettes')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the classified object .csv and .txt files. The full path for the saved files will be <output_directory>/project/date/cycle/location/classified_objects.csv')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to trained model directory containing the model.ckpt file')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for classification (default: 32)')
    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help='Input image size for the model (default: 50)')
    parser.add_argument('--input-depth', type=int, default=DEFAULT_INPUT_DEPTH,
                        help='Input image depth (channels) for the model (default: 1)')
    
    args = parser.parse_args()
    
    try:
        validated_args = validate_arguments(args)
        main(validated_args)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 