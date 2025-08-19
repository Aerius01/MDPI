#!/usr/bin/env python3
"""
Command line interface for object classification module.
Usage: python -m modules.object_classification [options]
"""

import argparse
import os
import sys
from pathlib import Path
from modules.common.cli_utils import CommonCLI
from modules.common.parser import parse_vignette_metadata
from .inference_engine import InferenceEngine
from typing import Dict, Any
import pickle
import pandas as pd
from .validated_arguments import ValidatedArguments
from modules.common.constants import CONSTANTS

# Destructure constants
DEFAULT_BATCH_SIZE = CONSTANTS.CLASSIFICATION_BATCH_SIZE
DEFAULT_INPUT_SIZE = CONSTANTS.CLASSIFICATION_INPUT_SIZE
DEFAULT_INPUT_DEPTH = CONSTANTS.CLASSIFICATION_INPUT_DEPTH
DEFAULT_CATEGORIES = CONSTANTS.CLASSIFICATION_CATEGORIES
CSV_FILENAME = 'object_data.csv'
PKL_FILENAME = 'object_data.pkl'

def process_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Validate paths
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    
    model_checkpoint = os.path.join(args.model, "model.ckpt")
    if not any(os.path.exists(f"{model_checkpoint}.{ext}") for ext in ['meta', 'index', 'data-00000-of-00001']):
        raise FileNotFoundError(f"Model checkpoint files not found in: {args.model}")
    
    # Get image paths from input directory
    print(f"[CLASSIFICATION]: Loading images from {args.input}")
    image_paths = CommonCLI.get_image_group_from_folder(args.input)
    print(f"[CLASSIFICATION]: Found {len(image_paths)} images")
    
    input_path = Path(args.input)
    metadata = parse_vignette_metadata(input_path)

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"])
    os.makedirs(output_path, exist_ok=True)

    # The detection CSV file must exist and be non-empty.
    csv_path = os.path.join(output_path, CSV_FILENAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Detection CSV file not found at {csv_path}. It must be created by the object detection module first.")
    
    try:
        detection_df = pd.read_csv(csv_path, sep=None, engine='python')
        if detection_df.empty:
            raise ValueError(f"Detection CSV file is empty: {csv_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Detection CSV file is empty: {csv_path}") from None

    return ValidatedArguments(
        image_paths=image_paths,
        output_path=output_path,
        model_path=args.model,
        metadata=metadata,
        batch_size=args.batch_size,
        input_size=args.input_size,
        input_depth=args.input_depth,
        categories=DEFAULT_CATEGORIES,
        detection_df=detection_df
    )

def save_results(results: Dict[str, Any], output_path: str, filename: str):
    """Save classification results to pickle file."""
    print('[CLASSIFICATION]: Creating pickle output...')
    
    with open(os.path.sep.join([output_path, filename]), 'wb') as handle:
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
            'FileName': file_name,
            'prediction': prediction_class_name,
            'label': prediction_class_name
        }
        new_data.append(row)

    return pd.DataFrame(new_data)

def left_join_dataframes(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
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
    updated_df = pd.merge(new_df, existing_df, on='FileName', how='left')
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
        merged_df = left_join_dataframes(classification_df, validated_args.detection_df)
        save_csv_results(merged_df, validated_args.output_path, CSV_FILENAME)
        
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
        validated_args = process_arguments(args)
        main(validated_args)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 