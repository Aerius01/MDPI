import os
from types import SimpleNamespace

from .inference_engine import _InferenceEngine
from .classification_data import _validate_arguments
from .processor import _ClassificationProcessor
import pandas as pd

def run_classification(
    run_config: SimpleNamespace,
    object_data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Main classification function for processing a single image group."""
    print('[CLASSIFICATION]: Starting vignettes classification...')
    
    classification_data = _validate_arguments(
        run_config,
        object_data_df,
    )

    inference_engine = _InferenceEngine(classification_data)
    processor = _ClassificationProcessor()
    
    inference_engine.setup_model()

    try:
        # Classify the vignettes
        results = inference_engine.process_location()
        
        # Save results to a .pkl file for use with the LabelChecker
        processor.save_results(results, classification_data.output_path, classification_data.pkl_filename)

        # Assemble the classification dataframe and merge the detection dataframe into it
        classification_df = processor.assemble_classification_dataframe(results, classification_data.left_join_key)
        merged_df = processor.left_join_dataframes(classification_df, classification_data.detection_df, key=classification_data.left_join_key)
        
    finally:
        inference_engine.close()

    output_csv_path = os.path.join(classification_data.output_path, classification_data.csv_filename)

    # Add metadata to the dataframe and save
    merged_df['recording_start_date'] = classification_data.recording_start_date
    merged_df.to_csv(output_csv_path, index=False, sep=classification_data.csv_separator)
    print(f"[CLASSIFICATION]: Final classified data with metadata saved to {output_csv_path}")
    
    return merged_df 