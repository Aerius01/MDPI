from .inference_engine import InferenceEngine
from .classification_data import ClassificationData
from .processor import ClassificationProcessor

def run_classification(
    data: ClassificationData,
    inference_engine: InferenceEngine,
    processor: ClassificationProcessor,
):
    """Main classification function for processing a single image group."""
    print('[CLASSIFICATION]: Starting vignettes classification...')
    
    inference_engine.setup_model()

    try:
        # Classify the vignettes
        results = inference_engine.process_location()
        
        # Save results to a .pkl file for use with the LabelChecker
        processor.save_results(results, data.output_path, data.pkl_filename)

        # Assemble the classification dataframe and merge the detection dataframe into it, before saving as a .csv file
        classification_df = processor.assemble_classification_dataframe(results, data.left_join_key)
        merged_df = processor.left_join_dataframes(classification_df, data.detection_df, key=data.left_join_key)
        processor.save_csv_results(merged_df, data.output_path, data.csv_filename)
        
    finally:
        inference_engine.close()
    
    print('[CLASSIFICATION]: Vignettes classification completed successfully!') 