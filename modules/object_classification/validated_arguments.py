from dataclasses import dataclass

@dataclass
class ValidatedArguments:
    image_paths: list
    output_path: str
    model_path: str
    metadata: dict
    batch_size: int
    input_size: int
    input_depth: int
    categories: list
    detection_df: 'pd.DataFrame' 