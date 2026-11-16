# Object Classification Module

## Abstract

The Object Classification module follows the `object_detection` module. Its purpose is to take the vignettes (small, cropped images of detected objects) and assign a class label to each one. This is achieved using a pre-trained image classification model in ONNX format, run with `onnxruntime` on the CPU. The module processes vignettes in batches, runs them through the model for inference, and outputs the classification results. In addition to the primary class label, it also calculates several uncertainty metrics for each prediction, providing insight into the model's confidence. The final output is a comprehensive CSV file that merges the classification results with the object data from the detection phase.

## How it Works

The classification process is orchestrated through several interconnected components:

1.  **Data Ingestion and Validation:** The module begins by locating and loading the vignettes from the specified input directory.

2.  **Image Preprocessing:** Before being fed into the model, all vignettes undergo a standardized preprocessing routine. This involves the steps listed under "Preprocessing applied before the model" below, ending with a resize to the model's input size N x N. Pixel normalization is then applied as declared by the model's metadata.

3.  **Model Loading and Inference:** The module loads the ONNX model given with `-m/--model` (default `model/mdpi_4class_v1.onnx`, resolved from the repository root) and checks it against the model contract below before any processing starts. The inference engine then processes the preprocessed vignettes in batches for efficiency. For each vignette, the model outputs a vector of raw prediction scores (logits) for each possible class.

4.  **Prediction Post-processing:** The raw logits from the model are converted into probabilities using a softmax function. The class with the highest probability is chosen as the predicted label for the vignette.

5.  **Uncertainty Calculation:** To gauge the model's confidence, several uncertainty metrics are calculated from the probability distribution for each prediction:
    *   **Least Confidence:** Measures the difference between 1 and the highest prediction probability.
    *   **Margin Sampling:** Calculates the difference between the two highest prediction probabilities.
    *   **Entropy:** Measures the overall uncertainty in the probability distribution.

6.  **Data Aggregation and Output:** The classification results (predicted label, probabilities, and uncertainty metrics) are compiled into a new DataFrame. This DataFrame is then merged with the data that was produced by the prior object detection module, using the vignette filename as a key. The final, enriched dataset is saved to disk in CSV format. A pickle file containing the raw classification results is also saved for potential use with the LabelChecker.

## Module Outputs

-   **Updated Object Data CSV:** The module creates the `object_data.csv` file (located one level above the vignettes directory) that holds a large wealth of data for each of the objects.
-   **Classification Pickle File:** A file named `object_data.pkl` is saved in the same directory as the CSV. It contains a serialized dictionary with the raw classification results, including probabilities and uncertainty metrics, which can be used by the class viewer (see `class_viewer.py`). It also has a `model` key, `results['model'] = {'path': str, 'sha256': str}`, which records the model file used for the run. The model is also printed to the console. The CSV format is unchanged.
-   **Entropy fix (`HC_en`):** Earlier versions returned the entropies sorted in descending order while indexing them per vignette, so `HC_en` in older pickle files is misaligned with the vignettes. Entropy is now returned per sample, like `least_confidence` and `margin_sampling`, so `HC_en` values in new pickle files differ from those in old ones.

## The bundled model

`model/mdpi_4class_v1.onnx` is used when no `-m` is given, and always by the Streamlit GUI. It is a reference for the contract below:

| Property | Value |
|----------|-------|
| Input | `[batch, 50, 50]` (layout `BHW`, N = 50) |
| `mdpi.classes` | `["cladocera", "copepod", "junk", "rotifer"]` |
| `mdpi.normalization` | `run_mean` |

## Using your own model

Any ONNX model that meets the contract below can be used with `python3 run_pipeline.py -i <input_directory> -m my_model_mdpi.onnx`. Converting a model to ONNX is up to you; only the contract is documented here. The contract is checked during input validation, before depth profiling starts. A missing model file raises `FileNotFoundError`; every other violation (wrong file suffix, a file onnxruntime cannot load, or any contract rule) raises `ModelContractError` with a message naming the problem. Nothing is silently defaulted. The model file is read once at validation and its SHA-256 is recorded; classification refuses to run if the file changed in between.

### Preprocessing applied before the model

The pipeline preprocesses every vignette (`tools/preprocessing/image_preprocessing.py`) as follows:

1.  Load the image as grayscale.
2.  Otsu threshold (inverted binary), then find contours and keep the largest contour by area.
3.  Compute the principal axis of the largest contour with PCA and rotate the image upright.
4.  Resize to N x N with `cv2.resize` and its default interpolation, where N is the model's static input size.

The result is a uint8 N x N image. Train your model on vignettes preprocessed in exactly this way. The model then receives these images as float32 after the normalization declared in its metadata (see below).

### Input

Exactly 1 input, element type `tensor(float)` (float32). Allowed shapes:

-   `[B, N, N]`: layout `BHW`
-   `[B, N, N, 1]`: layout `BHWC`
-   `[B, 1, N, N]`: layout `BCHW`

`B` must be dynamic (a symbolic name or unset). `N` must be a static integer >= 2 and equal in both spatial positions. A rank-4 shape counts as `BHWC` only if `shape[3] == 1 and shape[1] != 1`, and as `BCHW` only if `shape[1] == 1 and shape[3] != 1`. Anything else is an error.

### Output

Exactly 1 output, element type `tensor(float)` (float32), like the input, with shape `[B, C]`, where `C` is a static integer equal to `len(mdpi.classes)`. These must be raw scores (logits), not probabilities. The pipeline applies softmax itself and computes the uncertainty metrics from the result.

### Metadata

The model describes itself through ONNX `metadata_props`. Input size and layout come from the graph's input shape.

| Key | Required | Value |
|-----|----------|-------|
| `mdpi.classes` | always | JSON list of at least 2 unique non-empty strings, in output-index order |
| `mdpi.normalization` | always | `run_mean`, `fixed` or `unit` |
| `mdpi.norm_mean` | only when `fixed` (error if present otherwise) | float, on the 0-1 scale |
| `mdpi.norm_std` | only when `fixed` (error if present otherwise) | float > 0, on the 0-1 scale |

Unknown keys are ignored.

### Normalization

Here `x` is the preprocessed uint8 N x N image.

-   `run_mean`: `(x - m) / 255`, where `m = np.mean(all preprocessed images of the run)`. The mean is computed in float64 and the result is cast to float32.
-   `fixed`: `(x / 255 - norm_mean) / norm_std`
-   `unit`: `x / 255`

### Stamping metadata onto a model

This needs the `onnx` package, which the pipeline itself does not require.

```python
import json
import onnx

model = onnx.load("my_model.onnx")
metadata = {
    "mdpi.classes": json.dumps(["copepod", "cladocera", "rotifer", "junk"]),
    "mdpi.normalization": "fixed",
    "mdpi.norm_mean": "0.5",
    "mdpi.norm_std": "0.25",
}
for key, value in metadata.items():
    prop = model.metadata_props.add()
    prop.key, prop.value = key, value
onnx.save(model, "my_model_mdpi.onnx")
```

### Checking a model

To see what the pipeline will read from a model, run this in `mdpi-env`:

```python
import onnxruntime as ort

session = ort.InferenceSession("my_model_mdpi.onnx")
print(session.get_modelmeta().custom_metadata_map)        # the mdpi.* keys
print(session.get_inputs()[0].shape, session.get_inputs()[0].type)
print(session.get_outputs()[0].shape, session.get_outputs()[0].type)
```

Running the pipeline is the real check: the model is validated before any processing starts, so a model that breaks the contract fails within seconds:

```bash
echo Y | python3 run_pipeline.py -i /data/mdpi/Lake_X/20230425/day/E01_01 -m my_model_mdpi.onnx
```

The pipeline then prints `[PIPELINE]: Using classification model ...` or `[PIPELINE]: Error: ...` with the rule that failed. Per-class concentrations and plots (`plots/<class>/`) are named after the model's `mdpi.classes`.

## Constants, Concerns, and Limitations

### Constants

The class names and the input size are not hardcoded. They are read from the model (`mdpi.classes` metadata and the input shape), as described under "Using your own model".

### Concerns and Limitations

-   **Model Dependency:** The module's accuracy is entirely dependent on the quality and suitability of the pre-trained model. A model trained on one type of data may not generalize well to another.
-   **Class Set Comes From the Model:** To classify new types of objects, retrain the model, export it to ONNX and list the classes in its `mdpi.classes` metadata. No code change is needed.
-   **"Label" vs. "Prediction":** The module saves the model's prediction under two columns, `prediction` and `label`. This is a legacy feature and might be confusing, as the `label` in this context is not a ground-truth value but simply a copy of the model's output.
