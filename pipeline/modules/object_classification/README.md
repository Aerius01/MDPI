# Object Classification Module

## Abstract

The Object Classification module follows the `object_detection` module. Its purpose is to take the vignettes (small, cropped images of detected objects) and assign a class label to each one. This is achieved using a pre-trained Convolutional Neural Network (CNN). The module processes vignettes in batches, runs them through the CNN for inference, and outputs the classification results. In addition to the primary class label, it also calculates several uncertainty metrics for each prediction, providing insight into the model's confidence. The final output is a comprehensive CSV file that merges the classification results with the object data from the detection phase.

## How it Works

The classification process is orchestrated through several interconnected components:

1.  **Data Ingestion and Validation:** The module begins by locating and loading the vignettes from the specified input directory.

2.  **Image Preprocessing:** Before being fed into the CNN, all vignettes undergo a standardized preprocessing routine. This involves resizing each image to the model's expected input size (e.g., 50x50 pixels) and normalizing the pixel values.

3.  **Model Loading and Inference:** The module loads the pre-trained TensorFlow model from the specified checkpoint files. The inference engine then processes the preprocessed vignettes in batches for efficiency. For each vignette, the model outputs a vector of raw prediction scores (logits) for each possible class.

4.  **Prediction Post-processing:** The raw logits from the model are converted into probabilities using a softmax function. The class with the highest probability is chosen as the predicted label for the vignette.

5.  **Uncertainty Calculation:** To gauge the model's confidence, several uncertainty metrics are calculated from the probability distribution for each prediction:
    *   **Least Confidence:** Measures the difference between 1 and the highest prediction probability.
    *   **Margin Sampling:** Calculates the difference between the two highest prediction probabilities.
    *   **Entropy:** Measures the overall uncertainty in the probability distribution.

6.  **Data Aggregation and Output:** The classification results (predicted label, probabilities, and uncertainty metrics) are compiled into a new DataFrame. This DataFrame is then merged with the data that was produced by the prior object detection module, using the vignette filename as a key. The final, enriched dataset is saved to disk in CSV format. A pickle file containing the raw classification results is also saved for potential use with the LabelChecker.

## Module Outputs

-   **Updated Object Data CSV:** The module creates the `object_data.csv` file (located one level above the vignettes directory) that holds a large wealth of data for each of the objects.
-   **Classification Pickle File:** A file named `object_data.pkl` is saved in the same directory as the CSV. It contains a serialized dictionary with the raw classification results, including probabilities and uncertainty metrics, which can be used by the class viewer (see `class_viewer.py`).

## Constants, Concerns, and Limitations

### Constants

-   **`CLASSIFICATION_CATEGORIES`:** A list of class names that the model can predict (e.g., `['cladocera', 'copepod', 'junk', 'rotifer']`). This list must match the classes the model was trained on.
-   **Model Input Dimensions:** `CLASSIFICATION_INPUT_SIZE` and `CLASSIFICATION_INPUT_DEPTH` define the expected dimensions of the input vignettes for the CNN. These must align with the architecture of the loaded model.

### Concerns and Limitations

-   **Model Dependency:** The module's accuracy is entirely dependent on the quality and suitability of the pre-trained CNN model. A model trained on one type of data may not generalize well to another.
-   **TensorFlow 1.x:** The current implementation uses TensorFlow 1.x syntax. This is an older version of the framework, and the code may require significant updates to be compatible with modern TensorFlow 2.x environments.
-   **Fixed Class Set:** The set of possible classification categories is hardcoded. To classify new types of objects, the model must be retrained, and the `CLASSIFICATION_CATEGORIES` constant in the code must be updated.
-   **"Label" vs. "Prediction":** The module saves the model's prediction under two columns, `prediction` and `label`. This is a legacy feature and might be confusing, as the `label` in this context is not a ground-truth value but simply a copy of the model's output.
