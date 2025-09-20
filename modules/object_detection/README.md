# Object Detection Module

## Abstract

The Object Detection module follows the `flatfielding` module. Its primary function is to identify and isolate potential objects of interest (including plankton) from the corrected images. It works by applying a series of image morphology techniques to distinguish objects from the background. Once identified, each object is cropped from the source image into a smaller, individual image called a "vignette." The module then extracts a comprehensive set of morphological and intensity-based features for each vignette, which are crucial for the subsequent classification stage.

## How it Works

The object detection process is executed through a series of sequential steps:

2.  **Image Binarization:** Each flatfielded image is converted into a binary (black and white) format using an inverted binary threshold. This step effectively separates the darker zones of the image from the lighter background.

3.  **Object Labeling and Initial Filtering:** The binary images are then processed to identify contiguous regions of pixels, which are treated as potential objects. An initial size-based filtering is applied to discard regions that are either too small or too large to be of interest.

4.  **Region Property Analysis and Filtering:** For the remaining regions, a detailed analysis is performed to calculate a variety of properties, including:
    *   **Morphological:** Area, major/minor axis lengths, eccentricity, orientation, solidity, etc.
    *   **Intensity:** Max, min, and mean pixel intensity.
    A second, more stringent filtering pass is applied based on these properties. Regions are discarded if they don't meet specific criteria for eccentricity, mean intensity, major axis length, and minimum intensity. This helps to eliminate artifacts and non-target objects.

5.  **Vignette Generation:** For each valid object, a vignette is created by cropping a small rectangular area around the object's centroid from the original, non-binary image. The size of the crop is dynamically padded based on the object's major axis length to ensure the entire object is captured.

6.  **Data Aggregation and Output:** The feature data for every detected object is compiled into a single pandas DataFrame. This DataFrame is then merged with the depth information and metadata. The final dataset is saved as a CSV file, and the generated vignettes are saved as individual image files.

## Constants, Concerns, and Limitations

### Constants

The object detection process is heavily controlled by a set of constants defined in `modules/object_detection/detection_data.py`. These constants are fine-tuned for a specific hardware and imaging environment and may require adjustment for different datasets.

-   **Thresholding:** These values control the initial image binarization step.
    -   `THRESHOLD_VALUE` (190): The pixel intensity cutoff. Pixels darker than this value are considered part of a potential object, while lighter pixels are treated as background.
    -   `THRESHOLD_MAX` (255): The value assigned to pixels that meet the threshold criteria, effectively making them white in the binary image.

-   **Size Filtering:** These constants define the valid size range for detected objects, measured in total pixel area.
    -   `MIN_OBJECT_SIZE` (75): The minimum number of pixels a region must have to be considered a potential object. This is crucial for filtering out image noise and other small artifacts.
    -   `MAX_OBJECT_SIZE` (5000): The maximum pixel area for a region to be considered.

-   **Property Filtering:** After initial detection, objects are further filtered based on their morphological and intensity properties.
    -   `MAX_ECCENTRICITY` (0.97): Filters based on shape. Eccentricity measures how elongated an object is (a value of 0 is a perfect circle, while a value near 1 is a line). This constant removes objects that are too long and thin.
    -   `MAX_MEAN_INTENSITY` (130): An object's average pixel brightness must be below this value. This helps to discard faint, low-contrast regions that are likely not of interest.
    -   `MIN_MAJOR_AXIS_LENGTH` (25): The minimum length (in pixels) of the object's longest axis. This provides another filter based on object size.
    -   `MAX_MIN_INTENSITY` (65): The darkest pixel within an object must be below this value. This ensures that only objects with sufficient contrast to the background are kept.

-   **Padding for Vignettes:** These constants determine the amount of space to include around a detected object when creating its vignette.
    -   `SMALL_OBJECT_THRESHOLD` (40) & `MEDIUM_OBJECT_THRESHOLD` (50): These thresholds, based on the object's major axis length, categorize objects as "small," "medium," or "large."
    -   `SMALL_OBJECT_PADDING` (25), `MEDIUM_OBJECT_PADDING` (30), `LARGE_OBJECT_PADDING` (40): The number of pixels to add as a border around an object when cropping the vignette. The amount of padding is selected based on the object's size category.

### Concerns and Limitations

-   **Parameter Sensitivity:** The module's performance is highly sensitive to the constants used for filtering. The default values are optimized for a specific type of plankton and imaging setup. Different organisms, water turbidity, or camera settings will likely require significant tuning of these parameters to achieve optimal results.
-   **No Model-Based Detection:** This module uses traditional computer vision techniques (thresholding, contour analysis) rather than a trained machine learning model. This makes it fast and interpretable but potentially less robust at handling variations in object appearance or complex backgrounds compared to model-based approaches like YOLO or Faster R-CNN.
-   **Overlapping Objects:** The current implementation may struggle to correctly segment objects that are touching or overlapping in the original image, potentially treating them as a single, larger object.
