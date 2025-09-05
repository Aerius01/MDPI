# Duplicate Image Detection Module

This module is designed to identify and remove duplicate images from a specified directory. It serves as a pre-processing step in the MDPI pipeline to ensure that the dataset does not contain redundant images, skewing results.

## How it Works

The duplicate detection process is based on **perceptual hashing**. Unlike cryptographic hashes, which are sensitive to even minor changes in the input, perceptual hashes are designed to be similar for images that look alike to the human eye.

This module uses the **Difference Hash (`dhash`)** algorithm, which involves the following steps:

1.  **Grayscale Conversion**: The image is converted to grayscale to simplify the hashing process.
2.  **Resizing**: The grayscale image is downscaled to a small, fixed size (typically 9x8 pixels). This step helps to normalize the image and make the hash robust to changes in aspect ratio and resolution.
3.  **Difference Calculation**: The algorithm computes the difference between adjacent pixels, resulting in a binary hash that represents the image's gradient.
4.  **Hash Comparison**: The generated hashes are then compared to identify duplicate images.

Images with the same perceptual hash are considered duplicates. The module will keep the first encountered image and delete all subsequent duplicates.

## Usage

The module can be run as a standalone script from the command line:

```bash
python -m modules.duplicate_detection -i /path/to/your/image/directory
```

Replace `/path/to/your/image/directory` with the actual path to the directory containing your images.

## Important Considerations

*   **Destructive Operation**: This script permanently deletes files from your system.
*   **Hashing Limitations**: While perceptual hashing is effective at identifying identical or near-identical images, it may not catch duplicates that have been significantly altered (e.g., through heavy color correction, cropping, or rotation).
