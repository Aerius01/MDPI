import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import csvParser from 'csv-parser';
import { createReadStream } from 'fs';

// Image extensions to look for
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.tiff']);

// CSV format detection constants
const OLD_FORMAT_FIRST_COLUMN = 'Name';
const NEW_FORMAT_FIRST_COLUMN = 'Number';

/**
 * Quick validation for UI feedback - checks basic requirements without full validation
 * @param {string} folderPath - Path to the input folder
 * @returns {Promise<Object>} Validation result with metadata
 */
export async function quickValidate(folderPath) {
    const result = {
        valid: true,
        errors: [],
        warnings: [],
        metadata: {
            recordingStart: null,
            imageShape: null,
            cameraFormat: null,
            imageCount: 0,
            csvCount: 0
        }
    };

    try {
        // 1. Check folder exists and is readable
        const stats = await fs.stat(folderPath);
        if (!stats.isDirectory()) {
            result.valid = false;
            result.errors.push('Path must be a directory.');
            return result;
        }
    } catch (error) {
        result.valid = false;
        result.errors.push(`Cannot access directory: ${error.message}`);
        return result;
    }

    try {
        // 2. Read directory contents
        const files = await fs.readdir(folderPath);

        // 3. Count and find image files
        const imageFiles = files.filter(f => {
            const ext = path.extname(f).toLowerCase();
            return IMAGE_EXTENSIONS.has(ext);
        }).sort();

        result.metadata.imageCount = imageFiles.length;

        if (imageFiles.length === 0) {
            result.valid = false;
            result.errors.push('No image files found in directory. Expected files with extensions: .png, .jpg, .jpeg, .tiff');
            return result;
        }

        // 4. Count CSV files
        const csvFiles = files.filter(f => f.toLowerCase().endsWith('.csv'));
        result.metadata.csvCount = csvFiles.length;

        if (csvFiles.length === 0) {
            result.valid = false;
            result.errors.push('No CSV file found in directory. Each input folder must contain exactly one pressure sensor CSV file.');
            return result;
        }

        if (csvFiles.length > 1) {
            result.valid = false;
            result.errors.push(`Multiple CSV files found: ${csvFiles.join(', ')}. Each input folder must contain exactly one pressure sensor CSV file.`);
            return result;
        }

        // 5. Extract recording start from last image filename
        try {
            const lastImage = imageFiles[imageFiles.length - 1];
            const recordingStart = parseRecordingStartFromFilename(lastImage);
            result.metadata.recordingStart = recordingStart;
        } catch (error) {
            result.valid = false;
            result.errors.push(error.message);
            return result;
        }

        // 6. Read dimensions from first image
        try {
            const firstImagePath = path.join(folderPath, imageFiles[0]);
            const imageMetadata = await sharp(firstImagePath).metadata();
            result.metadata.imageShape = {
                width: imageMetadata.width,
                height: imageMetadata.height
            };

            if (!imageMetadata.width || !imageMetadata.height || imageMetadata.width <= 0 || imageMetadata.height <= 0) {
                result.valid = false;
                result.errors.push('Image dimensions must be greater than zero.');
                return result;
            }
        } catch (error) {
            result.valid = false;
            result.errors.push(`Could not read image dimensions: ${error.message}`);
            return result;
        }

        // 7. Parse CSV to detect camera format
        try {
            const csvPath = path.join(folderPath, csvFiles[0]);
            const cameraFormat = await detectCameraFormat(csvPath);
            result.metadata.cameraFormat = cameraFormat;
        } catch (error) {
            result.warnings.push(`Could not determine camera format: ${error.message}`);
            // Don't fail validation for this - Python will handle it
        }

    } catch (error) {
        result.valid = false;
        result.errors.push(`Validation error: ${error.message}`);
    }

    return result;
}

/**
 * Parse recording start datetime from image filename
 * Expected format: *_YYYYmmdd_HHMMSSfff_<replicate>.ext
 * @param {string} filename - Image filename
 * @returns {string} ISO datetime string
 */
function parseRecordingStartFromFilename(filename) {
    const baseName = path.parse(filename).name;
    const parts = baseName.split('_');

    if (parts.length < 3) {
        throw new Error(
            `Filename '${filename}' does not match expected format. ` +
            `Expected format: prefix_YYYYmmdd_HHMMSSfff_replicate.ext ` +
            `(e.g., MDPI_20240315_143052123_001.jpg)`
        );
    }

    // Extract date and time from filename
    const dateStr = parts[parts.length - 3];
    const timeStr = parts[parts.length - 2];

    // Validate date format (YYYYMMDD)
    if (dateStr.length !== 8 || !/^\d{8}$/.test(dateStr)) {
        throw new Error(
            `Date part '${dateStr}' in filename '${filename}' is invalid. ` +
            `Expected 8 digits in YYYYmmdd format (e.g., 20240315)`
        );
    }

    // Validate time format (HHMMSSfff)
    if (timeStr.length !== 9 || !/^\d{9}$/.test(timeStr)) {
        throw new Error(
            `Time part '${timeStr}' in filename '${filename}' is invalid. ` +
            `Expected 9 digits in HHMMSSfff format (e.g., 143052123)`
        );
    }

    // Parse date
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);

    // Parse time
    const hours = timeStr.substring(0, 2);
    const minutes = timeStr.substring(2, 4);
    const seconds = timeStr.substring(4, 6);
    const milliseconds = timeStr.substring(6, 9);

    // Construct ISO datetime string
    const isoString = `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${milliseconds}`;

    // Validate it's a valid date
    const date = new Date(isoString);
    if (isNaN(date.getTime())) {
        throw new Error(
            `Invalid date/time extracted from filename '${filename}': ${isoString}`
        );
    }

    return isoString;
}

/**
 * Detect camera format (old vs new) by reading CSV headers
 * @param {string} csvPath - Path to CSV file
 * @returns {Promise<string>} 'old' or 'new'
 */
function detectCameraFormat(csvPath) {
    return new Promise((resolve, reject) => {
        let headerDetected = false;

        createReadStream(csvPath)
            .pipe(csvParser({ separator: ';' }))
            .on('headers', (headers) => {
                if (headers && headers.length > 0) {
                    const firstColumn = headers[0];

                    if (firstColumn === OLD_FORMAT_FIRST_COLUMN) {
                        headerDetected = true;
                        resolve('old');
                    } else if (firstColumn === NEW_FORMAT_FIRST_COLUMN) {
                        headerDetected = true;
                        resolve('new');
                    } else {
                        reject(new Error(
                            `Could not determine camera format. ` +
                            `First column is '${firstColumn}', expected '${OLD_FORMAT_FIRST_COLUMN}' or '${NEW_FORMAT_FIRST_COLUMN}'`
                        ));
                    }
                }
            })
            .on('error', (error) => {
                reject(error);
            })
            .on('end', () => {
                if (!headerDetected) {
                    reject(new Error('Could not read CSV headers'));
                }
            });
    });
}
