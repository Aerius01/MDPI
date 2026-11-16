from typing import Any, TypedDict

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Custom modules located within the object_classification module
from .classification_data import ClassificationData
from .model_contract import ClassifierModel, InputLayout, Normalization

# Custom modules located externally from the object_classification module
from tools.nn.soft_max import soft_max
from tools.preprocessing.image_preprocessing import image_preprocessing
from tools.metrics.entropy import entropy
from tools.metrics.least_confidence import least_confidence
from tools.metrics.margin_sampling import margin_sampling

_PIXEL_SCALE: float = 255.0


class ModelRecord(TypedDict):
    path: str
    sha256: str


class ClassificationResults(TypedDict):
    classes: list[str]
    samples: list[dict[str, Any]]
    model: ModelRecord


def normalize_batch(images: list[np.ndarray], images_mean: np.float64, model: ClassifierModel) -> np.ndarray:
    """Normalize uint8 images as the model declares, in float64, and return float32 (B, N, N)."""
    pixels = np.asarray(images, dtype=np.float64)
    match model.normalization:
        case Normalization.RUN_MEAN:
            normalized = (pixels - images_mean) / _PIXEL_SCALE
        case Normalization.FIXED:
            normalized = (pixels / _PIXEL_SCALE - model.norm_mean) / model.norm_std
        case Normalization.UNIT:
            normalized = pixels / _PIXEL_SCALE
    return normalized.astype(np.float32)


def to_layout(batch: np.ndarray, layout: InputLayout) -> np.ndarray:
    """Reshape a (B, N, N) batch to the model's input layout."""
    match layout:
        case InputLayout.BHW:
            return batch
        case InputLayout.BHWC:
            return batch[:, :, :, np.newaxis]
        case InputLayout.BCHW:
            return batch[:, np.newaxis, :, :]


def predict_logits(
    session: ort.InferenceSession,
    model: ClassifierModel,
    images: list[np.ndarray],
    images_mean: np.float64,
    batch_size: int,
) -> np.ndarray:
    """Run the model over all images in batches and return the concatenated logits (n, C)."""
    num_batches = int(np.ceil(len(images) / batch_size))
    print(f"[CLASSIFICATION]: Processing {len(images)} images in {num_batches} batches")
    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(images), batch_size), desc='[CLASSIFICATION]'):
        batch = normalize_batch(images[start:start + batch_size], images_mean, model)
        batches.append(session.run(None, {model.input_name: to_layout(batch, model.layout)})[0])
    return np.concatenate(batches, axis=0).astype(np.float32)


def summarize_predictions(
    logits: np.ndarray, vignette_paths: list[str], angles: list[float], model: ClassifierModel
) -> ClassificationResults:
    """Apply softmax and the uncertainty metrics, and assemble the results dict saved to the pkl."""
    predictions = soft_max(logits.copy())

    _, max_y = least_confidence(predictions)
    _, diff = margin_sampling(predictions)
    _, ent = entropy(predictions)
    cat_predictions = np.argmax(predictions, axis=1).tolist()

    samples = [
        {
            'paths': image_path,
            'y_predicted': cat_predictions[i],
            'full_y_predicted': predictions[i, :],
            'HC_lc': max_y[i],
            'HC_ms': diff[i],
            'HC_en': ent[i],
            'orientation': angles[i],
        }
        for i, image_path in enumerate(vignette_paths)
    ]
    return {
        'classes': list(model.classes),
        'samples': samples,
        'model': {'path': str(model.path), 'sha256': model.sha256},
    }


def classify_vignettes(data: ClassificationData) -> ClassificationResults:
    """Preprocess and classify all vignettes with the ONNX model."""
    model = data.classifier_model
    print(f"[CLASSIFICATION]: Using model {model.path.name} (sha256 {model.sha256[:12]})")
    session = ort.InferenceSession(str(model.path), providers=["CPUExecutionProvider"])

    angles, images, images_mean = image_preprocessing(data.vignette_paths, model.input_size)
    logits = predict_logits(session, model, images, images_mean, data.batch_size)
    return summarize_predictions(logits, data.vignette_paths, angles, model)
