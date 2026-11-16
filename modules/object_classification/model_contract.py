"""Loader and validator for the MDPI ONNX classifier model contract."""

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

import onnxruntime as ort

DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "model" / "mdpi_4class_v1.onnx"

_KEY_CLASSES = "mdpi.classes"
_KEY_NORMALIZATION = "mdpi.normalization"
_KEY_NORM_MEAN = "mdpi.norm_mean"
_KEY_NORM_STD = "mdpi.norm_std"
_FLOAT_TYPE = "tensor(float)"
_HASH_CHUNK_BYTES = 1 << 20
_INPUT_EXPECTED = "[B, N, N], [B, N, N, 1] or [B, 1, N, N] with dynamic B and static N >= 2"

Dim: TypeAlias = int | str | None


class InputLayout(StrEnum):
    BHW = "BHW"
    BHWC = "BHWC"
    BCHW = "BCHW"


class Normalization(StrEnum):
    RUN_MEAN = "run_mean"
    FIXED = "fixed"
    UNIT = "unit"


class ModelContractError(ValueError):
    """The ONNX file does not satisfy the MDPI model contract."""


@dataclass(frozen=True)
class ClassifierModel:
    path: Path
    sha256: str
    input_name: str
    input_size: int
    layout: InputLayout
    classes: tuple[str, ...]
    normalization: Normalization
    norm_mean: float | None
    norm_std: float | None


_METADATA_HELP = (
    "Valid metadata keys and values: "
    f"'{_KEY_CLASSES}' (always; JSON list of at least 2 unique non-empty strings in output-index order), "
    f"'{_KEY_NORMALIZATION}' (always; one of {', '.join(Normalization)}), "
    f"'{_KEY_NORM_MEAN}' (only with 'fixed'; float on the 0-1 scale), "
    f"'{_KEY_NORM_STD}' (only with 'fixed'; float > 0 on the 0-1 scale)."
)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(path: Path, message: str) -> ModelContractError:
    return ModelContractError(f"Model '{path}': {message}")


def _is_static(dim: Dim) -> bool:
    return isinstance(dim, int)


def _infer_input(path: Path, shape: list[Dim]) -> tuple[int, InputLayout]:
    """Return (N, layout) from the input shape, or raise."""
    rank = len(shape)
    if rank not in (3, 4):
        raise _fail(path, f"input must have rank 3 or 4, found shape {shape}. Expected {_INPUT_EXPECTED}. Re-export the model with a supported input shape.")
    if _is_static(shape[0]):
        raise _fail(path, f"input batch dimension must be dynamic, found static size {shape[0]} in shape {shape}. Re-export with a dynamic batch axis (e.g. dynamic_axes in torch.onnx.export).")

    if rank == 3:
        layout = InputLayout.BHW
        height, width = shape[1], shape[2]
    elif shape[3] == 1 and shape[1] != 1:
        layout = InputLayout.BHWC
        height, width = shape[1], shape[2]
    elif shape[1] == 1 and shape[3] != 1:
        layout = InputLayout.BCHW
        height, width = shape[2], shape[3]
    else:
        raise _fail(path, f"cannot infer the input layout from rank-4 shape {shape} (single-channel grayscale only). Expected {_INPUT_EXPECTED}.")

    if not (isinstance(height, int) and isinstance(width, int)):
        raise _fail(path, f"input spatial dimensions must be static ints, found {height!r} and {width!r} in shape {shape}. Re-export with a fixed image size.")
    if height != width:
        raise _fail(path, f"input must be square, found {height} x {width} in shape {shape}. Re-export with equal height and width.")
    if height < 2:
        raise _fail(path, f"input size N must be >= 2, found {height}. Re-export with a larger image size.")
    return height, layout


def _parse_classes(path: Path, raw: str | None) -> tuple[str, ...]:
    if raw is None:
        raise _fail(path, f"missing required metadata key '{_KEY_CLASSES}'. {_METADATA_HELP}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise _fail(path, f"'{_KEY_CLASSES}' is not valid JSON (found {raw!r}: {error}). {_METADATA_HELP}") from error
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise _fail(path, f"'{_KEY_CLASSES}' must be a JSON list of non-empty strings, found {raw!r}. {_METADATA_HELP}")
    if len(value) < 2:
        raise _fail(path, f"'{_KEY_CLASSES}' must list at least 2 classes, found {len(value)}. {_METADATA_HELP}")
    duplicates = sorted({name for name in value if value.count(name) > 1})
    if duplicates:
        raise _fail(path, f"'{_KEY_CLASSES}' must contain unique names, found duplicates {duplicates}. {_METADATA_HELP}")
    return tuple(value)


def _parse_float(path: Path, key: str, raw: str) -> float:
    try:
        number = float(raw)
    except ValueError as error:
        raise _fail(path, f"'{key}' must be a float, found {raw!r}. {_METADATA_HELP}") from error
    if not math.isfinite(number):
        raise _fail(path, f"'{key}' must be a finite float, found {raw!r}. {_METADATA_HELP}")
    return number


def _parse_normalization(
    path: Path, metadata: dict[str, str]
) -> tuple[Normalization, float | None, float | None]:
    raw = metadata.get(_KEY_NORMALIZATION)
    if raw is None:
        raise _fail(path, f"missing required metadata key '{_KEY_NORMALIZATION}'. {_METADATA_HELP}")
    try:
        normalization = Normalization(raw)
    except ValueError as error:
        raise _fail(path, f"'{_KEY_NORMALIZATION}' has invalid value {raw!r}. {_METADATA_HELP}") from error

    if normalization is not Normalization.FIXED:
        present = [key for key in (_KEY_NORM_MEAN, _KEY_NORM_STD) if key in metadata]
        if present:
            raise _fail(path, f"{present} are only allowed with '{_KEY_NORMALIZATION}' = 'fixed', but normalization is '{normalization}'. Remove those keys or switch to 'fixed'. {_METADATA_HELP}")
        return normalization, None, None

    missing = [key for key in (_KEY_NORM_MEAN, _KEY_NORM_STD) if key not in metadata]
    if missing:
        raise _fail(path, f"normalization 'fixed' requires {missing}, which are missing. {_METADATA_HELP}")
    mean = _parse_float(path, _KEY_NORM_MEAN, metadata[_KEY_NORM_MEAN])
    std = _parse_float(path, _KEY_NORM_STD, metadata[_KEY_NORM_STD])
    if std <= 0:
        raise _fail(path, f"'{_KEY_NORM_STD}' must be > 0, found {std}. {_METADATA_HELP}")
    return normalization, mean, std


def load_classifier_model(path: Path) -> ClassifierModel:
    """Load an ONNX classifier and validate it against the MDPI model contract."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: '{path}'. Check the path passed to -m/--model.")
    if not path.is_file():
        raise _fail(path, "path is not a file. Pass a .onnx file to -m/--model.")
    if path.suffix.lower() != ".onnx":
        raise _fail(path, f"expected a '.onnx' file, found suffix '{path.suffix}'. Convert the model to ONNX first.")

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:  # onnxruntime raises several unrelated exception types
        raise _fail(path, f"onnxruntime cannot load this file ({error}). Check that it is a valid ONNX model.") from error

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise _fail(path, f"model must have exactly 1 input, found {len(inputs)} ({[i.name for i in inputs]}). Re-export with a single image input.")
    model_input = inputs[0]
    if model_input.type != _FLOAT_TYPE:
        raise _fail(path, f"input '{model_input.name}' must be {_FLOAT_TYPE}, found {model_input.type}. Re-export with a float32 input.")
    input_size, layout = _infer_input(path, list(model_input.shape))

    outputs = session.get_outputs()
    if len(outputs) != 1:
        raise _fail(path, f"model must have exactly 1 output, found {len(outputs)} ({[o.name for o in outputs]}). Re-export with a single output of raw scores (logits).")
    output_shape = list(outputs[0].shape)
    if len(output_shape) != 2 or not _is_static(output_shape[1]):
        raise _fail(path, f"output must have shape [B, C] with static C, found {output_shape}. Re-export with raw scores of shape [batch, num_classes].")

    metadata = dict(session.get_modelmeta().custom_metadata_map)
    classes = _parse_classes(path, metadata.get(_KEY_CLASSES))
    if output_shape[1] != len(classes):
        raise _fail(path, f"output has {output_shape[1]} scores per sample but '{_KEY_CLASSES}' lists {len(classes)} classes. Make the class list match the output size.")
    normalization, norm_mean, norm_std = _parse_normalization(path, metadata)

    return ClassifierModel(
        path=path,
        sha256=_sha256_of(path),
        input_name=model_input.name,
        input_size=input_size,
        layout=layout,
        classes=classes,
        normalization=normalization,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
