# Roadmap OX: ONNX classification models

Load the `orchestrate` skill before any implementation work. The user names the phase to run.

- **Roadmap ID:** `OX`
- **Phases:** 1 (`ox-1-onnx-models`)
- **Base commit for phase 1:** `773be99` on `main`, plus the commit that adds this file (see "Before phase 1").

## Goal

Classification runs on ONNX models through `onnxruntime` instead of a TF1 checkpoint. `run_pipeline.py` gets an optional `-m/--model PATH.onnx` that defaults to the bundled model, which is the current checkpoint converted to ONNX. Any model that meets the contract below can be dropped in. Converting their own models to ONNX is the users' job; we only document the contract.

## Decisions (settled with the user, 2026-11-16)

| # | Decision | Reason |
|---|----------|--------|
| D1 | The pipeline does all preprocessing: grayscale load, Otsu threshold, largest contour, PCA rotation to upright, resize to N x N (`tools/preprocessing/image_preprocessing.py`, unchanged). Normalization is declared by the model. The model outputs raw scores; the pipeline applies softmax and the uncertainty metrics. | Keeps softmax and metrics in one place. Most PyTorch and Keras exports do not include normalization in the graph. |
| D2 | The model describes itself through ONNX `metadata_props` (keys below). Input size and layout come from the graph's input shape. No silent fallbacks: missing or malformed fields raise errors. | One file, nothing to get out of sync. |
| D3 | `-m/--model` takes a `.onnx` file only. It is optional and defaults to `<repo>/model/mdpi_4class_v1.onnx`, resolved from the repo root, not the working directory. | Unambiguous; the name says what the model is when a second one appears. |
| D4 | TensorFlow is removed from the code and from `environment_from_history.yml`. `architecture.py` and the `.ckpt` files are deleted (they stay in git history). The conversion script lives only in the scratchpad and is not committed. **The user must confirm the `.ckpt` deletion before wave 2 is spawned.** | Users convert their own models; only the contract is documented. |
| D5 | CLI only. The Streamlit GUI (`app/app.py`) just switches to the new default path, with no model picker. | The scientist asked about the CLI. A picker can be added later. |
| D6 | Conversion is accepted only if, on every vignette of all 4 sample runs, TF and ONNX pick the same class and the raw scores differ by less than 1e-4 (max abs). | Shows the conversion changes nothing. |
| D7 | `onnxruntime` CPU only. | The network is tiny (50x50 inputs, about 1M parameters). |
| D8 | `object_data.pkl` gets `results['model'] = {'path': str, 'sha256': str}`, and the model is printed to the console. The CSV format is unchanged. | The pkl is the record of the classification run. The LabelChecker (`DispGrid.py`) keeps unknown keys when it re-saves. |
| D9 | Fix `tools/metrics/entropy.py`: it returns entropies sorted in descending order, but they are indexed per vignette, so `HC_en` in every pkl so far is misaligned. It must return per-sample values like `least_confidence` and `margin_sampling`. | A known bug right next to the code being rewritten. New pkls will have different `HC_en` values than old ones. |
| D10 | The pipeline accepts input layouts `[B,H,W]`, `[B,H,W,1]` and `[B,1,H,W]` and works out which from the shape. | Covers TF/Keras (channels-last) and PyTorch (channels-first) exports. |
| D11 | `mdpi.normalization` is required. Allowed values are `run_mean`, `fixed` and `unit`. | No silent defaults; makes everyone think about how they trained. |
| D12 | `validate_model` is replaced: the full model contract is checked during input validation, before depth profiling starts. The result is a `ClassifierModel` dataclass stored on `run_config`. | A bad model fails in seconds, not after detection has run. |
| D13 | The contract lives in `modules/object_classification/README.md` (replacing its outdated TF1 notes), together with a short `onnx` metadata-stamping snippet. The top-level `README.md` gets the `-m` usage and a link to it. | The scientist reads the top-level README; the details sit with the module. |
| D14 | The runtime environment gets `onnxruntime` only, not `onnx`. | The pipeline reads metadata through `onnxruntime` (`get_modelmeta().custom_metadata_map`). `onnx` is only needed by people stamping their own models. |

## Model contract (normative; WP 1.2 implements it, WP 1.4 documents it)

**Input:** exactly 1 input, element type `tensor(float)` (float32). Allowed shapes:
- `[B, N, N]`: layout `BHW`
- `[B, N, N, 1]`: layout `BHWC`
- `[B, 1, N, N]`: layout `BCHW`

`B` must be dynamic (a symbolic name or unset). `N` must be a static int >= 2 and equal in both spatial positions. A rank-4 shape counts as `BHWC` only if `shape[3] == 1 and shape[1] != 1`, and as `BCHW` only if `shape[1] == 1 and shape[3] != 1`. Anything else is an error.

**Output:** exactly 1 output, element type `tensor(float)` (float32; added at the phase 1 gate review), shape `[B, C]` with `C` a static int equal to `len(mdpi.classes)`. These are raw scores (logits), not probabilities.

**Metadata (`metadata_props`):**

| Key | Required | Value |
|-----|----------|-------|
| `mdpi.classes` | always | JSON list of at least 2 unique non-empty strings, in output-index order |
| `mdpi.normalization` | always | `run_mean`, `fixed` or `unit` |
| `mdpi.norm_mean` | only when `fixed` (error if present otherwise) | float, on the 0-1 scale |
| `mdpi.norm_std` | only when `fixed` (error if present otherwise) | float > 0, on the 0-1 scale |

Unknown keys are ignored.

**Normalization** (`x` is the preprocessed uint8 N x N image):
- `run_mean`: `(x - m) / 255`, where `m = np.mean(all preprocessed images of the run)`. This is today's behaviour; compute it in float64 and cast the result to float32.
- `fixed`: `(x / 255 - norm_mean) / norm_std`
- `unit`: `x / 255`

## Shared interface (created by WP 1.2 in wave 1, used by 1.5 and 1.6 in wave 2)

File: `modules/object_classification/model_contract.py`

```python
class InputLayout(StrEnum):
    BHW = "BHW"
    BHWC = "BHWC"
    BCHW = "BCHW"

class Normalization(StrEnum):
    RUN_MEAN = "run_mean"
    FIXED = "fixed"
    UNIT = "unit"

class ModelContractError(ValueError): ...

@dataclass(frozen=True)
class ClassifierModel:
    path: Path              # resolved absolute path
    sha256: str             # hex digest of the file
    input_name: str
    input_size: int         # N
    layout: InputLayout
    classes: tuple[str, ...]
    normalization: Normalization
    norm_mean: float | None # set only for FIXED
    norm_std: float | None  # set only for FIXED

DEFAULT_MODEL_PATH: Path  # Path(__file__).resolve().parents[2] / "model" / "mdpi_4class_v1.onnx"

def load_classifier_model(path: Path) -> ClassifierModel:
    """Load an ONNX classifier and validate it against the MDPI model contract."""

def open_session(model: ClassifierModel) -> ort.InferenceSession:
    """Open a CPU inference session on the model file, verifying it is unchanged since validation."""
```

`open_session` was added at the phase 1 gate review: the loader reads the file once and hashes those bytes; `open_session` re-reads it, raises `ModelContractError` if the sha256 differs, and is the only place a session is built (CPU provider, D7).

`load_classifier_model` raises `FileNotFoundError` if the file is missing. Every other violation (wrong suffix, file onnxruntime cannot load, any contract rule) raises `ModelContractError`, with a message naming the file, the rule broken, what was found, and how to fix it. For metadata errors the message lists the valid keys and values.

`run_config` (a `SimpleNamespace`) gets `classifier_model: ClassifierModel`, which replaces `model_dir`.

## Project facts and constraints

- **Interpreter:** `~/miniconda3/envs/mdpi-env/bin/python`. Since the phase 1 close it is built from `environment_from_history.yml`: Python 3.11.18, numpy 2.4.6, opencv 4.13.0, scikit-image 0.26.0, scikit-learn 1.9.1, onnxruntime 1.32.0, streamlit 1.67.0, no TensorFlow. (Before phase 1: Python 3.11.13, TF 2.18.0, numpy 2.3.3, and no streamlit despite the yml.) Call it by path; do not use `conda run`. Run from the repo or worktree root so `modules` and `tools` import.
- **Lint:** `~/miniconda3/envs/scrap/bin/ruff check <changed .py files>` (ruff 0.6.9, default rules). Lint only the files a WP changes; untouched legacy code is out of scope.
- **Tests:** the repo has no test suite. Must-pass checks are commands and scratch scripts. Workers put their scripts in `$SCRATCH/ox-<wp>/`, where `$SCRATCH` is the orchestrator session's scratchpad. The orchestrator gives the absolute path in each brief.
- **One-off packages:** install into the `scrap` env (for example `onnx` for building synthetic test models), never into `mdpi-env`, except the `onnxruntime` install in WP 1.2.
- **Sample data:** `profiles/` is git-ignored, so **worktrees do not contain it**. Always use absolute paths into the main checkout: `/home/david-james/Desktop/04-MDPI/MDPI/profiles/...`. There are 4 runs, each with a CSV, about 700 TIFFs, and an existing `output/` (including `vignettes/`) from a TF run:
  - `Project_Example/20230425/day/E01_01` (9479 vignettes)
  - `Project_Example/20230425/day/E02_01` (8434)
  - `Project_Example/20230424/night/E07_01` (7473)
  - `test_2025/20250512/day/E01_01` (11051)
- **Never write into `profiles/`.** The pipeline writes to `<input>/output/` and would overwrite the reference outputs. For full pipeline runs, build a smoke input in the scratchpad: a folder with symlinks to one run's `*.tiff` files plus a copy of its CSV (raw data is about 2.9 GB per run; `/tmp` is a 16 GB tmpfs, so do not copy the images). If validation rejects symlinks, stop and report.
- **Interactive prompt:** `run_pipeline.py` asks `Accept these values? (Y/N)`. Feed it with `printf 'Y\n' | ...`.
- **TF checkpoint quirks:** `architecture.py` names both the fc weight and the fc bias `fc6` (and likewise `fc7`, `fc8`); TF makes them unique by creation order, so a rebuild must create variables in exactly the same order. `keep_prob` is a placeholder fed with 1 at inference, which makes dropout the identity. Conv1 uses stride 2 with `SAME` padding (asymmetric). LRN uses `depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0`. Note that TF and ONNX scale LRN alpha differently, so let `tf2onnx` do the mapping and do not hand-build the graph.
- **Current preprocessing quirk:** `image_preprocessing` returns `images` as a Python **list** of uint8 arrays and `images_mean` as `np.float64`. `batch_generator` computes `(images[i:j] - images_mean) / 255.`, which works only through numpy-scalar broadcasting and yields float64. The reference behaviour is therefore "float64 math, fed as float32".
- **Worktrees:** `.claude/settings.local.json` has `"worktree": {"baseRef": "head"}` (added in phase 1).
- **Untracked files:** `CLAUDE.md`, `.claudeignore` and `electron/` are untracked. Leave `electron/` alone.
- **Out of scope:** `class_viewer.py` and `DispGrid.py` (they read the pkl and need no change), `tools/nn/soft_max.py`, the Electron AppImage (it needs a rebuild later to pick this up; not part of this roadmap), and uninstalling TF from the live `mdpi-env` (see "Open for the grill").

## Before phase 1 (orchestrator, with the user's go-ahead)

1. Commit `ROADMAP.md` on `main`, so worktrees branching from the phase branch contain it.
2. Add `"worktree": {"baseRef": "head"}` to `.claude/settings.local.json`.
3. Get the user's confirmation for the `.ckpt` deletion (D4). It can be collected during the phase grill and must be on record before wave 2.

## Phase 1: ONNX models (`ox-1-onnx-models`)

### Grill outcomes (2026-11-16)
- G1: The synthetic contract checks from WP 1.2 stay in the scratchpad; no committed pytest suite (YAGNI).
- G2: At phase end, with the user's go-ahead at that point, `mdpi-env` is recreated from the updated yml as proof that it is complete.
- G3: At phase end, with the user's go-ahead, the `mdpi-onnx-convert` env is deleted. The reference `.npz` files keep the TF outputs.
- G4: The user confirmed the `.ckpt` deletion (D4) on 2026-11-16.
- Baseline correction: `validation.py` also defines `MODEL_CHECKPOINT_EXTENSIONS`; WP 1.6 deletes it together with `MODEL_CHECKPOINT_FILENAME`.

### Wave 1

#### WP 1.1: Convert the checkpoint to ONNX and prove it is equivalent
- **Model:** Opus. Conversion pitfalls (LRN scaling, SAME padding, variable order) are subtle.
- **Wave:** 1
- **Owns:** `model/mdpi_4class_v1.onnx` (new). Scratch: `$SCRATCH/ox-1.1/`.
- **Uses:** `model/model.ckpt.*`, a copy of `modules/object_classification/architecture.py` (read only), and `tools/preprocessing/image_preprocessing.py` (imported read only from the main checkout).
- **Scope:**
  1. Create a separate conda env `mdpi-onnx-convert` (Python 3.11) with pip packages: `tensorflow==2.15.*`, `tf2onnx==1.16.1`, `numpy<2`, `onnx`, `onnxruntime`, `opencv-python-headless`, `scikit-learn`, `imutils`, `tqdm`. Pick `onnx` and `onnxruntime` versions compatible with `tf2onnx` 1.16.1. If no working combination exists, stop and report; do not hand-build the graph.
  2. In `$SCRATCH/ox-1.1/convert.py`, rebuild the graph from a copy of `architecture.py` in TF1 compat mode. Variables must be created in the same order. Replace the `keep_prob` placeholder with the constant `1.0`. Restore `model/model.ckpt`, freeze the graph, and convert with `tf2onnx` (opset 17). The input is `[None, 50, 50]` float32 (layout `BHW`); the output is the fc8 logits `[None, 4]`. The batch dim must stay dynamic.
  3. Stamp the metadata: `mdpi.classes = '["cladocera", "copepod", "junk", "rotifer"]'`, `mdpi.normalization = 'run_mean'`. Save to `model/mdpi_4class_v1.onnx`.
  4. In `$SCRATCH/ox-1.1/equivalence.py`, for each of the 4 runs: preprocess `output/vignettes/` exactly as the pipeline does (vignette order from `_get_vignette_sort_key`, `image_preprocessing`, then `(images - mean)/255` in float64 cast to float32), run TF (restored checkpoint, `keep_prob=1`) and ONNX (onnxruntime CPU) on the same batches of 32, and compare.
  5. Save the TF reference per run to `$SCRATCH/ox-1.1/reference/<run_label>.npz` with arrays `filenames` (vignette basenames), `logits`, `probs` (`tools/nn/soft_max` applied to a copy). Use run labels `PE_0425_E01_01`, `PE_0425_E02_01`, `PE_0424_E07_01`, `T25_0512_E01_01`. Later WPs and the gate compare against these files.
  6. Write `$SCRATCH/ox-1.1/report.md` with versions used, per-run vignette counts, max abs logit difference and argmax mismatches.
- **Must pass:**
  - `~/miniconda3/envs/mdpi-onnx-convert/bin/python $SCRATCH/ox-1.1/equivalence.py` exits 0 and prints, for all 4 runs, `argmax mismatches: 0` and `max |dlogit| < 1e-4`.
  - Loading `model/mdpi_4class_v1.onnx` in onnxruntime shows input `[<symbolic>, 50, 50]` float, output `[<symbolic>, 4]`, and the two metadata keys above.
  - The 4 reference `.npz` files exist and the length of `filenames` matches the vignette count of each run.

#### WP 1.2: Model contract module and onnxruntime dependency
- **Model:** Sonnet. Fully pinned down by the contract.
- **Wave:** 1
- **Owns:** `modules/object_classification/model_contract.py` (new) and `environment_from_history.yml` (add `conda-forge::onnxruntime`). It also installs `onnxruntime` into `mdpi-env`.
- **Uses:** nothing.
- **Scope:**
  1. Implement exactly the "Shared interface" and "Model contract" sections. Use `onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])`; read inputs and outputs from `get_inputs()` / `get_outputs()` and metadata from `get_modelmeta().custom_metadata_map`. Compute SHA-256 by streaming the file in chunks. Use strict typing and short docstrings, and keep helpers private.
  2. Install with `conda install -n mdpi-env -c conda-forge onnxruntime`. Run `--dry-run` first; if the solve would remove or downgrade existing packages (beyond adding onnxruntime and its deps), stop and report the plan instead of installing.
  3. Do not touch `modules/common/validation.py`, `run_pipeline.py` or `__init__.py`. Those change in wave 2.
- **Must pass:**
  - `~/miniconda3/envs/mdpi-env/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"` succeeds.
  - `$SCRATCH/ox-1.2/build_models.py` (run with the `scrap` env, `onnx` installed there) builds synthetic models, and `$SCRATCH/ox-1.2/check_contract.py` (run with `mdpi-env`) verifies:
    1. Valid `BHW` + `run_mean` loads with `input_size=50`, `layout=BHW`, and the right classes.
    2. Valid `BCHW` + `fixed` (mean 0.5, std 0.25) loads with those values.
    3. Valid `BHWC` + `unit` loads.
    4. Missing `mdpi.classes` gives `ModelContractError` naming the key.
    5. Missing `mdpi.normalization` gives an error listing `run_mean, fixed, unit`.
    6. `fixed` without `mdpi.norm_std` gives an error.
    7. `unit` with `mdpi.norm_mean` present gives an error.
    8. Output size not equal to the class count gives an error.
    9. Non-square input gives an error.
    10. Static batch dim 1 gives an error.
    11. 3-channel input gives an error.
    12. Duplicate class names give an error.
    13. A nonexistent path gives `FileNotFoundError`.
    14. A `.txt` file gives `ModelContractError`.

    It exits 0 with every case printed as PASS.
  - `ruff check modules/object_classification/model_contract.py` is clean.

#### WP 1.3: Fix the entropy metric
- **Model:** Sonnet
- **Wave:** 1
- **Owns:** `tools/metrics/entropy.py`
- **Uses:** nothing.
- **Scope:** Return `(np.argsort(-ent), ent)` with `ent` in sample order (currently `ent[np.argsort(-ent)]`). Keep the signature and the `-inf` guard. Do not change the input array.
- **Must pass:**
  - A `python -c` check: for `p = np.array([[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]])`, `entropy(p)[1][0] < entropy(p)[1][1]` and `entropy(p)[0].tolist() == [1, 0]`.
  - ruff on the file is clean.

#### WP 1.4: Document the model contract
- **Model:** Sonnet
- **Wave:** 1
- **Owns:** `modules/object_classification/README.md`, `README.md`
- **Uses:** nothing (the contract text is fixed by this roadmap).
- **Scope:**
  1. Module README: replace the TF1 references (checkpoint loading, the "TensorFlow 1.x" limitation, hardcoded categories and input size) with ONNX equivalents. Add a "Using your own model" section that states the full contract from this roadmap: input layouts, dynamic batch, raw-score output, metadata table, normalization formulas, and the exact preprocessing the pipeline applies before the model (grayscale, Otsu plus largest contour, PCA rotation to upright, `cv2.resize` to N x N with default interpolation). Note that models should be trained on vignettes preprocessed the same way, and that the pipeline applies softmax.
  2. Include this snippet:
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
  3. Document the pkl's new `model` key and the `HC_en` fix (D8, D9) in the outputs section.
  4. Top-level README: CLI usage showing `-m` as optional (default `model/mdpi_4class_v1.onnx`) with one example of a custom model, plus a link to the module README section. Keep the rest unchanged.
  5. No em-dashes anywhere.
- **Must pass:**
  - `grep -n -i "tensorflow\|ckpt\|checkpoint" modules/object_classification/README.md README.md` returns nothing.
  - `grep -c $'\xe2\x80\x94' modules/object_classification/README.md README.md` (em-dash count) gives 0 for both.
  - The orchestrator checks the contract text against this roadmap line by line.

### Wave 2 (needs: 1.1's ONNX model and reference npz, 1.2's `model_contract.py` and installed onnxruntime, 1.3's fixed entropy; the user's `.ckpt` deletion confirmation on record)

#### WP 1.5: ONNX inference engine and TensorFlow removal
- **Model:** Opus. Normalization semantics and the equivalence check.
- **Wave:** 2
- **Owns:**
  - `modules/object_classification/inference_engine.py`, `run.py`, `classification_data.py` and `__init__.py`
  - `modules/object_classification/architecture.py` (delete)
  - `tools/io/` (delete the package; `batch_generator` is used only by the inference engine)
  - `environment_from_history.yml` (remove `conda-forge::tensorflow`)
  - `model/checkpoint` and `model/model.ckpt.*` (`git rm`)
- **Uses:** `ClassifierModel`, `Normalization`, `InputLayout` from 1.2; `run_config.classifier_model` (set by 1.6 in the same wave; agreed interface); fixed `entropy` from 1.3; reference npz from 1.1.
- **Scope:**
  1. `classification_data.py`: remove `CLASSIFICATION_INPUT_SIZE`, `CLASSIFICATION_INPUT_DEPTH`, `CLASSIFICATION_CATEGORIES` and `model_path`. `ClassificationData` gets `classifier_model: ClassifierModel`, and categories and input size come from it. Keep `CLASSIFICATION_BATCH_SIZE = 32`. Type the dataclass fields properly (`list[str]` and so on).
  2. `inference_engine.py`: create one onnxruntime CPU session from `classifier_model.path`. Normalize per `classifier_model.normalization` as in the contract (for `run_mean`, use the `images_mean` from `image_preprocessing`, float64 math then float32). Reshape each batch to the model's layout, run batches of 32, concatenate the logits, then apply the existing softmax, metrics and output dict. Add `'model': {'path': str(path), 'sha256': sha256}` to the results. Print `[CLASSIFICATION]: Using model <filename> (sha256 <first 12 hex>)`. No TF imports or log suppression remain.
  3. `run.py`: adapt to the above. Keep the output files and columns unchanged.
  4. `__init__.py`: make it empty (nothing imports from the package root).
  5. Delete `architecture.py`, `tools/io/`, `model/checkpoint` and `model/model.ckpt.*`, and remove TF from the yml.
- **Must pass:**
  - `grep -rn -i "tensorflow" --include=*.py --include=*.yml . | grep -v "^./electron"` returns nothing.
  - `$SCRATCH/ox-1.5/classify_existing.py` (mdpi-env) builds a `run_config` for `PE_0425_E01_01` with `output_root` = a scratch folder holding a symlink `vignettes -> <profile>/output/vignettes`, `metadata` from `validate_input_directory(<profile>)`, `classifier_model = load_classifier_model(DEFAULT_MODEL_PATH)`, and `object_data_df` = the profile's existing `output/object_data.csv` (sep `;`). It calls `run_classification` and checks that:
    - the scratch `object_data.pkl` has a `model` key with the right sha256;
    - per vignette filename, `argmax(full_y_predicted)` equals the reference npz argmax, and `max |probs - reference probs| < 1e-5`;
    - `HC_en[i]` equals the entropy of `full_y_predicted[i]` for every sample.

    It exits 0.
  - ruff on the changed `.py` files is clean.

#### WP 1.6: CLI, GUI and validation wiring
- **Model:** Sonnet
- **Wave:** 2
- **Owns:** `run_pipeline.py`, `app/app.py`, `modules/common/validation.py`
- **Uses:** `load_classifier_model`, `DEFAULT_MODEL_PATH`, `ClassifierModel` from 1.2; the ONNX model from 1.1.
- **Scope:**
  1. `run_pipeline.py`: `-m/--model` becomes optional, `type=Path`, `default=DEFAULT_MODEL_PATH`. The help text reads: `Path to an ONNX classification model (default: model/mdpi_4class_v1.onnx). See modules/object_classification/README.md for the model contract.`
  2. `validate_inputs_and_setup` takes `model_path: Path`, calls `load_classifier_model`, prints `[PIPELINE]: Using classification model <path>`, and sets `run_config.classifier_model` (removing `model_dir`). Update the docstring and epilog examples (default, and `-m path/to/custom.onnx`). Errors keep flowing to the existing `except` that prints and exits 1.
  3. `app/app.py`: replace `os.path.join(PROJECT_ROOT, "model")` with `DEFAULT_MODEL_PATH` and rename `model_dir` to `model_path` through `pipeline_worker` and its call into `validate_inputs_and_setup`.
  4. `validation.py`: delete `validate_model`, `MODEL_CHECKPOINT_FILENAME` and `MODEL_CHECKPOINT_EXTENSIONS`.
- **Must pass:**
  - `python run_pipeline.py --help` shows `-m` as optional with the help text above.
  - With a scratch smoke input (see Project facts; symlinked TIFFs from `PE_0425_E01_01` plus a copied CSV), `printf 'Y\n' | timeout 120 python run_pipeline.py -i <smoke> -m /nonexistent.onnx` exits 1, prints an error naming the path, and does **not** print `Running depth profiling`.
  - The same run with a 1.2 synthetic model that lacks `mdpi.normalization` exits 1 with the contract message and no depth profiling.
  - A `python -c` call of `validate_inputs_and_setup(<smoke>, DEFAULT_MODEL_PATH, ...)` returns a run_config whose `classifier_model.classes == ("cladocera", "copepod", "junk", "rotifer")`.
  - `python -m py_compile app/app.py` succeeds.
  - ruff on the 3 files is clean.

### Phase gate (on the merged phase head, after review fixes)
1. All Must-pass checks of 1.1 through 1.6 re-run green. 1.1's equivalence script uses its own env, and the `.ckpt` files are restored from git into the scratchpad for this re-run only, if needed.
2. Full smoke run: `printf 'Y\n' | timeout 3600 ~/miniconda3/envs/mdpi-env/bin/python run_pipeline.py -i <smoke PE_0425_E01_01>` with **no** `-m` exits 0. The smoke `output/` contains `object_data.csv`, `object_data.pkl` (with a `model` key) and the plots. Per vignette filename, predictions match the `PE_0425_E01_01` reference npz argmax. Any vignette-set difference must be explained (for example, detection nondeterminism) before the gate passes.
3. `grep -rn -i tensorflow` over tracked `.py` and `.yml` files is empty.
4. Human handover: `./start_streamlit.sh`, run one sample profile through the GUI (copy or symlink it into a smoke folder first), and check the log shows the ONNX model line and the run completes.

### Carry-over to after phase 1
- Update the project `CLAUDE.md`: the "Model files" line (now `model/mdpi_4class_v1.onnx`), the Development Commands example (`-m` optional), and the "Model validation checks for required checkpoint files" line (now the ONNX contract via `load_classifier_model`).
- The Electron AppImage must be rebuilt to ship the change (outside this roadmap).

## Outcomes

### Phase 1 (done 2026-11-16, merged to `main`)

**Commits** (`797ea8a..` on `ox-1-onnx-models`, fast-forwarded to `main`):
- Wave 1: `76074bc` 1.1 ONNX model, `731993c` 1.2 contract loader + onnxruntime, `78bbba1` 1.3 entropy fix, `db0cf29` 1.4 docs.
- Wave 2: `3d46fb8` 1.5 onnxruntime inference + TF removal + ckpt deletion, `47c78ce` 1.6 CLI/GUI/validation wiring.
- Gate review fixes: `7e999d4` 1.2 (read and hash model bytes once, `open_session` with sha re-check, float output rule, neutral error wording), `3e9f776` 1.4 (docs aligned), `a0d643a` 1.6 (model validated first, loader normalizes the path), `9b7b25d` 1.5 (sessions via `open_session`, `SampleResult` TypedDict, `assert_never`, empty-vignettes error).
- Orchestrator: `18bec9d` pin `scikit-image>=0.26`, `f00a3a8` pin `opencv<5`, `648d69b` GUI log via `st.iframe`.

**Test results (gate on `9b7b25d`, re-checked after the env rebuild):**
- D6 equivalence, TF checkpoint vs ONNX on all 4 runs (36437 vignettes): 0 argmax mismatches, max |dlogit| 2.1e-5, max |dprob| 3.2e-6.
- Classification of `PE_0425_E01_01` through `run_classification`: 0 mismatches vs the TF reference, max |dprob| 2.6e-6, `HC_en` per sample, CSV predictions identical to the old TF output.
- Full CLI smoke run with no `-m`: exit 0, all outputs and plots, pkl `model` key with the right sha256, 9479/9479 vignettes identical to the reference with 0 mismatches.
- Contract: 17 synthetic cases plus 5 review-fix cases pass. Bad model (missing file or missing metadata) exits 1 before depth profiling and creates no `output/`.
- GUI (human check, `E02_01` via symlinked smoke folder): run completes, logs show the ONNX model; 8434/8434 predictions match the reference.
- ruff clean on all changed files; no `tensorflow` in tracked `.py`/`.yml`.

**Decisions taken during the phase:** G1-G4 above. Gate review added two contract rules recorded in the contract and interface sections: output must be `tensor(float)`, and `open_session` re-verifies the sha256 before inference. Out-of-WP fixes approved by the user: pin `scikit-image>=0.26` and `opencv<5`; `mdpi-env` recreated from the yml (G2 done). G3 done after the merge: `mdpi-onnx-convert` deleted.

**Lessons:**
- The pre-phase base was already broken: `773be99` uses the scikit-image 0.26 `max_size` API, but the env had 0.25.2. Run a full smoke run during baseline verification, not only at the gate.
- The live env had drifted from the yml (no streamlit). Rebuilding the env from the yml is the only real proof of it.
- Unpinned opencv resolved to 5.1, whose rotate/resize changed preprocessed pixels by up to 5 levels and flipped 28/9479 predictions. Preprocessing is numerically sensitive: pin libraries in the preprocessing path and compare against the reference `.npz` after any env change.
- Scratch check scripts that take paths as arguments must say so in their report; two re-runs initially failed only for that reason.

**Scratch artifacts** (session scratchpad, not durable): `ox-1.1/` (convert.py, equivalence.py, reference/*.npz with the TF outputs), `ox-1.2/`, `ox-1.2r/` (contract checks and synthetic models), `ox-1.5/`, `ox-1.5r/`, `ox-1.6/`, `ox-1.6r/`, `orch/` (gate.sh and smoke folders).

**Side effects outside the repo:** installing `onnx` into `scrap` upgraded protobuf and broke grpcio-tools there; at the user's request `scrap` was then recreated fresh (python 3.11 plus `ruff==0.6.9` for the lint command).

### Carry-over (no further phase in this roadmap)
- Update the project `CLAUDE.md` (untracked): "Model files" line, the Development Commands example (`-m` optional, takes a `.onnx` file), and the "Model validation checks for required checkpoint files" line (now `load_classifier_model`).
- Rebuild the Electron AppImage to ship the change.
- Out of scope but noted by the gate review: `tools/nn/soft_max.py` subtracts the global max instead of the per-row max, so a row far below the global max could underflow to NaN in float32 (same behaviour as the old TF path).
- `start_streamlit.sh` defaults `PROJECT_ROOT` to `~/MDPI`; this checkout needs `PROJECT_ROOT=$PWD`.
