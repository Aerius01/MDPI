#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure project root (/app) is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.depth_profiling.depth_profile_data import CAPTURE_RATE, IMAGE_HEIGHT_CM
from run_pipeline import validate_inputs_and_setup, execute_pipeline, DEFAULT_IMG_DEPTH, DEFAULT_IMG_WIDTH


def main():
    # Input directory is mounted at /app/input by Electron
    input_dir = os.environ.get('MDPI_INPUT_DIR', '/app/input')
    model_dir = os.environ.get('MDPI_MODEL_DIR', '/app/model')

    # Configuration mirrors Streamlit defaults and conversions
    capture_rate = CAPTURE_RATE
    image_height_cm = IMAGE_HEIGHT_CM
    img_depth_dm = DEFAULT_IMG_DEPTH  # already in dm
    img_width_dm = DEFAULT_IMG_WIDTH  # already in dm

    try:
        run_config = validate_inputs_and_setup(
            input_dir=input_dir,
            model_dir=model_dir,
            capture_rate=capture_rate,
            image_height_cm=image_height_cm,
            img_depth=img_depth_dm,
            img_width=img_width_dm,
        )
        execute_pipeline(run_config)
        print("[PIPELINE]: All steps completed successfully!")
    except Exception as e:
        print(f"[PIPELINE]: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


