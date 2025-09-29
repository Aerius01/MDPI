import os
import sys
from flask import Flask, request, jsonify
import subprocess
import threading
import multiprocessing

# Add project root to sys.path to allow module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.common.validation import validate_input_directory
from run_pipeline import validate_inputs_and_setup, execute_pipeline

app = Flask(__name__)

HOST_HOME_DIR = os.environ.get('HOST_HOME_DIR')
CONTAINER_HOME_DIR = '/host_home'
CONTAINER_REPO_ROOT = '/projects/MDPI'


# --- State ---
pipeline_process = None


def sequential_pipeline_worker(input_dirs, config, model_dir, host_home_dir):
    """
    The target function for the pipeline thread.
    It validates, sets up, and executes the pipeline for each input directory sequentially.
    """
    total_runs = len(input_dirs)
    print("[SEPARATOR]")

    for i, host_path in enumerate(input_dirs):
        print(f"[PIPELINE]: Starting run {i + 1}/{total_runs} → '{host_path}'")
        try:
            container_path = host_path
            if host_home_dir and host_path.startswith(host_home_dir):
                relative_path = os.path.relpath(host_path, host_home_dir)
                container_path = os.path.join(CONTAINER_HOME_DIR, relative_path)

            run_config = validate_inputs_and_setup(
                input_dir=container_path,
                model_dir=model_dir,
                capture_rate=config.get('capture_rate'),
                image_height_cm=config.get('image_height_cm'),
                img_depth=config.get('image_depth_cm') / 10.0,
                img_width=config.get('image_width_cm') / 10.0,
            )

            execute_pipeline(run_config)
            print(f"[PIPELINE]: Run {i + 1}/{total_runs} completed successfully")
        except Exception as e:
            print(f"[PIPELINE]: Error during run {i + 1}/{total_runs} for '{host_path}': {e}")
            # Continue to the next run even if one fails
        print("[SEPARATOR]")
    print("[PIPELINE]: All steps completed successfully!")


@app.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    path = data.get('path')

    if not path:
        return jsonify({"error": "Path is missing."}), 400

    try:
        container_path = path
        if HOST_HOME_DIR and path.startswith(HOST_HOME_DIR):
            relative_path = os.path.relpath(path, HOST_HOME_DIR)
            container_path = os.path.join(CONTAINER_HOME_DIR, relative_path)
        
        results, metadata, _, camera_format = validate_input_directory(container_path)

        # Sanitize paths in results for display on the host
        sanitized_results = []
        for success, message in results:
            if isinstance(message, str):
                # Remove single quotes for replacement to catch paths in error messages
                message = message.replace(CONTAINER_HOME_DIR, HOST_HOME_DIR)
            sanitized_results.append((success, message))
        
        all_passed = all(s for s, _ in sanitized_results)

        # If any check failed, return a 400 with partial results but no sensitive metadata
        if not all_passed:
            first_error = next((m for s, m in sanitized_results if not s), "Validation failed")
            return jsonify({
                "error": first_error,
                "results": sanitized_results,
                "metadata": {}  # Return empty metadata on failure
            }), 400
        
        # On full success, add camera format to metadata
        if camera_format:
            metadata['camera_format'] = camera_format

        # Convert non-serializable types to strings before sending
        if 'recording_start_time' in metadata and hasattr(metadata['recording_start_time'], 'strftime'):
            metadata['recording_start_time'] = metadata['recording_start_time'].strftime('%H:%M:%S.%f')[:-3]

        return jsonify({
            "results": sanitized_results,
            "metadata": metadata
        })
    except Exception as e:
        # Catch any other unexpected errors during validation
        error_message = str(e).replace(CONTAINER_HOME_DIR, HOST_HOME_DIR)
        return jsonify({"error": f"An unexpected error occurred: {error_message}"}), 500


@app.route('/run', methods=['POST'])
def run():
    global pipeline_process
    if pipeline_process and pipeline_process.is_alive():
        return jsonify({"ok": False, "error": "Pipeline is already running."}), 409

    data = request.get_json()
    input_dirs = data.get('input_paths')
    config = data.get('config')
    model_dir = os.path.join(PROJECT_ROOT, 'model')

    if not input_dirs:
        return jsonify({"ok": False, "error": "No input directories provided."}), 400

    try:
        # Launch the pipeline in a separate process so it can be terminated
        pipeline_process = multiprocessing.Process(
            target=sequential_pipeline_worker,
            args=(input_dirs, config, model_dir, HOST_HOME_DIR)
        )
        pipeline_process.daemon = True
        pipeline_process.start()

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop():
    global pipeline_process
    if pipeline_process and pipeline_process.is_alive():
        print('[PIPELINE]: Terminating pipeline process...')
        pipeline_process.terminate()  # Send SIGTERM
        pipeline_process.join(timeout=10) # Wait for process to exit
        if pipeline_process.is_alive():
            print('[PIPELINE]: Process did not terminate gracefully, killing.')
            pipeline_process.kill() # Force kill if it doesn't respond
        pipeline_process = None
        return jsonify({"ok": True, "message": "Pipeline stopped."})
    return jsonify({"ok": False, "error": "Pipeline not running."})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

