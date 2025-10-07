import os
import sys
from flask import Flask, request, jsonify
import multiprocessing
import threading

# Add project root to sys.path to allow module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.modules.common.validation import validate_input_directory
from pipeline.run_pipeline import validate_inputs_and_setup, execute_pipeline

app = Flask(__name__)

# Build path mappings from environment variables
NUM_MOUNTS = int(os.environ.get('NUM_MOUNTS', 0))
PATH_MAPPINGS = {}  # host_path -> container_path

for i in range(NUM_MOUNTS):
    host_path = os.environ.get(f'HOST_PATH_{i}')
    container_path = os.environ.get(f'CONTAINER_PATH_{i}')
    if host_path and container_path:
        PATH_MAPPINGS[host_path] = container_path

MODEL_DIR = os.environ.get('MODEL_DIR', '/model')


# --- State ---
pipeline_process = None
pipeline_lock = threading.Lock()
pipeline_stop_event = None


def host_to_container_path(host_path):
    """Convert host path to container path using known mappings."""
    for host_prefix, container_prefix in PATH_MAPPINGS.items():
        if host_path.startswith(host_prefix):
            relative = os.path.relpath(host_path, host_prefix)
            if relative == '.':
                return container_prefix
            return os.path.join(container_prefix, relative)
    # If no mapping found, path is already a container path
    return host_path


def container_to_host_path(container_path):
    """Convert container path back to host path for display."""
    for host_prefix, container_prefix in PATH_MAPPINGS.items():
        if container_path.startswith(container_prefix):
            relative = os.path.relpath(container_path, container_prefix)
            if relative == '.':
                return host_prefix
            return os.path.join(host_prefix, relative)
    # If no mapping found, return as-is
    return container_path


def sequential_pipeline_worker(input_dirs, config, model_dir, stop_event):
    """
    The target function for the pipeline thread.
    It validates, sets up, and executes the pipeline for each input directory sequentially.
    """
    total_runs = len(input_dirs)
    print("[SEPARATOR]")

    for i, container_path in enumerate(input_dirs):
        if stop_event and stop_event.is_set():
            break

        # Convert to host path for display
        display_path = container_to_host_path(container_path)
        print(f"[PIPELINE]: Starting run {i + 1}/{total_runs} → '{display_path}'")

        try:
            run_config = validate_inputs_and_setup(
                input_dir=container_path,
                model_dir=model_dir,
                capture_rate=config.get('capture_rate'),
                image_height_cm=config.get('image_height_cm'),
                img_depth=config.get('image_depth_cm') / 10.0,
                img_width=config.get('image_width_cm') / 10.0,
            )

            execute_pipeline(run_config, stop_check=(stop_event.is_set if stop_event else None))

            if not (stop_event and stop_event.is_set()):
                print(f"[PIPELINE]: Run {i + 1}/{total_runs} completed successfully")
        except Exception as e:
            print(f"[PIPELINE]: Error during run {i + 1}/{total_runs} for '{display_path}': {e}")
            # Continue to the next run even if one fails
        print("[SEPARATOR]")

    if stop_event and stop_event.is_set():
        print("[PIPELINE]: All runs aborted.")
    else:
        print("[PIPELINE]: All runs completed successfully!")


@app.route('/validate', methods=['POST'])
def validate():
    """
    Validate endpoint - performs authoritative Python validation.
    Note: This is now primarily used for full validation before running the pipeline.
    Quick validation for UI feedback happens in Electron/Node.js.
    """
    data = request.get_json()
    path = data.get('path')

    if not path:
        return jsonify({"error": "Path is missing."}), 400

    try:
        # Convert host path to container path if needed
        container_path = host_to_container_path(path)

        results, metadata, _, camera_format = validate_input_directory(container_path)

        # Sanitize paths in results for display on the host
        sanitized_results = []
        for success, message in results:
            if isinstance(message, str):
                # Convert container paths back to host paths for display
                message_display = container_to_host_path(message) if message.startswith('/') else message
                sanitized_results.append((success, message_display))
            else:
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
        # Flask may serialize a date with an RFC-style string that includes "00:00:00 GMT".
        # Format the date explicitly to avoid the extra time component in the UI.
        if 'recording_start_date' in metadata and hasattr(metadata['recording_start_date'], 'strftime'):
            metadata['recording_start_date'] = metadata['recording_start_date'].strftime('%a, %d %b %Y')

        return jsonify({
            "results": sanitized_results,
            "metadata": metadata
        })
    except Exception as e:
        # Catch any other unexpected errors during validation
        error_message = str(e)
        # Convert container paths in error messages to host paths
        for container_prefix in PATH_MAPPINGS.values():
            if container_prefix in error_message:
                error_message = error_message.replace(container_prefix, container_to_host_path(container_prefix))
        return jsonify({"error": f"An unexpected error occurred: {error_message}"}), 500


@app.route('/run', methods=['POST'])
def run():
    global pipeline_process, pipeline_stop_event
    # Ensure only one caller can check/start at a time
    with pipeline_lock:
        if pipeline_process and pipeline_process.is_alive():
            return jsonify({"ok": False, "error": "Pipeline is already running."}), 409

    data = request.get_json()
    input_dirs = data.get('input_paths')  # These are already container paths from Electron
    config = data.get('config')

    if not input_dirs:
        return jsonify({"ok": False, "error": "No input directories provided."}), 400

    try:
        # Launch the pipeline in a separate process so it can be terminated
        with pipeline_lock:
            pipeline_stop_event = multiprocessing.Event()
            pipeline_process = multiprocessing.Process(
                target=sequential_pipeline_worker,
                args=(input_dirs, config, MODEL_DIR, pipeline_stop_event)
            )
            pipeline_process.daemon = True
            pipeline_process.start()

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop():
    global pipeline_process, pipeline_stop_event
    # Make stop idempotent and thread-safe
    with pipeline_lock:
        proc = pipeline_process
        if proc and proc.is_alive():
            # Request cooperative stop first
            if pipeline_stop_event:
                print('[PIPELINE]: Stop request received.')
                pipeline_stop_event.set()
            # Wait a short while for graceful shutdown
            proc.join(timeout=5)
            if proc.is_alive():
                print('[PIPELINE]: Terminating pipeline process...')
                proc.terminate()  # Send SIGTERM
                proc.join(timeout=10)  # Wait for process to exit
            if proc.is_alive():
                print('[PIPELINE]: Process did not terminate gracefully, killing.')
                proc.kill()  # Force kill if it doesn't respond
            pipeline_process = None
            pipeline_stop_event = None
            return jsonify({"ok": True, "message": "Pipeline stopped."})

        # Already stopped (idempotent success)
        pipeline_process = None
        pipeline_stop_event = None
        return jsonify({"ok": True, "message": "Pipeline already stopped."})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

