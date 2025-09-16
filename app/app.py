import streamlit as st
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import multiprocessing
import queue
import time
import glob
import html
import streamlit.components.v1 as components


def main():
    st.set_page_config(layout="wide")


    # Project root is the directory above the 'app' directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Ensure project root is on sys.path for local execution
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Common
    from modules.common.parser import _parse_path_metadata
    from run_pipeline import (
        execute_pipeline,
        CAPTURE_RATE,
        IMAGE_HEIGHT_CM,
        DEFAULT_IMG_DEPTH,
        DEFAULT_IMG_WIDTH
    )


    class QueueLogger:
        """A file-like object that writes to a multiprocessing queue."""
        def __init__(self, queue):
            self.queue = queue

        def write(self, message):
            if message.strip():
                self.queue.put(message.strip())

        def flush(self):
            """This is needed for file-like objects."""
            pass


    def validate_raw_images_path(path):
        """Check if the raw images path is valid."""
        results = []
        if not path or not os.path.isdir(path):
            results.append((False, "Path must be a valid directory."))
            return results

        results.append((True, "Path is a valid directory."))

        try:
            _parse_path_metadata(Path(path))
            results.append((True, "Directory structure is valid."))
        except ValueError as e:
            results.append((False, f"Directory structure error: {e}"))

        image_extensions = ('*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png')
        image_files = [f for ext in image_extensions for f in glob.glob(os.path.join(path, ext))]
        if image_files:
            results.append((True, f"Found {len(image_files)} image file(s)."))
        else:
            results.append((False, "No supported image files found (.tif, .jpg, .png)."))

        # Match CSV files case-insensitively (e.g., .csv, .CSV, .CsV)
        csv_files = glob.glob(os.path.join(path, '*.[Cc][Ss][Vv]'))
        if len(csv_files) == 1:
            results.append((True, "Found one pressure sensor CSV file."))
        else:
            results.append((False, f"Expected 1 CSV file, but found {len(csv_files)}."))
        return results

    def validate_output_path(path):
        """Check if the output path is valid."""
        results = []
        if not path or not path.strip():
            return [(False, "Output path cannot be empty.")]

        parent_dir = os.path.dirname(path) or '.'
        if not os.path.isdir(parent_dir):
            return [(False, f"Parent directory does not exist: {parent_dir}")]

        if os.access(parent_dir, os.W_OK):
            results.append((True, "Output directory is writable."))
        else:
            results.append((False, "Output directory is not writable."))

        if os.path.exists(path) and not os.path.isdir(path):
            results.append((False, "Path exists but is a file, not a directory."))
        return results

    def display_validation(validation_results):
        """Render validation results with icons."""
        for is_valid, message in validation_results:
            if is_valid:
                st.markdown(f"<p style='color:green; margin-bottom:0.1rem; font-size:0.9rem;'>✔ {message}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:red; margin-bottom:0.1rem; font-size:0.9rem;'>❌ {message}</p>", unsafe_allow_html=True)

    def display_path_parsing(path):
        """Show how the raw image path will be parsed in a collapsible expander."""
        with st.expander("Path Parsing Details", expanded=True):
            try:
                metadata = _parse_path_metadata(Path(path))
                items = list(metadata.items())
                for i, (key, value) in enumerate(items):
                    # Add extra bottom margin to the last item for padding
                    margin_bottom = '0.8rem' if i == len(items) - 1 else '0.1rem'
                    st.markdown(f"<p style='margin-bottom:{margin_bottom}; font-size:0.9rem; margin-left: 1rem;'><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>", unsafe_allow_html=True)
            except ValueError as e:
                st.markdown(f"<p style='color:orange; margin-bottom:0.1rem; font-size:0.9rem;'>⚠️ Could not parse path: {e}</p>", unsafe_allow_html=True)


    def pipeline_worker(log_queue: multiprocessing.Queue, input_dir, output_root, model_dir, capture_rate, image_height_cm, img_depth, img_width):
        """The target function for the pipeline process. No st calls here."""
        # Redirect stdout and stderr to the queue
        logger = QueueLogger(log_queue)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger

        try:
            execute_pipeline(
                input_dir,
                output_root,
                model_dir,
                capture_rate,
                image_height_cm,
                img_depth,
                img_width
            )
            log_queue.put("---DONE---")
            # Use original stdout for the final success message to avoid race condition
            original_stdout.write("[PIPELINE]: All steps completed successfully!\n")
        except Exception as e:
            log_queue.put("---ERROR---")
            # Use original stderr for the final error message
            original_stderr.write(f"[PIPELINE]: Error: {e}\n")
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr


    def select_folder(key: str):
        """Open folder dialog and update session_state."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        if folder_path:
            st.session_state[key] = folder_path

    # --- Initialize Session State ---
    if 'pipeline_process' not in st.session_state:
        st.session_state.pipeline_process = None
    if 'log_queue' not in st.session_state:
        st.session_state.log_queue = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []


    # --- Title & Intro ---
    st.title("MDPI Image Processing Pipeline")
    st.write("Select input folders and run the pipeline.")

    # --- Configuration ---
    st.subheader("Configuration")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        capture_rate = st.number_input("Capture Rate (Hz)", value=CAPTURE_RATE)
        image_depth_cm = st.number_input("Image Depth (cm)", value=DEFAULT_IMG_DEPTH * 10)
    with c_col2:
        image_height_cm = st.number_input("Image Height (cm)", value=IMAGE_HEIGHT_CM)
        image_width_cm = st.number_input("Image Width (cm)", value=DEFAULT_IMG_WIDTH * 10)


    # --- Folder Pickers ---
    st.subheader("Input Paths")
    if 'raw_images' not in st.session_state:
        # Default to empty to prompt user selection
        st.session_state.raw_images = ""

    if 'output_dir' not in st.session_state:
        # Default to the project's output folder
        st.session_state.output_dir = os.path.join(PROJECT_ROOT, "output")

    st.write("Path to Raw Images Folder")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("Browse", on_click=select_folder, args=("raw_images",), key="browse_raw", use_container_width=True)
    with col2:
        st.text_input("Path to Raw Images Folder", key="raw_images", label_visibility="collapsed")
    raw_images_validation = validate_raw_images_path(st.session_state.raw_images)
    display_validation(raw_images_validation)
    if all(v[0] for v in raw_images_validation):
        st.write("")  # Add vertical space
        display_path_parsing(st.session_state.raw_images)
        st.write("")  # Add vertical space
    else:
        # Maintain vertical space if expander is not shown
        st.write("")

    st.write("Path to Output Folder")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("Browse", on_click=select_folder, args=("output_dir",), key="browse_output", use_container_width=True)
    with col2:
        st.text_input("Path to Output Folder", key="output_dir", label_visibility="collapsed")
    display_validation(validate_output_path(st.session_state.output_dir))

    st.write("")  # Add vertical space

    st.divider()

    # --- Run/Stop Buttons ---
    def start_pipeline():
        raw_images = st.session_state.raw_images
        output_dir = st.session_state.output_dir
        model_dir = os.path.join(PROJECT_ROOT, "model")

        raw_valid = all(v[0] for v in validate_raw_images_path(raw_images))
        out_valid = all(v[0] for v in validate_output_path(output_dir))

        if not (raw_valid and out_valid):
            st.error("Please fix the validation errors before running the pipeline.")
            return
        else:
            log_queue = multiprocessing.Queue()
            # Convert cm to dm before passing to worker
            image_depth_dm = image_depth_cm / 10.0
            image_width_dm = image_width_cm / 10.0
            process = multiprocessing.Process(
                target=pipeline_worker,
                args=(log_queue, raw_images, output_dir, model_dir, capture_rate, image_height_cm, image_depth_dm, image_width_dm)
            )
            process.daemon = True
            st.session_state.pipeline_process = process
            st.session_state.log_queue = log_queue
            st.session_state.logs = ["Starting pipeline..."]
            process.start()

    def stop_pipeline():
        process = st.session_state.pipeline_process
        if process and process.is_alive():
            process.terminate()
            # Wait for a short period to allow for graceful termination
            process.join(timeout=2)
            # If the process is still alive, force kill it
            if process.is_alive():
                process.kill()
            
            st.session_state.pipeline_process = None
            st.session_state.log_queue = None
            st.session_state.logs.append("Pipeline stopped by user.")


    is_running = st.session_state.pipeline_process is not None and st.session_state.pipeline_process.is_alive()

    b_col1, b_col2, _ = st.columns([1, 1, 5])
    with b_col1:
        st.button("Run", on_click=start_pipeline, disabled=is_running, use_container_width=True)
    with b_col2:
        st.button("Stop", on_click=stop_pipeline, disabled=not is_running, use_container_width=True)


    # --- Log Display ---
    st.subheader("Pipeline Output")
    log_placeholder = st.empty()

    if st.session_state.log_queue:
        while True:
            try:
                log_entry = st.session_state.log_queue.get_nowait()
                if log_entry == "---DONE---" or log_entry == "---ERROR---":
                    # Signal received. Get the final message that follows.
                    try:
                        final_message = st.session_state.log_queue.get_nowait()
                        st.session_state.logs.append(final_message)
                    except queue.Empty:
                        # Fallback if worker fails to send final message
                        if log_entry == "---ERROR---":
                            st.session_state.logs.append("[PIPELINE]: An unknown error occurred.")

                    # Clean up state
                    st.session_state.pipeline_process = None
                    st.session_state.log_queue = None
                    break  # Exit the log-reading loop
                else:
                    st.session_state.logs.append(log_entry)
            except queue.Empty:
                # No more messages for now
                break

    log_string = "\n".join(st.session_state.logs)
    escaped_logs = html.escape(log_string)
    log_html = f"""
    <div id="log-container" style="height: 400px; overflow-y: auto; border: 1px solid #333; border-radius: 5px; padding: 10px; background-color: #000;">
        <pre><code style="color: #fff;">{escaped_logs}</code></pre>
    </div>
    <script>
        var container = document.getElementById('log-container');
        container.scrollTop = container.scrollHeight;
    </script>
    """
    with log_placeholder.container():
        components.html(log_html, height=420)


    if is_running:
        time.sleep(0.1)
        st.rerun()

if __name__ == '__main__':
    main()
