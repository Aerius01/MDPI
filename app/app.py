import streamlit as st
import os
import sys
from pathlib import Path
import multiprocessing
import queue
import time
import glob
import html
import streamlit.components.v1 as components

# Tkinter is only available when a display server is present.
# In headless environments (e.g., WSL without GUI), importing or using it will fail.
try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog  # type: ignore
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

# Consider environment headless if DISPLAY is not set (common on WSL servers)
_HEADLESS_ENV = not bool(os.environ.get("DISPLAY"))
_GUI_BROWSE_AVAILABLE = _TK_AVAILABLE and not _HEADLESS_ENV


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


    def pipeline_worker(log_queue: multiprocessing.Queue, input_dirs, output_root, model_dir, capture_rate, image_height_cm, img_depth, img_width):
        """The target function for the pipeline process. Processes one or more input directories sequentially. No st calls here."""
        # Redirect stdout and stderr to the queue
        logger = QueueLogger(log_queue)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger

        try:
            # Normalize to list
            if isinstance(input_dirs, str):
                input_dirs = [input_dirs]

            total = len(input_dirs)
            for idx, input_dir in enumerate(input_dirs, start=1):
                print(f"[PIPELINE]: Starting {idx}/{total} → {input_dir}")
                execute_pipeline(
                    input_dir,
                    output_root,
                    model_dir,
                    capture_rate,
                    image_height_cm,
                    img_depth,
                    img_width
                )
                print(f"[PIPELINE]: Finished {idx}/{total} → {input_dir}")
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
        """Open folder dialog and update session_state. Disabled on headless."""
        if not _GUI_BROWSE_AVAILABLE:
            st.session_state["browse_error"] = (
                "GUI folder picker is unavailable in headless environments. "
                "Please paste the folder path manually."
            )
            return
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
    if 'raw_input_keys' not in st.session_state:
        st.session_state.raw_input_keys = ['raw_images_0']
    if 'raw_input_counter' not in st.session_state:
        st.session_state.raw_input_counter = 1
    if 'raw_images_0' not in st.session_state:
        st.session_state.raw_images_0 = ""
    if 'browse_error' not in st.session_state:
        st.session_state.browse_error = None


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
    # Dynamic list of input rows: each row has its own state key (e.g., 'raw_images_0')

    if 'output_dir' not in st.session_state:
        # Default to the project's output folder
        st.session_state.output_dir = os.path.join(PROJECT_ROOT, "output")

    st.write("Raw Images Folders")
    if not _GUI_BROWSE_AVAILABLE:
        st.info("GUI folder picker disabled (headless). Paste folder paths manually.")

    def add_input_row():
        key = f"raw_images_{st.session_state.raw_input_counter}"
        st.session_state.raw_input_counter += 1
        st.session_state.raw_input_keys.append(key)
        st.session_state[key] = ""

    def remove_input_row(row_key: str):
        if len(st.session_state.raw_input_keys) <= 1:
            return  # keep at least one row
        if row_key in st.session_state.raw_input_keys:
            st.session_state.raw_input_keys.remove(row_key)
            try:
                del st.session_state[row_key]
            except KeyError:
                pass

    # Render rows and collect validation state
    input_values = []
    input_valid_flags = []
    for idx, row_key in enumerate(list(st.session_state.raw_input_keys)):
        with st.container(border=True):
            col1, col2, col3 = st.columns([0.6, 4, 0.6])
            with col1:
                st.button(
                    "Browse",
                    on_click=select_folder,
                    args=(row_key,),
                    key=f"browse_{row_key}",
                    use_container_width=True,
                    disabled=not _GUI_BROWSE_AVAILABLE,
                )
            with col2:
                st.text_input("Path to Raw Images Folder", key=row_key, label_visibility="collapsed")
            with col3:
                if idx == 0:
                    st.button("Add", on_click=add_input_row, key="add_input_row", use_container_width=True)
                else:
                    st.button("Remove", on_click=remove_input_row, args=(row_key,), key=f"remove_{row_key}", use_container_width=True)

            current_value = st.session_state.get(row_key, "")
            v = validate_raw_images_path(current_value)
            display_validation(v)
            is_valid_path = all(_v[0] for _v in v)
            if is_valid_path:
                st.write("")
                display_path_parsing(current_value)
                st.write("")
            input_values.append(current_value)
            input_valid_flags.append(is_valid_path)

    all_inputs_valid = (len(input_values) > 0) and all(input_valid_flags)

    st.write("Path to Output Folder")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button(
            "Browse",
            on_click=select_folder,
            args=("output_dir",),
            key="browse_output",
            use_container_width=True,
            disabled=not _GUI_BROWSE_AVAILABLE,
        )
    with col2:
        st.text_input("Path to Output Folder", key="output_dir", label_visibility="collapsed")
    if st.session_state.browse_error:
        st.warning(st.session_state.browse_error)
        st.session_state.browse_error = None
    out_validation_results = validate_output_path(st.session_state.output_dir)
    display_validation(out_validation_results)
    out_valid = all(v[0] for v in out_validation_results)

    st.write("")  # Add vertical space

    st.divider()

    # --- Run/Stop Buttons ---
    def start_pipeline():
        output_dir = st.session_state.output_dir
        model_dir = os.path.join(PROJECT_ROOT, "model")

        # Build list of inputs from dynamic rows
        input_dirs = [
            st.session_state.get(k, "").strip()
            for k in st.session_state.raw_input_keys
            if st.session_state.get(k, "").strip()
        ]

        # Validate all inputs
        inputs_valid = (
            len(input_dirs) > 0 and all(all(v[0] for v in validate_raw_images_path(p)) for p in input_dirs)
        )
        out_valid = all(v[0] for v in validate_output_path(output_dir))

        if not inputs_valid:
            st.error("Please add at least one valid input folder. Fix validation errors above.")
            return
        if not out_valid:
            st.error("Please fix the output folder validation errors before running the pipeline.")
            return

        log_queue = multiprocessing.Queue()
        # Convert cm to dm before passing to worker
        image_depth_dm = image_depth_cm / 10.0
        image_width_dm = image_width_cm / 10.0
        process = multiprocessing.Process(
            target=pipeline_worker,
            args=(log_queue, input_dirs, output_dir, model_dir, capture_rate, image_height_cm, image_depth_dm, image_width_dm)
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
    validation_blocking = not (all_inputs_valid and out_valid)

    b_col1, b_col2, _ = st.columns([1, 1, 5])
    with b_col1:
        st.button("Run", on_click=start_pipeline, disabled=is_running or validation_blocking, use_container_width=True)
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
