import sys
import multiprocessing
from PySide6.QtCore import QObject, Signal, Slot
import contextlib
from io import StringIO

# Add project root to sys.path to allow for module imports from the new process
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from run_pipeline import execute_pipeline, validate_inputs_and_setup
import time
import traceback

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


class QtSignalStream(StringIO):
    """
    A simple stream object that redirects writes to a PyQt/PySide signal.
    """
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def write(self, text):
        # Emit lines as they come, preserving progress updates with carriage returns
        if text is None:
            return
        if text == "":
            return
        self.signal.emit(text)
    
    def flush(self):
        # This is needed for file-like objects
        pass


def pipeline_target_function(log_queue, input_dirs, model_dir, capture_rate, image_height_cm, img_depth, img_width):
    """
    This function is the target for the multiprocessing.Process.
    It runs the pipeline and redirects all stdout/stderr to a queue.
    """
    # Redirect stdout and stderr to the queue
    logger = QueueLogger(log_queue)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger

    try:
        total = len(input_dirs)
        for idx, input_dir in enumerate(input_dirs, start=1):
            # NOTE: We can't check a stop flag here because this function
            # is in a separate process with different memory.
            # Termination is handled by the parent process.
            log_queue.put(f"[PIPELINE]: Starting {idx}/{total} → {input_dir}")
            run_config = validate_inputs_and_setup(
                input_dir=input_dir,
                model_dir=model_dir,
                capture_rate=capture_rate,
                image_height_cm=image_height_cm,
                img_depth=img_depth,
                img_width=img_width,
            )
            execute_pipeline(run_config)
            log_queue.put(f"[PIPELINE]: Finished {idx}/{total} → {input_dir}")
        
        log_queue.put("[PIPELINE]: All steps completed successfully!")

    except Exception as e:
        log_queue.put(f"---ERROR---")
        log_queue.put(f"An error occurred: {e}")
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_queue.put("---DONE---")


class PipelineManager(QObject):
    """
    Manages the MDPI pipeline execution in a separate thread.
    """
    finished = Signal()
    log_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, *args):
        super().__init__()
        
        # Unpack arguments for clarity
        (
            self.input_dirs,
            self.model_dir,
            self.capture_rate,
            self.image_height_cm,
            self.img_depth,
            self.img_width
        ) = args
        
        self.should_stop = False

    @Slot()
    def run(self):
        """
        Executes the pipeline in a separate process and streams logs back to the GUI.
        This avoids buffering and GIL issues that can delay output in frozen apps.
        """
        try:
            self.log_message.emit("[PIPELINE]: Validating inputs and setting up...")

            total = len(self.input_dirs)
            for idx, input_dir in enumerate(self.input_dirs, start=1):
                if self.should_stop:
                    self.log_message.emit("Pipeline stop requested.")
                    break

                self.log_message.emit(f"\n[PIPELINE]: Starting {idx}/{total} → {input_dir}")

                log_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=pipeline_target_function,
                    args=(
                        log_queue,
                        [input_dir],
                        self.model_dir,
                        self.capture_rate,
                        self.image_height_cm,
                        self.img_depth,
                        self.img_width,
                    ),
                )
                process.start()

                saw_error = False
                error_lines = []

                while True:
                    if self.should_stop and process.is_alive():
                        self.log_message.emit("Pipeline stop requested by user (terminating worker).")
                        process.terminate()
                        process.join()
                        break

                    try:
                        message = log_queue.get(timeout=0.1)
                    except Exception:
                        if not process.is_alive():
                            break
                        continue

                    if message == "---DONE---":
                        break
                    if message == "---ERROR---":
                        saw_error = True
                        continue
                    if saw_error:
                        error_lines.append(message)
                        continue

                    self.log_message.emit(message)

                process.join()

                if saw_error and error_lines:
                    self.error_message.emit("\n".join(error_lines))
                    if self.should_stop:
                        break

            if not self.should_stop:
                self.log_message.emit("\n[PIPELINE]: All processing complete.")

        except Exception as e:
            tb_str = traceback.format_exc()
            detailed_error_msg = f"An error occurred: {e}\n{tb_str}"
            self.error_message.emit(detailed_error_msg)

        finally:
            self.finished.emit()
            
    @Slot()
    def stop(self):
        """Signals the pipeline to stop."""
        self.log_message.emit("Pipeline stop requested by user.")
        self.should_stop = True
