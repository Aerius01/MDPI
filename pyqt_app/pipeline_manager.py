import sys
import multiprocessing
from PyQt6.QtCore import QObject, pyqtSignal, QThread

# Add project root to sys.path to allow for module imports from the new process
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from run_pipeline import execute_pipeline, validate_inputs_and_setup

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
    Manages the pipeline process and communicates with the GUI thread.
    This object runs within a QThread.
    """
    log_message = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline_process = None
        self.queue = multiprocessing.Queue()
        self.process_args = args
        self.process_kwargs = kwargs
        self._is_running = True

    def run(self):
        """
        Starts the pipeline process and monitors the output queue.
        """
        try:
            self.pipeline_process = multiprocessing.Process(
                target=pipeline_target_function,
                args=(self.queue, *self.process_args),
                kwargs=self.process_kwargs
            )
            self.pipeline_process.start()

            # Monitor the queue for messages
            while self._is_running and self.pipeline_process.is_alive():
                try:
                    message = self.queue.get(timeout=0.1)
                    if message == "---DONE---" or message == "---ERROR---":
                        break
                    self.log_message.emit(message)
                except Exception: # queue.Empty
                    continue
            
            # Ensure process is cleaned up if it finishes on its own
            if self.pipeline_process.is_alive():
                self.pipeline_process.join(timeout=1)

        finally:
            self._is_running = False
            self.finished.emit()

    def stop(self):
        """Terminates the pipeline process."""
        self.log_message.emit("Pipeline stop requested by user.")
        self._is_running = False
        if self.pipeline_process and self.pipeline_process.is_alive():
            try:
                self.pipeline_process.terminate()
                self.pipeline_process.join(timeout=2) # Wait for graceful termination
                if self.pipeline_process.is_alive():
                    self.pipeline_process.kill() # Force kill if necessary
            except Exception as e:
                self.log_message.emit(f"Error during pipeline stop: {e}")
