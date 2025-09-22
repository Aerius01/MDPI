import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFormLayout, QGroupBox,
    QTextEdit, QFileDialog, QLabel, QMessageBox
)
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QTextCursor

from run_pipeline import (
    CAPTURE_RATE,
    IMAGE_HEIGHT_CM,
    DEFAULT_IMG_DEPTH,
    DEFAULT_IMG_WIDTH
)
from pyqt_app.widgets.input_path_widget import InputPathWidget
from pyqt_app.pipeline_manager import PipelineManager

# Correctly determine PROJECT_ROOT from this file's location
# This file is in pyqt-app, so we need to go up one level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MDPI Image Processing Pipeline")
        self.setGeometry(100, 100, 1000, 800)
        self.pipeline_thread = None
        self.pipeline_manager = None
        self.last_log_is_progress = False

        # --- Central Widget and Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Configuration Section ---
        self.config_group = QGroupBox("Configuration")
        self.config_group.setObjectName("ConfigurationGroup") # Add object name for styling
        config_layout = QFormLayout(self.config_group)
        config_layout.setLabelAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        self.capture_rate_input = QLineEdit(str(CAPTURE_RATE))
        self.image_depth_input = QLineEdit(str(DEFAULT_IMG_DEPTH * 10))
        self.image_height_input = QLineEdit(str(IMAGE_HEIGHT_CM))
        self.image_width_input = QLineEdit(str(DEFAULT_IMG_WIDTH * 10))

        # Helper function to create a row with a tooltip
        def create_row_with_tooltip(widget, title, tooltip_text):
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)

            button = QPushButton("?")
            button.setFixedSize(20, 20)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: white;
                    border: 1px solid #777;
                    border-radius: 10px; /* Make it circular */
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton:hover {
                    background-color: #666;
                }
                 QPushButton:pressed {
                    background-color: #4e4e4e;
                }
            """)
            button.clicked.connect(lambda: self.show_help_dialog(title, tooltip_text))
            layout.addWidget(button)
            return container

        config_layout.addRow("Capture Rate (Hz):", create_row_with_tooltip(
            self.capture_rate_input, 
            "Capture Rate",
            (
                "The capture rate configuration parameter represents the rate at which "
                "the camera captures images. This is a **foundational** parameter, as an "
                "inaccurately reported rate will completely skew the matching of depth "
                "values to images, leading to silent, erroneous results."
            )
        ))
        config_layout.addRow("Image Depth (cm):", create_row_with_tooltip(
            self.image_depth_input,
            "Image Depth",
            (
                "The image depth configuration parameter represents the depth of the "
                "camera's field of view into plane. That is to say, the maximum "
                "distance into the image plane at which we are confident we can still "
                "accurately identify plankton. This parameter is used to calculate "
                "the volume of water into which binned individuals are divided into "
                "when producing the plots at the end of the pipeline."
            )
        ))
        config_layout.addRow("Image Height (cm):", create_row_with_tooltip(
            self.image_height_input,
            "Image Height",
            (
                "The image height configuration parameter represents the height of the "
                "camera's field of view in metric units. This parameter is used to "
                "calculate the image overlaps, as well as the volume of water into "
                "which binned individuals are divided into when producing the plots "
                "at the end of the pipeline."
            )
        ))
        config_layout.addRow("Image Width (cm):", create_row_with_tooltip(
            self.image_width_input,
            "Image Width",
            "The image width configuration parameter represents the width of the "
            "camera's field of view in metric units. This parameter is used to "
            "calculate the volume of water into which binned individuals are divided "
            "into when producing the plots at the end of the pipeline."
        ))
        
        main_layout.addWidget(self.config_group)

        # --- Run/Stop Buttons (Define early, add to layout later) ---
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Pipeline")
        self.run_button.clicked.connect(self.start_pipeline)
        self.run_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop Pipeline")
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.stop_button.setEnabled(False)
        
        self.stop_help_button = QPushButton("?")
        self.stop_help_button.setFixedSize(20, 20)
        self.stop_help_button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid #777;
                border-radius: 10px; /* Make it circular */
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #4e4e4e;
            }
        """)
        self.stop_help_button.clicked.connect(
            lambda: self.show_help_dialog(
                "Stop the Pipeline",
                "The stop button can only queue up a 'stop' command. This command "
                "is read and executed only between successive modules, and so the pipeline "
                "will continue to run until it reaches the next module, at which time "
                "it will stop."
            )
        )

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.stop_help_button)
        button_layout.addStretch()

        # --- Input Paths Section ---
        self.paths_group = QGroupBox("Input Paths")
        self.paths_layout = QVBoxLayout(self.paths_group)
        self.input_path_widgets = []
        self.add_input_path_row(is_first=True)
        main_layout.addWidget(self.paths_group)

        # --- Add Buttons to Layout ---
        main_layout.addLayout(button_layout)

        # --- Log Display ---
        log_group = QGroupBox("Pipeline Output")
        log_layout = QVBoxLayout(log_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        main_layout.addWidget(log_group, 1) # Stretch log area

        self.apply_stylesheet()

        self.update_run_button_state()

    def show_help_dialog(self, title, message):
        QMessageBox.information(self, title, message)

    def add_input_path_row(self, is_first=False):
        """Adds a new row for an input path."""
        input_widget = InputPathWidget(is_first=is_first)
        input_widget.validation_changed.connect(self.update_run_button_state)
        
        if is_first:
            input_widget.add_button.clicked.connect(lambda: self.add_input_path_row())
        else:
            input_widget.remove_button.clicked.connect(lambda: self.remove_input_path_row(input_widget))
            
        input_widget.browse_button.clicked.connect(lambda: self.browse_for_folder(input_widget))

        self.paths_layout.addWidget(input_widget)
        self.input_path_widgets.append(input_widget)
        self.update_run_button_state()

    def remove_input_path_row(self, widget_to_remove):
        """Removes a specific input path row."""
        if widget_to_remove in self.input_path_widgets:
            widget_to_remove.deleteLater()
            self.input_path_widgets.remove(widget_to_remove)
            self.update_run_button_state()

    def browse_for_folder(self, input_widget):
        """Opens a folder dialog that shows files and sets the path."""
        dialog = QFileDialog(self, "Select Folder")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        # This option allows files to be visible, making navigation easier
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        
        if dialog.exec():
            folder_path = dialog.selectedFiles()[0]
            if folder_path:
                input_widget.path_input.setText(folder_path)

    def update_run_button_state(self):
        """
        Enables or disables the 'Run' button based on the validity of all input paths.
        """
        if not self.input_path_widgets:
            self.run_button.setEnabled(False)
            return
            
        # Check for at least one non-empty path
        has_at_least_one_path = any(
            w.path_input.text().strip() for w in self.input_path_widgets
        )
        
        # Check that all widgets report a valid state
        all_widgets_are_valid = all(w.is_valid for w in self.input_path_widgets)
        
        self.run_button.setEnabled(all_widgets_are_valid and has_at_least_one_path)

    def start_pipeline(self):
        """Collects inputs and starts the pipeline in a new thread."""
        input_dirs = [
            widget.path_input.text().strip()
            for widget in self.input_path_widgets
            if widget.path_input.text().strip()
        ]

        # This check is now redundant due to the button state, but kept as a safeguard
        if not input_dirs:
            self.log_message("Please provide at least one valid input directory.")
            return
            
        for d in input_dirs:
            if not os.path.isdir(d):
                self.log_message(f"Error: Input path is not a valid directory: {d}")
                return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.config_group.setEnabled(False)
        self.paths_group.setEnabled(False)
        self.log_display.clear()
        self.log_message("Starting pipeline...")

        model_dir = os.path.join(PROJECT_ROOT, "model")
        
        # Prepare arguments for the pipeline process
        args = (
            input_dirs,
            model_dir,
            float(self.capture_rate_input.text()),
            float(self.image_height_input.text()),
            float(self.image_depth_input.text()) / 10.0,
            float(self.image_width_input.text()) / 10.0
        )

        self.pipeline_thread = QThread()
        self.pipeline_manager = PipelineManager(*args)

        self.pipeline_manager.moveToThread(self.pipeline_thread)
        self.pipeline_thread.started.connect(self.pipeline_manager.run)
        self.pipeline_manager.finished.connect(self.pipeline_finished)
        self.pipeline_manager.log_message.connect(self.log_message)
        self.pipeline_manager.error_message.connect(self.pipeline_error)
        
        self.pipeline_thread.start()

    def stop_pipeline(self):
        """Stops the running pipeline thread."""
        if self.pipeline_manager:
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping Pipeline...")
            self.pipeline_manager.stop()
        # The thread will be cleaned up in pipeline_finished

    def pipeline_finished(self):
        """Cleans up after the pipeline finishes."""
        self.run_button.setEnabled(True)
        self.stop_button.setText("Stop Pipeline")
        self.stop_button.setEnabled(False)
        self.config_group.setEnabled(True)
        self.paths_group.setEnabled(True)
        if self.pipeline_thread:
            self.pipeline_thread.quit()
            self.pipeline_thread.wait()
        self.pipeline_thread = None
        self.pipeline_manager = None

    def pipeline_error(self, message):
        self.log_message(f"[ERROR]: {message}")
        self.pipeline_finished()

    def log_message(self, message):
        """Appends a message to the log display, handling carriage returns for progress bars."""
        # Allow empty messages to create blank lines
        if not message.strip() and message != "":
            return

        # TQDM progress bars use '|' and '%', and carriage returns for updates.
        is_progress = '\r' in message or ('|' in message and '%' in message)
        
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if is_progress:
            if self.last_log_is_progress:
                # If the last message was also progress, replace the line
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
            # Insert the new progress text
            cursor.insertText(message.strip())
        else:
            # It's a normal log line. If the last line was a progress bar,
            # QTextEdit's `append` will correctly place this on a new line.
            # We strip only carriage returns to preserve leading newlines for spacing.
            stripped_message = message.strip('\r')
            self.log_display.append(stripped_message)

            # If this is the stop message, add a blank line for tqdm to overwrite
            if "Pipeline stop requested" in stripped_message:
                self.log_display.append("")

        self.last_log_is_progress = is_progress
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def apply_stylesheet(self):
        style_path = os.path.join(os.path.dirname(__file__), "styles", "dark_theme.qss")
        try:
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Stylesheet not found at: {style_path}")

    def closeEvent(self, event):
        """Ensure the pipeline is stopped when the window is closed."""
        self.stop_pipeline()
        event.accept()
