from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel, QFormLayout
from PyQt6.QtCore import pyqtSignal

from pyqt_app.validation import get_detailed_validation_results
from pyqt_app.widgets.collapsible_box import CollapsibleBox


class InputPathWidget(QWidget):
    """A widget for a single input path, with validation and metadata feedback."""
    validation_changed = pyqtSignal()

    def __init__(self, path_text="", is_first=False):
        super().__init__()
        self.is_valid = False
        self.metadata = None
        self.camera_format = None

        # --- Layouts ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 5, 0, 5)
        main_layout.setSpacing(2)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)

        # --- Widgets ---
        self.path_input = QLineEdit(path_text)
        self.path_input.setPlaceholderText("Path to Raw Images Folder")
        self.path_input.textChanged.connect(self.validate_path)
        
        self.browse_button = QPushButton("Browse")
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        
        self.validation_label = QLabel("")
        self.validation_label.setVisible(False)

        # --- Metadata Display ---
        self.metadata_box = CollapsibleBox("File Metadata")
        self.metadata_widget = QWidget() # Content widget for the collapsible box
        self.metadata_layout = QFormLayout(self.metadata_widget)
        self.metadata_layout.setContentsMargins(9, 2, 9, 2) # left, top, right, bottom
        self.metadata_layout.setVerticalSpacing(2)
        self.metadata_box.setContent(self.metadata_widget)
        self.metadata_box.setVisible(False)

        # --- Assemble Layout ---
        input_layout.addWidget(self.path_input, 1)
        input_layout.addWidget(self.browse_button)
        if is_first:
            input_layout.addWidget(self.add_button)
        else:
            input_layout.addWidget(self.remove_button)

        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.validation_label)
        main_layout.addWidget(self.metadata_box)
        self.setLayout(main_layout)

        # Trigger initial validation
        self.validate_path(self.path_input.text())

    def update_metadata_display(self):
        """Populates the metadata box with formatted file info."""
        # Clear previous metadata
        while self.metadata_layout.rowCount() > 0:
            self.metadata_layout.removeRow(0)

        if not self.metadata:
            self.metadata_box.setVisible(False)
            return

        try:
            # Format metadata as in the Streamlit app
            data = self.metadata.copy()
            start_date = data.pop("recording_start_date")
            start_time = data.pop("recording_start_time")
            
            # The time object from parser includes milliseconds, so format it.
            recording_start = f"{start_date} {start_time.strftime('%H:%M:%S.%f')[:-3]}"
            
            width = data.pop("image_width_pixels")
            height = data.pop("image_height_pixels")
            image_shape = f"{width} x {height} pixels"

            # Add rows to the form layout
            self.metadata_layout.addRow(QLabel("Recording Start:"), QLabel(recording_start))
            self.metadata_layout.addRow(QLabel("Image Shape:"), QLabel(image_shape))
            
            # Ensure camera_format is a string before capitalizing
            camera_format_display = self.camera_format.capitalize() if self.camera_format else "N/A"
            self.metadata_layout.addRow(QLabel("Camera Format:"), QLabel(camera_format_display))

            self.metadata_box.setVisible(True)

        except (KeyError, ValueError) as e:
            self.metadata_box.setVisible(False)
            print(f"Could not parse and display metadata: {e}")

    def validate_path(self, path):
        """Validates the text in the input field and updates UI accordingly."""
        path = path.strip()
        
        # Reset state
        self.metadata = None
        self.camera_format = None
        
        if not path:
            self.is_valid = False # An empty path is not a valid input for running
            self.validation_label.setText("❌ Path cannot be empty.")
            self.validation_label.setStyleSheet("color: #F44336; background: transparent; padding-left: 5px;") # Red
            self.path_input.setStyleSheet("border: 1px solid #F44336;")
            self.validation_label.setVisible(True)
            self.update_metadata_display()
            self.validation_changed.emit()
            return

        validation_results, metadata, camera_format = get_detailed_validation_results(path)
        
        self.metadata = metadata
        self.camera_format = camera_format
        
        # Overall validity is true only if all checks passed
        self.is_valid = all(res[0] for res in validation_results)
        
        # Format messages for display using HTML for rich text coloring
        formatted_messages = []
        for is_success, message in validation_results:
            icon = "✔" if is_success else "❌"
            color = "#4CAF50" if is_success else "#F44336"  # Green or Red
            # Use HTML to set the color for each line individually
            formatted_messages.append(f'<p style="color: {color}; margin: 0;">{icon} {message}</p>')

        self.validation_label.setText("".join(formatted_messages))
        
        # Style the input border based on overall validity, but not the label text
        border_color = "#4CAF50" if self.is_valid else "#F44336"
        
        self.validation_label.setStyleSheet("background: transparent; padding-left: 5px;") # Ensure no background is drawn by the label itself
        self.path_input.setStyleSheet(f"border: 1px solid {border_color};")
        self.validation_label.setVisible(True)

        self.update_metadata_display()
        self.validation_changed.emit()
