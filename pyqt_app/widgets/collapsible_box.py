from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QScrollArea, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup

class CollapsibleBox(QWidget):
    """
    A custom collapsible widget with a button header and smooth animation.
    """
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.title = title
        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: 1px solid #777;
                border-radius: 4px;
                background-color: #3c3c3c; /* Match groupbox color */
                padding: 4px;
            }
            QPushButton:hover {
                border-color: #999;
            }
        """)

        # Use a layout on the button to hold an arrow and title label
        button_layout = QHBoxLayout(self.toggle_button)
        button_layout.setContentsMargins(5, 5, 5, 5)
        button_layout.setSpacing(5)

        self.arrow_label = QLabel()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; background: transparent;")
        
        button_layout.addWidget(self.arrow_label)
        button_layout.addWidget(self.title_label, 1) # Stretch to fill space
        
        self.content_area = QScrollArea()
        self.content_area.setWidgetResizable(True) # Crucial for content to show
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        
        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setDuration(200)
        self.toggle_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self.toggle)
        self.update_arrow(False)

    def setContent(self, content_widget):
        """Sets the widget to be displayed in the content area."""
        self.content_area.setWidget(content_widget)
        # When content is set, collapse the box to calculate the initial collapsed size
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)

    def toggle(self, checked):
        """Handles the expand/collapse animation."""
        self.update_arrow(checked)
        
        content_height = self.content_area.widget().sizeHint().height()
        
        self.toggle_animation.setStartValue(self.content_area.maximumHeight())
        self.toggle_animation.setEndValue(content_height if checked else 0)
        self.toggle_animation.start()

    def update_arrow(self, checked):
        """Updates the arrow icon on the label."""
        arrow_char = "▼" if checked else "▶"
        self.arrow_label.setText(arrow_char)
        self.arrow_label.setStyleSheet("font-size: 9pt; background: transparent;")
