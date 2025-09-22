
import sys
import os
from PySide6.QtWidgets import QApplication

# This ensures the script can be run from anywhere and still find the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pyqt_app.app_window import MainWindow

def main():
    """Initializes and runs the PyQt application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
