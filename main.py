"""Smart Classroom - desktop application.

    python main.py
"""
import sys

from PySide6.QtWidgets import QApplication

from smartclass.config import APP_NAME
from smartclass.ui.main_window import MainWindow
from smartclass.ui.theme import STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
