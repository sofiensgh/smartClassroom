"""Dark theme for the whole application."""

ACCENT = "#4f8cff"

STYLESHEET = f"""
QWidget {{ background: #14161c; color: #e6e8ee; font-family: 'Segoe UI'; font-size: 10.5pt; }}
QMainWindow::separator {{ background: #23262f; }}
#Sidebar {{ background: #0f1115; border: none; outline: none; padding-top: 8px; }}
#Sidebar::item {{ padding: 12px 18px; border-radius: 8px; margin: 2px 8px; color: #aab0bf; }}
#Sidebar::item:selected {{ background: #1f2533; color: white; border-left: 3px solid {ACCENT}; }}
#Sidebar::item:hover:!selected {{ background: #181b22; }}
#Title {{ font-size: 20pt; font-weight: 600; }}
#Subtitle {{ color: #8a90a0; font-size: 11pt; }}
#Card {{ background: #1b1e26; border: 1px solid #262a35; border-radius: 12px; }}
#CardTitle {{ font-size: 12.5pt; font-weight: 600; }}
#Muted {{ color: #8a90a0; }}
#Badge {{ background: #24324d; color: #9cc0ff; border-radius: 9px; padding: 2px 10px; font-size: 9pt; }}
#BadgeDone {{ background: #1e3b2b; color: #7fe0a3; border-radius: 9px; padding: 2px 10px; font-size: 9pt; }}
#Video {{ background: #0b0c10; border: 1px solid #262a35; border-radius: 12px; color: #5d6372; }}
#BigValue {{ font-size: 22pt; font-weight: 600; color: {ACCENT}; }}
QPushButton {{ background: #232836; border: 1px solid #2e3445; border-radius: 8px; padding: 8px 16px; }}
QPushButton:hover {{ background: #2b3142; }}
QPushButton#Primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #6a9dff; }}
QPushButton#Danger {{ background: #c2414b; border: none; color: white; font-weight: 600; }}
QPushButton:disabled, QPushButton#Primary:disabled, QPushButton#Danger:disabled {{
    background: #1b1e26; border: 1px solid #262a35; color: #5d6372; }}
QComboBox, QSpinBox, QLineEdit {{ background: #1b1e26; border: 1px solid #2e3445; border-radius: 6px; padding: 5px 8px; }}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox:disabled, QLineEdit:disabled, QCheckBox:disabled {{ color: #5d6372; }}
QTableWidget {{ background: #14161c; border: 1px solid #262a35; border-radius: 6px; gridline-color: #262a35; }}
QPlainTextEdit {{ background: #0f1115; border: 1px solid #262a35; border-radius: 6px;
    font-family: Consolas; font-size: 8.5pt; color: #aab0bf; }}
QProgressBar {{ background: #14161c; border: 1px solid #262a35; border-radius: 5px; max-height: 10px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QHeaderView::section {{ background: #1b1e26; color: #8a90a0; border: none; padding: 4px 6px; }}
QCheckBox {{ spacing: 8px; }}
QStatusBar {{ background: #0f1115; color: #8a90a0; }}
QScrollArea {{ border: none; }}
"""
