import sys
import logging
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from pathlib import Path

def setup_logging():
    """设置日志系统"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """应用程序入口点"""
    setup_logging()
    logging.info("Starting Emotion Recognition System")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()