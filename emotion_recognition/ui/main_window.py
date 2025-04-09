import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QWidget, 
    QMessageBox, QTextEdit, QTabWidget, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPixmap
import torch
from models.simple_cnn import SimpleCNN  
from models.mlp import MLP  
from utils.dataset import FER2013Dataset, get_transforms
from torch.utils.data import DataLoader
from scripts.train import train_model
from scripts.evaluate import evaluate_model, generate_metrics, plot_metrics
from utils.visualization import TrainingPlot
from pathlib import Path
from config import config
from threading import Thread
import json
from torchvision import transforms
from PIL import Image

class MainWindow(QMainWindow):
    status_signal = pyqtSignal(str)  # 定义信号，用于安全更新状态
    plot_signal = pyqtSignal(dict)  # 定义信号，用于更新训练图表

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.model = None
        self.is_training = False
        self.stop_training = [False]
        self.history = None
        
        self.initUI()
        self.status_signal.connect(self.update_status)
        self.plot_signal.connect(self.update_training_plot)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # 创建选项卡
        tabs = QTabWidget()
        tabs.addTab(self.createTrainingTab(), "Training")
        tabs.addTab(self.createPredictionTab(), "Prediction")
        main_layout.addWidget(tabs)
        
        central_widget.setLayout(main_layout)

    def createTrainingTab(self):
        training_widget = QWidget()
        layout = QVBoxLayout()
        
        # 模型配置组
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SimpleCNN", "MLP"])
        model_layout.addRow("Model Type:", self.model_selector)
        
        # 添加模型特定参数
        self.layer_sizes = QLineEdit("512,128")
        model_layout.addRow("Hidden Layer Sizes:", self.layer_sizes)
        
        self.dropout_rate = QDoubleSpinBox()
        self.dropout_rate.setRange(0, 1)
        self.dropout_rate.setSingleStep(0.1)
        self.dropout_rate.setValue(0.5)
        model_layout.addRow("Dropout Rate:", self.dropout_rate)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 训练参数组
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout()
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)
        training_layout.addRow("Learning Rate:", self.learning_rate)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(20)
        training_layout.addRow("Epochs:", self.epochs)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 256)
        self.batch_size.setValue(32)
        training_layout.addRow("Batch Size:", self.batch_size)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # 训练控制按钮
        button_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training_process)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # 训练可视化
        self.training_plot = TrainingPlot()
        layout.addWidget(self.training_plot)
        
        # 状态输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        training_widget.setLayout(layout)
        return training_widget

    def createPredictionTab(self):
        prediction_widget = QWidget()
        layout = QVBoxLayout()
        
        # 图像显示和上传
        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.predict_button = QPushButton("Predict Emotion")
        self.predict_button.clicked.connect(self.predict_emotion)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.predict_button)
        layout.addLayout(button_layout)
        
        # 预测结果显示
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        prediction_widget.setLayout(layout)
        return prediction_widget

    def stop_training_process(self):
        """停止训练进程"""
        if self.is_training:
            self.stop_training[0] = True
            self.update_status("Training process will be stopped after current epoch...")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def predict_emotion(self):
        if not hasattr(self, "image_path"):
            QMessageBox.warning(self, "Warning", "Please upload an image first!")
            return

        if not self.model:
            QMessageBox.warning(self, "Warning", "Please select and train a model first!")
            return

        # 图像预处理
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = Image.open(self.image_path)
        image_tensor = transform(image).unsqueeze(0)

        # 模型预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        predicted_emotion = emotions[predicted.item()]
        self.result_label.setText(f"Prediction Result: {predicted_emotion}")

    def start_training(self):
        # 获取用户输入
        try:
            model_type = self.model_selector.currentText()
            epochs = int(self.epochs.value())  
            batch_size = int(self.batch_size.value())  
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid hyperparameters!")
            return

        # 初始化模型
        if model_type == "SimpleCNN":
            self.model = SimpleCNN(num_classes=7)
        elif model_type == "MLP":
            self.model = MLP(input_size=48 * 48, output_size=7)

        # 数据加载
        try:
            print("Loading dataset...")  # 调试日志
            train_dataset = FER2013Dataset(
                csv_file="data/fer2013/fer2013.csv", usage_filter="Training"
            )
            val_dataset = FER2013Dataset(
                csv_file="data/fer2013/fer2013.csv", usage_filter="PublicTest"
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            print("Dataset loaded successfully.")  # 调试日志

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Dataset file not found! Please check the path.")
            return

        # 开始训练
        self.is_training = True
        self.stop_training[0] = False

        def train_task():
            try:
                self.status_signal.emit("Training task started...")  # 状态更新信号
                history = train_model(
                    self.model,
                    train_loader,
                    val_loader,
                    config=config,  # 传递 config 对象
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    status_callback=self.status_signal.emit,
                    stop_flag=self.stop_training
                )
                self.status_signal.emit("Training task completed!")  # 任务完成信号

                # 更新训练图表
                self.update_training_plot(history.history)

                # 评估模型
                val_results = evaluate_model(self.model, val_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                metrics = generate_metrics(val_results)
                plot_metrics(metrics, save_dir=Path("results"))

            except Exception as e:
                print(f"Error in training task: {e}")  # 调试日志
                self.status_signal.emit(f"Training error: {str(e)}")
                QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")

        print("Starting training thread...")  # 调试日志
        training_thread = Thread(target=train_task)
        training_thread.start()
        print("Training thread started.")  # 调试日志

    def update_training_plot(self, history):
        """更新训练图表"""
        if history:
            self.training_plot.update_plot(history)

    def update_status(self, message):
        """更新状态日志"""
        self.log_text.append(message)

class TrainWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, train_task):
        super().__init__()
        self.train_task = train_task

    def run(self):
        self.train_task()
        self.finished.emit()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
