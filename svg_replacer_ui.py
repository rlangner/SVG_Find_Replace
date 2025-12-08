from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QObject, Signal
from ui_mainwindow import Ui_MainWindow
import os
import sys
from svg_replacer import *

class Stream(QObject):
    newText = Signal(str)

    def write(self, text):
        self.newText.emit(text)

    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()     # ← THIS IS THE CORRECT WAY
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

#        self.process = QProcess(self)
#        self.process.readyReadStandardOutput.connect(self.handle_stdout)
#        self.process.readyReadStandardError.connect(self.handle_stderr)

        self.stream = Stream()
        self.stream.newText.connect(self.append_text)
        sys.stdout = self.stream

        self.ui.browseButton.clicked.connect(self.pick_file_input)
        self.ui.browseButton_2.clicked.connect(self.pick_file_lookup)
        self.ui.browseButton_3.clicked.connect(self.pick_file_output)
        self.ui.runButton.clicked.connect(self.run_find_replace)

    def pick_file_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "SVG Files (*.svg)"
        )
        if file_path:
            self.ui.filePathEdit.setText(file_path)
            self.ui.filePathEdit_3.setText(os.path.dirname(file_path) + "/output.svg")

    def pick_file_lookup(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "SVG Files (*.svg)"
        )
        if file_path:
            self.ui.filePathEdit_2.setText(file_path)

    def pick_file_output(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "SVG Files (*.svg)"
        )
        if file_path:
            self.ui.filePathEdit_3.setText(file_path)

    def run_find_replace(self):
        input_file = self.ui.filePathEdit.text()
        lookup_file = self.ui.filePathEdit_2.text()
        output_file = self.ui.filePathEdit_3.text()
        print(input_file + "\n" + lookup_file)
        replace_groups_in_svg(input_file, lookup_file, output_file)

    def append_text(self, text):
        self.ui.textBrowser.insertPlainText(text)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()

