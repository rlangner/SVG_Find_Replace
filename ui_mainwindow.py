# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTextBrowser, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(719, 580)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 701, 71))
        self.browseButton = QPushButton(self.groupBox)
        self.browseButton.setObjectName(u"browseButton")
        self.browseButton.setGeometry(QRect(10, 30, 81, 25))
        self.filePathEdit = QLineEdit(self.groupBox)
        self.filePathEdit.setObjectName(u"filePathEdit")
        self.filePathEdit.setGeometry(QRect(110, 30, 581, 25))
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 90, 701, 71))
        self.browseButton_2 = QPushButton(self.groupBox_2)
        self.browseButton_2.setObjectName(u"browseButton_2")
        self.browseButton_2.setGeometry(QRect(10, 30, 81, 25))
        self.filePathEdit_2 = QLineEdit(self.groupBox_2)
        self.filePathEdit_2.setObjectName(u"filePathEdit_2")
        self.filePathEdit_2.setGeometry(QRect(110, 30, 581, 25))
        self.runButton = QPushButton(self.centralwidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(10, 250, 141, 25))
        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(10, 290, 701, 261))
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 170, 701, 71))
        self.browseButton_3 = QPushButton(self.groupBox_3)
        self.browseButton_3.setObjectName(u"browseButton_3")
        self.browseButton_3.setGeometry(QRect(10, 30, 81, 25))
        self.filePathEdit_3 = QLineEdit(self.groupBox_3)
        self.filePathEdit_3.setObjectName(u"filePathEdit_3")
        self.filePathEdit_3.setGeometry(QRect(110, 30, 581, 25))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 719, 22))
        self.menubar.setDefaultUp(True)
        self.menubar.setNativeMenuBar(True)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Input SVG File", None))
        self.browseButton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Lookup SVG File", None))
        self.browseButton_2.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.runButton.setText(QCoreApplication.translate("MainWindow", u"Run Find and Replace", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Output SVG File", None))
        self.browseButton_3.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
    # retranslateUi

