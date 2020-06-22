# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.setEnabled(True)
        main_window.resize(1115, 696)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        main_window.setSizePolicy(sizePolicy)
        main_window.setAutoFillBackground(False)
        self.central_widget = QtWidgets.QWidget(main_window)
        self.central_widget.setObjectName("central_widget")
        self.loadpic = QtWidgets.QPushButton(self.central_widget)
        self.loadpic.setGeometry(QtCore.QRect(830, 520, 261, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadpic.sizePolicy().hasHeightForWidth())
        self.loadpic.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.loadpic.setFont(font)
        self.loadpic.setObjectName("loadpic")
        self.loadpic.setStyleSheet("border:0px")
        self.loadpic.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.displaypic = QtWidgets.QGraphicsView(self.central_widget)
        self.displaypic.setGeometry(QtCore.QRect(20, 20, 1081, 471))
        self.displaypic.setObjectName("displaypic")
        self.displaypic.setStyleSheet("border:0px")
        self.displayid = QtWidgets.QLineEdit(self.central_widget)
        self.displayid.setGeometry(QtCore.QRect(20, 520, 731, 121))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.displayid.sizePolicy().hasHeightForWidth())
        self.displayid.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.displayid.setFont(font)
        self.displayid.setObjectName("displayid")
        self.displayid.setStyleSheet("border:0px")
        self.identifypic = QtWidgets.QPushButton(self.central_widget)
        self.identifypic.setGeometry(QtCore.QRect(830, 590, 261, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.identifypic.sizePolicy().hasHeightForWidth())
        self.identifypic.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.identifypic.setFont(font)
        self.identifypic.setObjectName("identifypic")
        self.identifypic.setStyleSheet("border:0px")
        self.identifypic.setStyleSheet("background-color: rgb(255, 255, 255);")
        main_window.setCentralWidget(self.central_widget)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "银行卡识别"))
        self.loadpic.setText(_translate("main_window", "加  载"))
        self.identifypic.setText(_translate("main_window", "识  别"))
