import os
import sys
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QMessageBox, QApplication, QMainWindow

from AdvancedEAST.predict import predict_txt
from crnn.predict import recognition
from gui.main import Ui_main_window


class APP(QMainWindow,Ui_main_window):
    def __init__(self, parent=None):
        super(APP,self).__init__(parent)
        self.setupUi(self)
        self.loadpic.clicked.connect(self.loadpicture)
        self.identifypic.clicked.connect(self.identify)
        self.last = "D:/idcard/dataset/test/"
        self.model = ["D:/idcard/AdvancedEAST/saved_model/east_model.h5","D:/idcard/crnn/model/crnn_model.h5"]

    def display(self, picarray):
        picarray = cv2.cvtColor(picarray, cv2.COLOR_BGR2RGB)
        y, x, _ = picarray.shape
        bytespl = 3 * x
        frame = QImage(picarray.data, x, y, bytespl, QImage.Format_RGB888)
        pix = QGraphicsPixmapItem(QPixmap.fromImage(frame))
        displayscene = QGraphicsScene()
        displayscene.addItem(pix)
        self.displaypic.setScene(displayscene)

    def loadpicture(self):
        self.picname, _ = QFileDialog.getOpenFileName(None, "选择图片", self.last, "*.png;*.jpg;*.jpeg")
        self.last = os.path.split(self.picname)[0]
        self.picarray = cv2.imread(self.picname)
        self.displayid.setText("")
        self.display(self.picarray.copy())

    def identify(self):
        if not self.displaypic.scene():
            QMessageBox.information(None,
                                "提示!",
                                "请先加载一张图片！")
            return
        if not os.path.exists(self.model[0]):
            name, ext = QFileDialog.getOpenFileName(None, "选择AdvancedEAST模型", self.last, "*.h5")
            self.model[0] = name
        if not os.path.exists(self.model[1]):
            name, ext = QFileDialog.getOpenFileName(None, "选择CRNN模型", self.last, "*.h5")
            self.model[1] = name
        result = predict_txt(self.picname, self.model[0])
        resultarray = cv2.imread(self.picname)
        if len(result):
            array1 = np.array(result[0], dtype=int).reshape((4, 2))
            result=(np.min(array1[:, 0], axis=0),
                    np.min(array1[:, 1], axis=0),
                    np.max(array1[:, 0], axis=0),
                    np.max(array1[:, 1], axis=0))
            x = abs(result[0] - result[2])
            y = abs(result[1] - result[3])
            x0 = min(result[0],result[2])
            y0 = min(result[1],result[3])
            ipic = resultarray[y0: y0 + y, x0: x0 + x, :]
            self.display(ipic.copy())
            self.displayid.setText(recognition(ipic, (256, 32), self.model[1]))
        else:
            QMessageBox.critical(None,
                                 "提示!",
                                 "识别失败！")
