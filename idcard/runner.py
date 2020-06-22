import sys
from PyQt5.QtWidgets import QApplication
from gui.app import APP

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = APP()
    main_app.show()
    sys.exit(app.exec_())