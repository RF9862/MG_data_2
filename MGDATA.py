from PyQt5 import QtWidgets, uic,QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QDialog
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import sys, os, getpass, shutil
from threading import Thread
import uuid
from AESCipher import AESCipher
from main_MG import mainMG as mgData

#uuid.getnode()
class MainPage(QMainWindow):
    def __init__(self):
        super(MainPage,self).__init__()     
        uic.loadUi(resource_path("main.ui"),self)
        self.cnt2=0
        self.totol2=0
        self.cnt1=0
        self.totol1=0
        self.initUI() 
    
    def initUI(self):
        self.lnk_tab2.setEnabled(False)
        self.btn_browse2.clicked.connect(self.btnSelectFile2)
        self.btn_process2.clicked.connect(self.btnProcessFile2)
        self.pbar_tab2.valueChanged.connect(self.onChangeValue2)
        self.lnk_tab2.clicked.connect(self.onClickLblTab2)
        self.single_done2.connect(self.progress2)
        # winScale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100 # get windows scale
        # self.btn_process2.setEnabled(False)
    def openOutput(self,path): 
        if path:  
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
    
    def btnSelectFile2(self):
        self.input = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        try:
            self.le_tab2.setText(self.input)   
        except: pass
        print("okay")
    def btnProcessFile2(self):
        t1=Thread(target=self.Operation2)
        t1.start()
        # t1.join()

    def process(self):
        self.lbl1_tab2.setText(f"scan starting ... ")  
        self.output = os.path.join(self.input, "results")
        ERR_DIR = "failed"
        result = mgData(self, self.input, self.output, ERR_DIR) # 1 means file name, and 0 means qrcode
        if result:
            # self.lbl1_tab2.setText(f"Done ...") 
            self.lbl1_tab2.setText("Result : Successfully processed "+ str(self.total2) +"%")
            self.lnk_tab2.setEnabled(True)
            self.lnk_tab2.setText("Go Output Folder")

        return None    

    def Operation2(self):
        
        if self.input == "":
            self.lbl1_tab2.setText("Wanring: Please Select Config File Exactly")
            return None        
        if os.path.isfile("C:/Program Files/Tesseract-OCR/tesseract.exe"):
            self.process()
        else:
            self.lbl1_tab2.setText("Wanring: Please check tesseract.exe in C:/Program FilesTesseract-OCR/")
            return None
    def onChangeValue2(self,val):
        self.pbar_tab2.setFormat(str(self.cnt2) + '/' + str(self.total2))

    single_done2 = pyqtSignal()
    @pyqtSlot()
    def progress2(self):
        self.pbar_tab2.setValue(int((self.cnt2/self.total2)*100))

    def openFolder2(self, path):
        # self.lbl1_tab2.setText("Result : Successfully processed "+ str(self.total2) +"%")
        self.path2=path
        self.openOutput(path)
    
    def onClickLblTab2(self):
        self.openFolder2(self.output)
        #QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

def window():
    app = QApplication(sys.argv)
    win = MainPage()
    win.show()
    sys.exit(app.exec_())

def windowValidate():
    app = QApplication(sys.argv)
    win = KeyWindow()
    win.show()
    sys.exit(app.exec_())


class KeyWindow(QMainWindow):
    def __init__(self):
        super(KeyWindow,self).__init__()
        self.node = str(uuid.getnode())
        uic.loadUi(resource_path("keyWindow.ui"),self)
        self.lbl_id.setText(self.node)
        self.initUI()

    def initUI(self):
        self.btn_submit.clicked.connect(self.onSubmit)
    
    def onSubmit(self):
        
        if self.txt_key.toPlainText():
            fp = open(logPath+'/.validate', 'w')
            fp.write(self.txt_key.toPlainText())
            fp.close()

            if validate():
                self.win = MainPage()
                self.win.show()
                self.hide()
            else:
                self.lbl_msg.setText("Invalid key")
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def validate():
    try:
        fp =open(logPath+'/.validate','rb')
        data = fp.read()
        fp.close()
    except: return False
    if data:
        try:
            c = AESCipher()
            if str(uuid.getnode()) == c.decrypt(data):
                return True
        except:
            pass
    return False
def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

logPath = f"C:/Users/{getpass.getuser()}/.mgData"
makedir(logPath)  

if not validate():
    windowValidate()
else:
    window()