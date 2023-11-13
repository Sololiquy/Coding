from PyQt5.QtWidgets import (QMainWindow, QApplication, 
                             QToolButton, QTableWidget, 
                             QTableWidgetItem, QFileDialog, QStyle)
                             
from PyQt5.QtCore import Qt, QSize


class mainWin(QMainWindow):
    def __init__(self, parent = None):
        super(mainWin, self).__init__(parent)
        self.setupUI()
        
    def setupUI(self):
        self.setGeometry(0, 0, 800, 600)
        self.setContentsMargins(10, 5, 10, 5)
        self.lb = QTableWidget()
        self.setCentralWidget(self.lb)
        self.create_toolbar()
        self.csv_file = ""
        self.csv_file_name = ""
        
    def create_toolbar(self):
        ### create toolbar
        self.toolbar = self.addToolBar("File")
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.setMovable(False)
        self.toolbar.setAllowedAreas(Qt.TopToolBarArea)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setStyleSheet("border: 0px; padding: 3px;")
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        
        ### open button
        open_icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        open_btn = QToolButton(text = "Open", icon = open_icon)
        open_btn.clicked.connect(self.open_file)
        open_btn.setStyleSheet("QToolButton:hover {background: #a5dcff;}")
        self.toolbar.addWidget(open_btn)
        
        ### open button
        save_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        save_btn = QToolButton(text = "Save", icon = save_icon)
        save_btn.clicked.connect(self.save_file)
        save_btn.setStyleSheet("QToolButton:hover {background: #a5dcff;}")
        self.toolbar.addWidget(save_btn) 
 
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet("font-size: 8pt; color: #888a85")
        self.statusbar.showMessage("Ready")
        
    def open_file(self):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open file', '', 
                                              "CSV Files (*.csv *.tsv *.txt);;All Files (*.*)")
        if fname:
            self.csv_file = fname
            self.load_csv(self.csv_file)
            self.statusbar.showMessage(f"{fname} loaded")

    def save_file(self):
        if self.lb.rowCount() < 1:
            return
        if self.csv_file != "":
            file_name = self.csv_file
        else:
            file_name = "*.csv"
            
        fname,_ = QFileDialog.getSaveFileName(self, 'Save file', file_name, 
                                              "CSV Files (*.csv *.tsv *.txt);;All Files (*.*)")
        if fname:
            self.save_csv(fname)
            self.csv_file = fname
            
    def save_csv(self, filename):
        rowtext = ""
        for row in range(self.lb.rowCount()-1):
            
            for column in range(self.lb.columnCount()-1):
                celltext = self.lb.item(row, column).text()
                rowtext += f"{celltext}\t"
            rowtext = rowtext.rstrip("\t")
            rowtext += "\n"
        with open(filename, "w") as f:
            f.write(rowtext)

    def load_csv(self, filename):
        self.csv_text = open(filename, "r").read()
        ### count tab / comma
        tab_counter = self.csv_text.splitlines()[0].count("\t")
        comma_counter = self.csv_text.splitlines()[0].count(",")
        if tab_counter > comma_counter:
            self.lb.setColumnCount(tab_counter + 1)
            delimiter = "\t"
        else:
            self.lb.setColumnCount(comma_counter + 1)
            delimiter = ","
        
        row = 0
        for listrow in self.csv_text.splitlines():
            self.lb.insertRow(row)
            rowtext = listrow.split(delimiter)
            column = 0
            for cell in rowtext:
                celltext = QTableWidgetItem(cell)
                self.lb.setItem(row, column, celltext)
                column += 1
            row += 1
            
        

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = mainWin()
    win.setWindowTitle("CSV Example")
    win.show()

    sys.exit(app.exec_())