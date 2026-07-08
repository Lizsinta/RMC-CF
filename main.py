import os
from sys import argv, exit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize

from PyQt6.QtGui import QFont, QDoubleValidator
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QLabel, QLineEdit, QSizePolicy, \
    QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QWidget, QCheckBox, QSpinBox, QGridLayout, QMenuBar, QStatusBar, QDoubleSpinBox, QSpacerItem, QFrame, QComboBox
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

from scipy.spatial.distance import cdist

from lib.qtgraph import barPlotWidget
from lib.xafs import xanes_analysis, read_9809_xafs, read_dat
from lib.rmccf import Worker
from icon import get_icon

colors = list(mcolors.TABLEAU_COLORS.keys())

warnings.filterwarnings(
    "ignore",
    message="Input line .* contained no data and will not be counted towards",
    category=UserWarning
)

def color_convert(c, a, octal=True) -> list:
    if type(c) == str:
        c = list(mcolors.to_rgb(c))
    if octal:
        a *= 255
        if max(c) <= 1.0:
            color = [c[0] * 255, c[1] * 255, c[2] * 255, a]
        else:
            color = [c[0], c[1], c[2], a]
    else:
        if max(c) <= 1.0:
            color = [c[0], c[1], c[2], a]
        else:
            color = [c[0] / 255, c[1] / 255, c[2] / 255, a]
    return color

def line(x=np.array([]), y=np.array([]), c=(0, 0, 0), alpha=1.0, width=1.0, name=''):
        return pg.PlotDataItem(x, y, pen={'color': color_convert(c, alpha, octal=True), 'width': width}, name=name)


def bar(x=np.array([]), y=np.array([]), width=0.3, c=(0, 0, 0), alpha=1.0, name=''):
    return pg.BarGraphItem(x=x, height=y, width=width, brush=color_convert(c, alpha, octal=True), name=name)


class Intro(QDialog):
    def __init__(self, parent=None, file=''):
        super(Intro, self).__init__(parent)
        self.setWindowTitle('RMC-CF input')
        self.setWindowIcon(get_icon())
        self.setModal(True)
        self.resize(600, 600)
        self.selected_file = file
        self.element = ''
        self.snr = 50
        self.ratiop = np.array([])
        self.ratioa = np.array([])

        # Widget
        self.exp_label = QLabel('Experimental data analysis')
        self.exp_label.setFont(QFont('Arial', 10))

        self.file_label = QLabel('file：')
        self.file_label.setFont(QFont('Arial', 10))

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText('Select/enter file' if file == '' else file)
        self.file_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.select_btn = QPushButton('...')
        self.select_btn.setFixedSize(30, 30)
        self.select_btn.clicked.connect(self.select_file)

        self.exp_btn = QPushButton('Run')
        self.exp_btn.setObjectName('exp')
        self.exp_btn.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.exp_btn.setFixedSize(80, 35)
        self.exp_btn.clicked.connect(self.on_run_click)

        self.model_label = QLabel('Simulated data analysis')
        self.model_label.setFont(QFont("Arial", 10))

        self.ele_label = QLabel('Element: ')
        self.ele_label.setFont(QFont('Arial', 10))

        self.ele_comboBox = QComboBox()
        self.ele_comboBox.addItems(os.listdir(os.getcwd() + '\\ref'))
        self.ele_comboBox.setCurrentText('Cu')
        self.ele_comboBox.currentIndexChanged.connect(self.select_ele)

        self.snr_label = QLabel('S/N: ')
        self.snr_label.setFont(QFont("Arial", 10))

        self.snr_spinBox = QSpinBox()
        self.snr_spinBox.setMinimum(30)
        self.snr_spinBox.setMaximum(100)
        self.snr_spinBox.setSingleStep(10)
        self.snr_spinBox.setValue(50)
        self.snr_spinBox.setFont(QFont("Arial", 10))

        species = os.listdir(os.getcwd() + f'\\ref\\{self.ele_comboBox.currentText()}')
        self.speciesLabel = np.zeros(len(species), dtype=QLabel)
        for i in range(self.speciesLabel.size):
            self.speciesLabel[i] = QLabel(species[i].split('.')[0])
            self.speciesLabel[i].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.speciesLabel[i].setFont(QFont('Arial', 10))

        self.weight_label = QLabel('Weight')
        self.weight_label.setFont(QFont('Arial', 10))
        self.weight_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.weightLine = np.zeros(4, dtype=QLineEdit)
        self.axisLine = np.zeros((4, self.speciesLabel.size), dtype=QLineEdit)
        self.float_validator = QDoubleValidator(0.0, 10, 3)
        self.float_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        ratiop = np.array([0.7, 0.3, 0, 0])
        ratioa = np.array([[1.8, 0.9, 0.0],
                           [0.0, 0.3, 0.5],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]])
        for i in range(self.weightLine.size):
            self.weightLine[i] = QLineEdit()
            self.weightLine[i].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.weightLine[i].setValidator(self.float_validator)
            self.weightLine[i].setText(f'{ratiop[i]:.3f}')
            for j in range(self.speciesLabel.size):
                self.axisLine[i][j] = QLineEdit()
                self.axisLine[i][j].setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.axisLine[i][j].setValidator(self.float_validator)
                self.axisLine[i][j].setText(f'{ratioa[i][j]:.3f}')

        self.model_btn = QPushButton('Run')
        self.model_btn.setObjectName('model')
        self.model_btn.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.model_btn.setFixedSize(80, 35)
        self.model_btn.clicked.connect(self.on_run_click)

        # Layout
        exp_file_layout = QHBoxLayout()
        exp_file_layout.addWidget(self.file_label)
        exp_file_layout.addWidget(self.file_edit)
        exp_file_layout.addWidget(self.select_btn)
        exp_file_layout.setSpacing(10)
        exp_file_layout.setContentsMargins(20, 20, 20, 10)

        exp_btn_layout = QHBoxLayout()
        exp_btn_layout.addStretch()
        exp_btn_layout.addWidget(self.exp_btn)
        exp_btn_layout.addStretch()
        exp_btn_layout.setContentsMargins(20, 10, 20, 20)

        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        h_line.setFrameShadow(QFrame.Shadow.Sunken)
        h_line.setLineWidth(1)

        ele_layout = QHBoxLayout()
        ele_layout.addStretch()
        ele_layout.addWidget(self.ele_label)
        ele_layout.addWidget(self.ele_comboBox)
        ele_layout.addStretch()
        ele_layout.addWidget(self.snr_label)
        ele_layout.addWidget(self.snr_spinBox)
        ele_layout.setContentsMargins(20, 20, 20, 10)

        self.species_layout = QHBoxLayout()
        self.species_layout.addWidget(self.weight_label)
        for i in range(self.speciesLabel.size):
            self.species_layout.addWidget(self.speciesLabel[i])

        self.ratio_layout = QGridLayout()
        for i in range(self.weightLine.size):
            self.ratio_layout.addWidget(self.weightLine[i], i, 0, 1, 1)
            for j in range(self.speciesLabel.size):
                self.ratio_layout.addWidget(self.axisLine[i][j], i, j + 1, 1, 1)

        model_btn_layout = QHBoxLayout()
        model_btn_layout.addStretch()
        model_btn_layout.addWidget(self.model_btn)
        model_btn_layout.addStretch()
        model_btn_layout.setContentsMargins(20, 10, 20, 20)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.exp_label)
        main_layout.addLayout(exp_file_layout)
        main_layout.addLayout(exp_btn_layout)
        main_layout.addWidget(h_line)
        main_layout.addWidget(self.model_label)
        main_layout.addLayout(ele_layout)
        main_layout.addLayout(self.species_layout)
        main_layout.addLayout(self.ratio_layout)
        main_layout.addLayout(model_btn_layout)
        self.setLayout(main_layout)

    def select_file(self):
        file_path = QFileDialog.getOpenFileName(
            self,
            'select file...',
            filter='*.dat',
            directory=os.path.dirname(self.selected_file),
        )[0]
        if file_path:
            self.file_edit.setText(file_path)
            self.selected_file = file_path.strip()

    def select_ele(self):
        species = os.listdir(os.getcwd() + f'\\ref\\{self.ele_comboBox.currentText()}')
        dif = len(species) - self.speciesLabel.size
        if dif > 0:
            for j in range(dif):
                self.speciesLabel = np.append(self.speciesLabel, QLabel(''))
                self.speciesLabel[-1].setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.speciesLabel[-1].setFont(QFont('Arial', 10))
                self.species_layout.addWidget(self.speciesLabel[-1])
                ratioLine = np.array([QLineEdit('0.000') for _ in range(self.weightLine.size)])
                self.axisLine = np.hstack((self.axisLine, ratioLine[:, None]))
                for i in range(self.weightLine.size):
                    self.axisLine[i][-1].setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.axisLine[i][-1].setFont(QFont('Arial', 10))
                    self.axisLine[i][-1].setValidator(self.float_validator)
                    self.ratio_layout.addWidget(self.axisLine[i][-1], i, self.axisLine.shape[1], 1, 1)
        elif dif < 0:
            for j in range(-dif):
                self.species_layout.removeWidget(self.speciesLabel[-1])
                self.speciesLabel[-1].deleteLater()
                self.speciesLabel = self.speciesLabel[:-1]
                for i in range(self.weightLine.size):
                    self.ratio_layout.removeWidget(self.axisLine[i][-1])
                    self.axisLine[i][-1].deleteLater()
                self.axisLine = self.axisLine[:, :-1]
        for i in range(self.speciesLabel.size):
            self.speciesLabel[i].setText(species[i].split('.')[0])


    def on_run_click(self):
        target = self.sender()
        if target.objectName() == 'exp':
            if not self.selected_file == '':
                if not self.file_edit.text() == '':
                    self.selected_file = self.file_edit.text().strip()
                if os.path.exists(self.selected_file):
                    self.accept()
                else:
                    QMessageBox.information(self, 'Error', 'Wrong file path', QMessageBox.StandardButton.Ok)
            else:
                self.file_edit.setPlaceholderText("Please select a file")
        elif target.objectName() == 'model':
            self.element = self.ele_comboBox.currentText()
            self.snr = self.snr_spinBox.value()
            ratiop = np.zeros(self.weightLine.size, dtype=float)
            ratioa = np.zeros(self.axisLine.shape, dtype=float)
            for i in range(self.weightLine.size):
                ratiop[i] = float(self.weightLine[i].text())
                for j in range(self.speciesLabel.size):
                    ratioa[i][j] = float(self.axisLine[i][j].text())
            self.ratioa = ratioa[ratiop > 10e-6]
            ratiop = ratiop[ratiop > 10e-6]
            self.ratiop = ratiop / np.sum(ratiop)
            self.accept()
        else:
            pass


class MainWindow(QMainWindow):
    def __init__(self, file, element='', ratio_axis=np.array([]), ratio_plane=np.array([]), snr=50):
        super(MainWindow, self).__init__()
        self.setObjectName("RMC")
        self.setWindowTitle("RMC-CF")
        self.setWindowIcon(get_icon())
        self.resize(1280, 720)
        # font = QFont()
        # font.setFamily("Adobe Arabic")
        # font.setPointSize(16)
        # font.setBold(False)
        # font.setWeight(50)
        # self.setFont(font)
        self.cw = QWidget()
        self.setCentralWidget(self.cw)
        self.glay = QGridLayout()
        self.cw.setLayout(self.glay)

        # menubar&statusbar initialization
        self.menubar = QMenuBar(parent=self)
        self.menubar.setObjectName("menubar")
        self.menufile = self.menubar.addMenu('file')
        self.menufile.setObjectName("menufile")
        self.menuoption = self.menubar.addMenu('option')
        self.menuoption.setObjectName("menuoption")
        self.setMenuBar(self.menubar)
        # self.readAction = self.menufile.addAction('read')
        # self.readAction.setObjectName("readAction")
        self.loadAction = self.menufile.addAction('load')
        self.loadAction.setObjectName("loadAction")
        #self.reloadAction = self.menuoption.addAction('reload')
        #self.reloadAction.setObjectName("reloadAction")
        #self.analysisAction = self.menuoption.addAction('analysis')
        #self.analysisAction.setObjectName("analysisAction")
        self.contourAction = self.menuoption.addAction('contourMap')
        self.contourAction.setObjectName("contourAction")

        self.statusbar = QStatusBar(parent=self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # fitting plotting initialization
        self.plot = pg.PlotWidget(background=(255, 255, 255, 255))
        self.glay.addWidget(self.plot, 0, 0, 1, 2)
        self.plot.addLegend((0, self.plot.size().height() * 0.05), labelTextSize='12pt', labelTextColor='k')
        name = ['experiment', 'fitting']
        c = ['k', 'r']
        self.line = np.array([line([0], [0], c=c[_], name=name[_], width=2) for _ in range(2)])
        for i in range(self.line.size):
            self.plot.addItem(self.line[i])

        self.thread = Worker(file=file, element=element, ratio_axis=ratio_axis, ratio_plane=ratio_plane, snr=snr)
        self.model_flag = True if not element == '' else False
        self.reload_flag = self.read()

        # information initialization
        self.stdInfoWidget = QWidget(parent=self.cw)
        self.stdInfoLayout = QGridLayout(self.stdInfoWidget)
        self.glay.addWidget(self.stdInfoWidget, 0, 2, 1, 1)
        self.stdInfo1Label = QLabel(parent=self.stdInfoWidget, text='ref')
        self.stdInfo2Label = QLabel(parent=self.stdInfoWidget, text='average-standard error')
        self.stdInfo1Label.setFixedHeight(40)
        self.stdInfo2Label.setFixedHeight(40)
        self.stdInfoLayout.addWidget(self.stdInfo1Label, 0, 0, 1, 1)
        self.stdInfoLayout.addWidget(self.stdInfo2Label, 0, 1, 1, 2)

        self.stdLabel = np.zeros(len(self.thread.species), dtype=QCheckBox)
        self.stdLine = np.zeros(len(self.thread.species), dtype=QLineEdit)
        self.stdAddrBtn = np.zeros(len(self.thread.species), dtype=QPushButton)
        self.stdAddr = [''] * len(self.thread.species)
        for i in range(len(self.thread.species)):
            self.stdLabel[i] = QCheckBox(parent=self.stdInfoWidget)
            self.stdLabel[i].setChecked(True)
            self.stdLabel[i].setObjectName(f'stdLabel{i}')
            self.stdLabel[i].setText(self.thread.species[i])
            self.stdInfoLayout.addWidget(self.stdLabel[i], i + 1, 0, 1, 1)

            self.stdLine[i] = QLineEdit(parent=self.stdInfoWidget)
            self.stdLine[i].setReadOnly(True)
            self.stdLine[i].setObjectName(f'stdLine{i}')
            self.stdInfoLayout.addWidget(self.stdLine[i], i + 1, 1, 1, 1)

            self.stdAddrBtn[i] = QPushButton(parent=self.stdInfoWidget, text='...')
            self.stdAddrBtn[i].setFixedWidth(30)
            self.stdAddrBtn[i].setObjectName(f'stdAddr{i}')
            self.stdInfoLayout.addWidget(self.stdAddrBtn[i], i + 1, 2, 1, 1)

        self.stdPlusBtn = QPushButton(parent=self.stdInfoWidget, text='+')
        self.stdPlusBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.stdPlusBtn.setFixedHeight(30)
        self.stdInfoLayout.addWidget(self.stdPlusBtn, self.stdAddrBtn.size + 1, 0, 1, 3)
        self.stdLineCount = self.stdLine.size

        self.stdInfoSpacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.stdInfoLayout.addItem(self.stdInfoSpacerItem, self.stdAddrBtn.size + 2, 0, 1, 3)
        for i in range(self.stdAddrBtn.size + 2):
            self.stdInfoLayout.setRowStretch(i, 1)
        self.stdInfoLayout.setSpacing(20)

        # parameter initialization
        self.paramInfoWidget = QWidget(parent=self.cw)
        self.paramInfoLayout = QGridLayout(self.paramInfoWidget)
        self.glay.addWidget(self.paramInfoWidget, 0, 3, 1, 1)

        self.tauLabel = QLabel(parent=self.paramInfoWidget, text="tau: ")
        self.tauLabel.setFixedHeight(40)
        self.tauLabel.setObjectName("tauLabel")
        self.paramInfoLayout.addWidget(self.tauLabel, 0, 0, 1, 1)
        self.tauWidget = QWidget(parent=self.paramInfoWidget)
        self.tauLayout = QHBoxLayout(self.tauWidget)
        self.paramInfoLayout.addWidget(self.tauWidget, 0, 1, 1, 1)
        self.tauLayout.setSpacing(0)
        self.tauLayout.setContentsMargins(0, 0, 0, 0)
        self.tauDBox = QDoubleSpinBox(parent=self.tauWidget)
        self.tauDBox.setDecimals(1)
        self.tauDBox.setMaximum(10.0)
        self.tauDBox.setProperty("value", 1.0)
        self.tauDBox.setObjectName("tauDBox")
        self.tauLayout.addWidget(self.tauDBox)
        self.taudotLabel = QLabel(parent=self.tauWidget, text=" ^- ")
        self.taudotLabel.setObjectName("taudotLabel")
        self.taudotLabel.setFixedHeight(40)
        self.tauLayout.addWidget(self.taudotLabel)
        self.tauFBox = QSpinBox(parent=self.tauWidget)
        self.tauFBox.setProperty("value", 8)
        self.tauFBox.setObjectName("tauFBox")
        self.tauLayout.addWidget(self.tauFBox)
        self.tauAtButton = QPushButton(parent=self.tauWidget, text='M')
        self.tauAtButton.setObjectName("tauAtButton")
        self.tauAtButton.setFixedWidth(30)
        self.tauLayout.addWidget(self.tauAtButton)
        tauSpacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.tauLayout.addItem(tauSpacerItem)

        self.blocksLabel = QLabel(parent=self.paramInfoWidget, text="Blocks: ")
        self.blocksLabel.setObjectName("blocksLabel")
        self.paramInfoLayout.addWidget(self.blocksLabel, 1, 0, 1, 1)
        self.blocksLine = QSpinBox(parent=self.paramInfoWidget)
        self.blocksLine.setMinimum(1)
        self.blocksLine.setMaximum(1000)
        self.blocksLine.setProperty("value", 100)
        self.blocksLine.setObjectName("blocksLine")
        self.paramInfoLayout.addWidget(self.blocksLine, 1, 1, 1, 1)

        self.stepSizeLabel = QLabel(parent=self.paramInfoWidget, text="Step size: ")
        self.stepSizeLabel.setObjectName("stepSizeLabel")
        self.paramInfoLayout.addWidget(self.stepSizeLabel, 2, 0, 1, 1)
        self.stepSizeLine = QSpinBox(parent=self.paramInfoWidget)
        self.stepSizeLine.setMinimum(1)
        self.stepSizeLine.setMaximum(1000)
        self.stepSizeLine.setProperty("value", 10)
        self.stepSizeLine.setDisplayIntegerBase(1)
        self.stepSizeLine.setObjectName("stepSizeLine")
        self.paramInfoLayout.addWidget(self.stepSizeLine, 2, 1, 1, 1)

        self.walkRangeLabel = QLabel(parent=self.paramInfoWidget, text="Walk range: ")
        self.walkRangeLabel.setObjectName("walkRangeLabel")
        self.paramInfoLayout.addWidget(self.walkRangeLabel, 3, 0, 1, 1)
        self.walkRangeBox = QDoubleSpinBox(parent=self.paramInfoWidget)
        self.walkRangeBox.setDecimals(3)
        self.walkRangeBox.setMaximum(1.0)
        self.walkRangeBox.setMinimum(0.001)
        self.walkRangeBox.setProperty("value", 0.01)
        self.walkRangeBox.setObjectName("walkRangeBox")
        self.paramInfoLayout.addWidget(self.walkRangeBox, 3, 1, 1, 1)

        self.walkStepLabel = QLabel(parent=self.paramInfoWidget, text="Walk step: ")
        self.walkStepLabel.setObjectName("walkStepLabel")
        self.paramInfoLayout.addWidget(self.walkStepLabel, 4, 0, 1, 1)
        self.walkStepBox = QDoubleSpinBox(parent=self.paramInfoWidget)
        self.walkStepBox.setDecimals(3)
        self.walkStepBox.setMaximum(1.0)
        self.walkStepBox.setMinimum(0.001)
        self.walkStepBox.setProperty("value", 0.01)
        self.walkStepBox.setObjectName("walkStepBox")
        self.paramInfoLayout.addWidget(self.walkStepBox, 4, 1, 1, 1)

        self.countLabel = QLabel(parent=self.paramInfoWidget, text="Count: ")
        self.countLabel.setObjectName("countLabel")
        self.paramInfoLayout.addWidget(self.countLabel, 5, 0, 1, 1)
        self.countLine = QLineEdit(parent=self.paramInfoWidget)
        self.countLine.setReadOnly(True)
        self.countLine.setObjectName("countLine")
        self.paramInfoLayout.addWidget(self.countLine, 5, 1, 1, 1)

        self.timeLabel = QLabel(parent=self.paramInfoWidget, text="Time: ")
        self.timeLabel.setObjectName("timeLabel")
        self.paramInfoLayout.addWidget(self.timeLabel, 6, 0, 1, 1)
        self.timeLine = QLineEdit(parent=self.paramInfoWidget)
        self.timeLine.setReadOnly(True)
        self.timeLine.setObjectName("timeLine")
        self.paramInfoLayout.addWidget(self.timeLine, 6, 1, 1, 1)

        self.R2Label = QLabel(parent=self.paramInfoWidget, text="R /best R: ")
        self.R2Label.setObjectName("R2Label")
        self.paramInfoLayout.addWidget(self.R2Label, 7, 0, 1, 1)
        self.R2Line = QLineEdit(parent=self.paramInfoWidget)
        self.R2Line.setReadOnly(True)
        self.R2Line.setObjectName("R2Line")
        self.paramInfoLayout.addWidget(self.R2Line, 7, 1, 1, 1)

        self.R2BLabel = QLabel(parent=self.paramInfoWidget, text="accept rate: ")
        self.R2BLabel.setObjectName("R2BLabel")
        self.paramInfoLayout.addWidget(self.R2BLabel, 8, 0, 1, 1)
        self.R2BLine = QLineEdit(parent=self.paramInfoWidget)
        self.R2BLine.setReadOnly(True)
        self.R2BLine.setObjectName("R2BLine")
        self.paramInfoLayout.addWidget(self.R2BLine, 8, 1, 1, 1)

        # for i in range(7):
        #     self.paramInfoLayout.setRowStretch(i, 1)

        paramInfoSpacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.paramInfoLayout.addItem(paramInfoSpacerItem, 9, 0, 1, 2)
        self.paramInfoLayout.setSpacing(20)

        # erange initailization
        self.erangeWidget = QWidget(parent=self.cw)
        self.erangeLayout = QHBoxLayout(self.erangeWidget)
        self.glay.addWidget(self.erangeWidget, 1, 0, 1, 1)

        self.eminLabel = QLabel(parent=self.erangeWidget, text="Energy: min ")
        self.eminLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eminLabel.setObjectName("eminLabel")
        self.erangeLayout.addWidget(self.eminLabel)
        self.eminLine = QDoubleSpinBox(parent=self.erangeWidget)
        self.eminLine.setDecimals(1)
        self.eminLine.setMaximum(30000.0)
        self.eminLine.setObjectName("eminLine")
        self.erangeLayout.addWidget(self.eminLine)
        self.emaxLabel = QLabel(parent=self.erangeWidget, text=" max ")
        self.emaxLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emaxLabel.setObjectName("emaxLabel")
        self.erangeLayout.addWidget(self.emaxLabel)
        self.emaxLine = QDoubleSpinBox(parent=self.erangeWidget)
        self.emaxLine.setDecimals(1)
        self.emaxLine.setMaximum(30000.0)
        self.emaxLine.setObjectName("emaxLine")
        self.erangeLayout.addWidget(self.emaxLine)

        # Button initialization
        self.buttonWidget = QWidget(parent=self.cw)
        self.buttonLayout = QHBoxLayout(self.buttonWidget)
        self.glay.addWidget(self.buttonWidget, 1, 2, 1, 2)

        self.startButton = QPushButton(parent=self.buttonWidget, text='start')
        self.startButton.setEnabled(True)
        self.startButton.setObjectName("startButton")
        self.buttonLayout.addWidget(self.startButton)
        self.endButton = QPushButton(parent=self.buttonWidget, text='end')
        self.endButton.setEnabled(False)
        self.endButton.setObjectName("endButton")
        self.buttonLayout.addWidget(self.endButton)
        # buttonSpacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # self.buttonLayout.addItem(buttonSpacerItem)

        # distribution bar plotting initialization
        self.barWidget = QWidget(parent=self.cw)
        self.barLayout = QHBoxLayout(self.barWidget)
        self.glay.addWidget(self.barWidget, 2, 0, 1, 4)

        self.stdBarWidget = np.zeros(len(self.thread.species), dtype=barPlotWidget)
        self.stdBarPlot = np.zeros(len(self.thread.species), dtype=pg.BarGraphItem)
        for i in range(len(self.thread.species)):
            self.stdBarWidget[i] = barPlotWidget(background=(255, 255, 255, 255))
            self.stdBarWidget[i].setTitle(self.thread.species[i], color='#000000', size='20pt')
            self.stdBarPlot[i] = bar([], [], c='k', name=self.thread.species[i], width=0.05)
            self.stdBarWidget[i].addItem(self.stdBarPlot[i])
            self.stdBarWidget[i].plotItem.autoBtn.clicked.connect(self.stdBarWidget[i].auto_range)
            self.barLayout.addWidget(self.stdBarWidget[i], 1)
            # self.barLayout.setStretch(i, 1)
        # stdBarSpacer = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.barLayout.addSpacerItem(stdBarSpacer)
        self.barLayout.setSpacing(2)  # 间距
        self.barLayout.setContentsMargins(0, 0, 0, 0)

        self.glay.setColumnStretch(0, 1)
        self.glay.setColumnStretch(1, 1)
        self.glay.setColumnStretch(2, 1)
        self.glay.setRowStretch(0, 15)
        self.glay.setRowStretch(1, 1)
        self.glay.setRowStretch(2, 10)

        # event connection
        #self.readAction.triggered.connect(self.read)
        self.loadAction.triggered.connect(self.load)
        #self.reloadAction.triggered.connect(self.reload_event)
        # self.tauATAction.changed.connect(self.auto_tau_switch)
        #self.analysisAction.triggered.connect(self.analysis_event)
        self.contourAction.triggered.connect(self.contourMap)
        self.startButton.clicked.connect(self.start)
        self.endButton.clicked.connect(self.end)
        self.tauDBox.valueChanged.connect(self.tau_change_m)  # D for decimal
        self.tauFBox.valueChanged.connect(self.tau_change_m)  # F for factor
        self.tauAtButton.clicked.connect(self.auto_tau_switch)
        self.blocksLine.valueChanged.connect(self.blocks_change)
        self.stepSizeLine.valueChanged.connect(self.step_size_change)
        self.walkRangeBox.valueChanged.connect(self.walk_param_change)
        self.walkStepBox.valueChanged.connect(self.walk_param_change)
        for i in range(self.thread.std.size):
            self.stdLabel[i].clicked.connect(self.std_check_event)
            self.stdAddrBtn[i].clicked.connect(self.ref_file_change)
        self.stdPlusBtn.clicked.connect(self.ref_line_add)
        self.eminLine.editingFinished.connect(self.energy_select_event)
        self.emaxLine.editingFinished.connect(self.energy_select_event)

        # GUI refreash timer
        self.refresh_timer = QTimer()
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(20)
        self.refresh_timer.timeout.connect(self._plotting)
        self.timer_busy = False

        # run RMC on another thread
        self.thread.sig_warning.connect(self.warning_window)
        self.thread.sig_statusbar.connect(self.show_massage)
        self.thread.sig_change_tau.connect(self.tau_change_a)
        # self.thread.sig_autoend.connect(self.end)
        self.thread.sig_flush.connect(self.updatePlot)

        # initialize GUI with RMC setting
        self.blocksLine.setValue(self.thread.nblock)
        self.stepSizeLine.setValue(self.thread.step_size)
        self.tau_change_a(self.thread.tau)
        self.endButton.setEnabled(False)
        self.tauDBox.setSingleStep(0.1)
        self.line[0].setData(x=self.thread.energy, y=-np.log(self.thread.exper0))
        self.eminLine.setValue(self.thread.emin)
        self.emaxLine.setValue(self.thread.emax)
        for i in np.arange(self.stdLabel.size):
            self.stdLabel[i].setEnabled(True)
            self.stdAddr[i] = os.getcwd() +f'\\ref\\{self.thread.element}\\{self.thread.species[i]}.dat'
        if self.reload_flag:
            self.reload()
        self.thread.plot_flag = True
        self.updatePlot()
        self.show()

    def read(self):
        # read target file
        reload_flag = False
        self.statusbar.showMessage('Reading', 0)
        # if reading success, load it on gui and initialize RMC
        # search for existing result, decide continue or start a new run
        folder_fname = self.thread.folder_save + '/' + self.thread.fname
        if os.path.exists(folder_fname + r'_ratio.dat'):
            if self.choice_window():
                reload_flag = True
                self.thread.load()
                self.thread.init(ratio_reset=False)
            else:
                name = folder_fname + r'_ratio.dat'
                i = 0
                while os.path.exists(name):
                    i += 1
                    name = folder_fname + f'_ratio_{i}.dat'
                os.rename(folder_fname + '_ratio.dat', folder_fname + f'_ratio_{i}.dat')
                if os.path.exists(folder_fname + r'_fitting.dat'):
                    os.rename(folder_fname + r'_fitting.dat', folder_fname + f'_fitting_{i}.dat')
                if os.path.exists(folder_fname + r'_rfactor.npy'):
                    os.rename(folder_fname + r'_rfactor.npy', folder_fname + f'_rfactor_{i}.npy')
                if os.path.exists(folder_fname + r'_tau.npy'):
                    os.rename(folder_fname + r'_tau.npy', folder_fname + f'_tau_{i}.npy')
                if self.model_flag:
                    self.thread.build_model()
                else:
                    self.thread.read()
                self.thread.init()
        else:
            if self.model_flag:
                self.thread.build_model()
            else:
                self.thread.read()
            self.thread.init()
        print(self.thread.energy.size, self.thread.edge.size)
        self.plot.setXRange(self.thread.energy[self.thread.edge[0]],
                            self.thread.energy[self.thread.edge[-1]], padding=0.1)
        self.statusbar.showMessage('Done!', 3000)
        return reload_flag

    def load(self):
        file_path = QFileDialog.getOpenFileName(
            self,
            'select file...',
            filter='*_ratio*.dat, *_fitting*.dat',
            directory=self.thread.folder,
        )[0]
        if file_path:
            self.thread.folder = os.path.dirname(file_path)
            file_path = file_path.replace('\\', '/')
            fname = file_path.split('/')[-1].split('.')[0]
            if fname.find('ratio') >= 0:
                dtype = '_ratio'
            elif fname.find('fitting') >= 0:
                dtype = '_fitting'
            self.thread.fname = fname.split(dtype)[0]
            self.thread.suffix = fname.split(dtype)[1]
            self.thread.load()
            if not self.thread.std.size == self.stdLineCount:
                self.ref_num_change()
            self.thread.init(ratio_reset=False)
            self.blocksLine.setReadOnly(True)

            for i in self.thread.std:
                self.stdLabel[i].setChecked(True)
                self.stdAddr[i] = self.thread.folder + '/' + self.thread.fname + f'_fitting{self.thread.suffix}.dat'
                self.stdLabel[i].setText(f'{self.thread.species[i]}')
            for i in range(self.stdAddrBtn.size):
                self.stdAddrBtn[i].setEnabled(False)
                #self.stdLabel[i].clicked.disconnect(self.std_check_event)
                self.stdLabel[i].setEnabled(False)
            self.stdPlusBtn.setEnabled(False)

    def reload(self):
        self.blocksLine.setValue(self.thread.nblock)
        self.blocksLine.setReadOnly(True)
        self.stepSizeLine.setValue(self.thread.step_size)
        self.tau_change_a(self.thread.tau)
        self.countLine.setText(f'{self.thread.step_count:d}')
        self.timeLine.setText(f'{int(self.thread.timestamp):d}')
        self.r_display(self.thread.r_factor, self.thread.r_factor_best)
        self.walkRangeBox.blockSignals(True)
        self.walkStepBox.blockSignals(True)
        self.walkRangeBox.setValue(self.thread.mvRange / self.thread.mvReso)
        self.walkStepBox.setValue(1 / self.thread.mvReso)
        self.walkRangeBox.blockSignals(False)
        self.walkStepBox.blockSignals(False)

        idx = np.arange(self.stdLabel.size)
        for i in idx[~np.isin(idx, self.thread.std)]:
            self.stdLabel[i].setChecked(False)
            self.stdBarPlot[i].setOpts(x=[], height=[])
        for i in self.thread.std:
            self.stdLabel[i].setChecked(True)
            self.stdAddr[i] = self.thread.folder + '/' + self.thread.fname + f'_fitting{self.thread.suffix}.dat'
            self.stdLabel[i].setText(f'{self.thread.species[i]}')
        for i in range(self.stdAddrBtn.size):
            self.stdAddrBtn[i].setEnabled(False)
        for i in idx:
            self.stdLabel[i].setEnabled(False)

    def start(self):
        # pushing start button, RMC thread start
        self.statusbar.showMessage('Running', 0)
        for i in np.arange(self.stdLabel.size):
            self.stdLabel[i].setEnabled(False)
            self.stdAddrBtn[i].setEnabled(False)
        self.stdPlusBtn.setEnabled(False)
        #self.readAction.setEnabled(False)
        #self.reloadAction.setEnabled(False)
        self.endButton.setEnabled(True)
        self.startButton.setEnabled(False)
        self.blocksLine.setReadOnly(True)
        self.thread.flag = True
        self.thread.start()

    def end(self):
        # pushing end button, RMC thread end by disable the flag in thread
        self.statusbar.showMessage('Saving...', 0)
        if self.thread.flag:
            self.thread.flag = False
            if self.thread.isRunning():
                self.thread.wait()
            print('iteration ended')

            self.thread.save()
            print('result wrote')
            self.statusbar.showMessage('Done!', 3000)
            self.endButton.setEnabled(False)
            self.startButton.setEnabled(True)
            #self.readAction.setEnabled(True)

    def updatePlot(self):
        if self.timer_busy:
            return
        self.refresh_timer.start()
        self.timer_busy = True

    def _plotting(self):
        # plotting fitting spectrum, cal distribution of ratio of std. and plot them
        self.countLine.setText('%d' % self.thread.step_count)
        self.timeLine.setText('%d' % self.thread.timestamp)
        self.R2Line.setText(f'{self.thread.r_factor:.8e} / {self.thread.r_factor_best:.8e}')
        self.R2BLine.setText(f'{self.thread.ac_rate:.4f}')
        if self.thread.plot_flag:
            xsize = self.line[1].xData.size
            _fit = -np.log(self.thread.fit)
            if self.thread.edge.size == _fit.size:
                self.line[1].setData(x=self.thread.energy[self.thread.edge], y=_fit)
            else:
                print(self.thread.edge.size, _fit.size)
            if not xsize == _fit.size:
                self.plot.setXRange(self.thread.emin, self.thread.emax, padding=0.1)
            _ratio = self.thread.ratio
            composition = self.thread.ratio.mean(0)
            for i in range(self.thread.std.size):
                content = _ratio[:, i]
                rdf = np.unique(np.round(content, 1), return_counts=True)
                if rdf[0].size == 1 or (rdf[0].max() - rdf[0][rdf[0] > 0].min()) < 0.5:
                    digit = 2
                else:
                    digit = 1
                rdf = np.unique(np.round(content, digit), return_counts=True)
                self.stdBarPlot[self.thread.std[i]].setOpts(x=rdf[0], height=rdf[1], width=0.5 / 10 ** digit)
                if self.stdBarWidget[self.thread.std[i]].autoBtn_flag:
                    min = rdf[0][rdf[0] > 0].min() if rdf[0].size > 1 else 0
                    self.stdBarWidget[self.thread.std[i]].setXRange(min, rdf[0].max(), padding=0.1)
                std = content[content > 1e-6].std() if (content > 1e-6).any() else 0
                self.stdLine[self.thread.std[i]].setText(f'{composition[i]:.3f}-{std:.3f}')
            self.thread.plot_flag = False
        self.timer_busy = False

    def tau_change_m(self):
        # change tau by Gui input
        target = self.sender()
        self.tauDBox.blockSignals(True)
        self.tauFBox.blockSignals(True)
        self.thread.auto_tau_flag = False
        self.tauAtButton.setText('M')
        if target.objectName() == 'tauDBox':
            newD = self.tauDBox.value()
            if newD == 1:
                self.tauDBox.setSingleStep(0.1)
            elif newD == 0:
                self.tauDBox.setSingleStep(1)
            elif 0 < newD < 1:
                self.tauDBox.setValue(int(newD * 10))
                self.tauDBox.setSingleStep(1)
                self.tauFBox.setValue(self.tauFBox.value() + 1)
            elif 1 < newD < 10:
                if round(newD % 1, 1) == 0 and self.tauDBox.singleStep() == 0.1:
                    self.tauDBox.setSingleStep(1)
            elif newD >= 10:
                self.tauDBox.setValue(newD / 10)
                if self.tauDBox.value() == 1:
                    self.tauDBox.setSingleStep(0.1)
                else:
                    self.tauDBox.setSingleStep(1 if self.tauDBox.value() % 1 == 0 else 0.1)
                self.tauFBox.setValue(self.tauFBox.value() - 1)
        self.thread.tau = round(self.tauDBox.value() * (10 ** (-self.tauFBox.value())),
                                self.tauFBox.value() + 1)
        print(self.thread.tau)
        self.tauDBox.blockSignals(False)
        self.tauFBox.blockSignals(False)

    def tau_change_a(self, tau):
        # change tau by metropolis pass rate
        self.tauDBox.blockSignals(True)
        self.tauFBox.blockSignals(True)
        tau_str = ('%.1e' % tau).split('e-') if not tau == 0 else ['0', '0']
        self.tauDBox.setValue(float(tau_str[0]))
        self.tauFBox.setValue(int(tau_str[1]))
        self.tauDBox.blockSignals(False)
        self.tauFBox.blockSignals(False)

    def walk_param_change(self):
        #target = self.sender()
        if self.walkRangeBox.value() < self.walkStepBox.value():
            self.walkStepBox.setValue(self.walkRangeBox.value())
        self.thread.mvRange = int(self.walkRangeBox.value() / self.walkStepBox.value())
        self.thread.mvReso = int(1 / self.walkStepBox.value())

    def r_display(self, rn, rb):
        self.R2Line.setText(f'{rn:.8e} \\ {rb:.8e}')

    def step_display(self, step, _time, tau):
        self.countLine.setText('%d' % step)
        self.timeLine.setText('%d' % _time)
        # if self.tauATAction.isChecked():
        #     self.tau_change_a(tau)

    def auto_tau_switch(self):
        if self.thread.auto_tau_flag:
            self.thread.auto_tau_flag = False
            self.tauAtButton.setText('M')
        else:
            self.thread.auto_tau_flag = True
            self.tauAtButton.setText('A')

    def step_size_change(self):
        signal = self.sender()
        if signal.hasFocus():
            self.thread.step_size = self.stepSizeLine.value()

    def blocks_change(self):
        signal = self.sender()
        if signal.hasFocus():
            if self.blocksLine.value() > 1:
                self.thread.nblock = self.blocksLine.value()
                self.stepSizeLine.setMaximum(self.thread.nblock)
                self.thread.step_size = max(int(self.thread.nblock / 10), 2)
                self.stepSizeLine.setValue(self.thread.step_size)
                if not self.thread.fname == '':
                    self.thread.init()

    def std_check_state(self):
        std_i = []
        for i in range(self.stdLabel.size):
            if self.stdLabel[i].isChecked():
                std_i.append(i)
        if len(std_i) == 0:
            self.sender().setChecked(True)
            for i in range(self.stdLabel.size):
                if self.stdLabel[i].isChecked():
                    std_i.append(i)
        return np.asarray(std_i)

    def std_check_event(self):
        self.thread.std = self.std_check_state()
        for i in np.arange(self.stdBarPlot.size):
            if not (self.thread.std == i).any():
                self.stdBarPlot[i].setOpts(x=[0], height=[0])
        if not self.thread.ref0.size == 0:
            self.thread.init()

    def stdBar_add(self):
        self.stdBarWidget = np.append(self.stdBarWidget, barPlotWidget(background=(255, 255, 255, 255)))
        self.stdBarPlot = np.append(self.stdBarPlot, bar([], [], c='k', name=self.thread.species[-1], width=0.05))
        self.stdBarWidget[-1].setTitle(self.thread.species[-1], color='#000000', size='20pt')
        self.stdBarWidget[-1].addItem(self.stdBarPlot[-1])
        self.stdBarWidget[-1].plotItem.autoBtn.clicked.connect(self.stdBarWidget[-1].auto_range)
        self.barLayout.addWidget(self.stdBarWidget[-1], 1)

    def stdBar_sub(self):
        self.barLayout.removeWidget(self.stdBarWidget[-1])
        self.stdBarWidget[-1].deleteLater()
        self.stdBarWidget = self.stdBarWidget[:-1]
        self.stdBarPlot = self.stdBarPlot[:-1]

    def ref_file_change(self):
        target = int(self.sender().objectName().split('stdAddr')[1])
        fname = QFileDialog.getOpenFileName(self, 'select XAFS dat file...', os.path.dirname(self.stdAddr[target]),
                                            f'*.dat')[0]
        if fname == '':
            return
        with open(fname, 'r') as f:
            if f.readline().find('9809') == -1:
                e, mu = read_dat(fname)
            else:
                e, mu = read_9809_xafs(fname)
        ut = xanes_analysis(e, mu, self.thread.eref, en=30, ep=50, return_fit=True)
        ut = np.interp(self.thread.energy, e, ut)
        if self.stdLabel[target].text() == 'Null':
            self.thread.ref0 = np.vstack((self.thread.ref0, ut))
            self.thread.ref = np.vstack((self.thread.ref, ut[self.thread.edge]))
            self.thread.species.append(os.path.splitext(os.path.basename(fname))[0])
            self.stdLabel[target].setEnabled(True)
            self.stdLabel[target].setChecked(True)
            self.thread.std = np.append(self.thread.std, target)
            self.stdBar_add()
        else:
            self.thread.ref0[target] = ut
            self.thread.ref[target] = ut[self.thread.edge]
            self.thread.species[target] = os.path.splitext(os.path.basename(fname))[0]
            self.stdBarWidget[target].setTitle(self.thread.species[target])
        self.thread.init(ratio_reset=True)
        self.stdAddr[target] = fname
        self.stdLabel[target].setText(self.thread.species[target])

    def ref_add(self):
        self.stdLabel = np.append(self.stdLabel, QCheckBox(parent=self.stdInfoWidget))
        self.stdLine = np.append(self.stdLine, QLineEdit(parent=self.stdInfoWidget))
        self.stdAddrBtn = np.append(self.stdAddrBtn, QPushButton(parent=self.stdInfoWidget, text='...'))
        self.stdAddr.append('')

        self.stdInfoLayout.addWidget(self.stdLabel[-1], self.stdLineCount + 1, 0, 1, 1)
        self.stdInfoLayout.addWidget(self.stdLine[-1], self.stdLineCount + 1, 1, 1, 1)
        self.stdInfoLayout.addWidget(self.stdAddrBtn[-1], self.stdLineCount + 1, 2, 1, 1)
        self.stdLabel[-1].clicked.connect(self.std_check_event)
        self.stdAddrBtn[-1].clicked.connect(self.ref_file_change)

    def ref_sub(self):
        self.stdLabel[-1].clicked.disconnect(self.std_check_event)
        self.stdAddrBtn[-1].clicked.disconnect(self.ref_file_change)
        self.stdInfoLayout.removeWidget(self.stdLabel[-1])
        self.stdLabel[-1].deleteLater()
        self.stdInfoLayout.removeWidget(self.stdLine[-1])
        self.stdLine[-1].deleteLater()
        self.stdInfoLayout.removeWidget(self.stdAddrBtn[-1])
        self.stdAddrBtn[-1].deleteLater()

        self.stdLabel = self.stdLabel[:-1]
        self.stdLine = self.stdLine[:-1]
        self.stdAddrBtn = self.stdAddrBtn[:-1]
        self.stdAddr = self.stdAddr[:-1]

    def ref_num_change(self):
        self.stdInfoLayout.removeWidget(self.stdPlusBtn)
        self.stdInfoLayout.removeItem(self.stdInfoSpacerItem)
        self.stdPlusBtn.setParent(None)
        self.stdPlusBtn.hide()

        while len(self.thread.species) < self.stdLineCount:
            self.ref_sub()
            self.stdBar_sub()
            self.stdLineCount -= 1
        while len(self.thread.species) > self.stdLineCount:
            self.ref_add()
            self.stdBar_add()
            self.stdLineCount += 1

        for i in range(len(self.thread.species)):
            self.stdLabel[i].setChecked(True)
            self.stdLabel[i].setObjectName(f'stdLabel{i}')
            self.stdLabel[i].setText(self.thread.species[i])

            self.stdLine[i].setReadOnly(True)
            self.stdLine[i].setObjectName(f'stdLine{i}')

            self.stdAddrBtn[i].setFixedWidth(30)
            self.stdAddrBtn[i].setObjectName(f'stdAddr{i}')

            self.stdBarPlot[i].opts['name'] = self.thread.species[i]
            self.stdBarWidget[i].setTitle(self.thread.species[i])

        self.stdInfoLayout.addWidget(self.stdPlusBtn, self.stdLineCount + 1, 0, 1, 3)
        self.stdPlusBtn.show()
        self.stdInfoLayout.addItem(self.stdInfoSpacerItem, self.stdLineCount + 2, 0, 1, 3)
        for i in range(self.stdLineCount + 2):
            self.stdInfoLayout.setRowStretch(i, 1)
        self.stdInfoLayout.setSpacing(20)

    def ref_line_add(self):
        if self.stdLineCount >= 5 or self.stdLabel[-1].text() == 'Null':
            return
        self.stdInfoLayout.removeWidget(self.stdPlusBtn)
        self.stdInfoLayout.removeItem(self.stdInfoSpacerItem)
        self.stdPlusBtn.setParent(None)
        self.stdPlusBtn.hide()

        self.stdLabel = np.append(self.stdLabel, QCheckBox(parent=self.stdInfoWidget))
        self.stdLine = np.append(self.stdLine, QLineEdit(parent=self.stdInfoWidget))
        self.stdAddrBtn = np.append(self.stdAddrBtn, QPushButton(parent=self.stdInfoWidget, text='...'))
        self.stdAddr.append('')

        self.stdLabel[self.stdLineCount].setChecked(False)
        self.stdLabel[self.stdLineCount].setEnabled(False)
        self.stdLabel[self.stdLineCount].setObjectName(f'stdLabel{self.stdLineCount}')
        self.stdLabel[self.stdLineCount].setText('Null')
        self.stdInfoLayout.addWidget(self.stdLabel[self.stdLineCount], self.stdLineCount + 1, 0, 1, 1)

        self.stdLine[self.stdLineCount].setReadOnly(True)
        self.stdLine[self.stdLineCount].setObjectName(f'stdLine{self.stdLineCount}')
        self.stdInfoLayout.addWidget(self.stdLine[self.stdLineCount], self.stdLineCount + 1, 1, 1, 1)

        self.stdAddrBtn[self.stdLineCount].setFixedWidth(30)
        self.stdAddrBtn[self.stdLineCount].setObjectName(f'stdAddr{self.stdLineCount}')
        self.stdInfoLayout.addWidget(self.stdAddrBtn[self.stdLineCount], self.stdLineCount + 1, 2, 1, 1)

        self.stdAddr[-1] = os.getcwd() +f'\\ref\\{self.thread.element}\\{self.thread.species[-1]}.dat'

        self.stdLabel[-1].clicked.connect(self.std_check_event)
        self.stdAddrBtn[-1].clicked.connect(self.ref_file_change)
        self.stdLineCount += 1

        self.stdInfoLayout.addWidget(self.stdPlusBtn, self.stdLineCount + 1, 0, 1, 3)
        self.stdPlusBtn.show()
        self.stdInfoLayout.addItem(self.stdInfoSpacerItem, self.stdLineCount + 2, 0, 1, 3)

        for i in range(self.stdLineCount + 2):
            self.stdInfoLayout.setRowStretch(i, 1)
        self.stdInfoLayout.setSpacing(20)

    def energy_select_event(self):
        signal = self.sender()
        if signal.hasFocus():
            if signal == self.eminLine:
                emin = self.eminLine.value()
                if emin >= self.thread.emax:
                    self.eminLine.setValue(self.thread.emin)
                    return
                ei = np.where((self.thread.energy > emin) & (self.thread.energy < self.thread.emax))[0]
                if not ei.size == 0:
                    self.thread.emin = emin
                else:
                    self.eminLine.setValue(self.thread.emin)
                    return
            elif signal == self.emaxLine:
                emax = float(self.emaxLine.value())
                if emax <= self.thread.emin:
                    self.emaxLine.setValue(self.thread.emax)
                    return
                ei = np.where((self.thread.energy > self.thread.emin) & (self.thread.energy < emax))[0]
                if not ei.size == 0:
                    self.thread.emax = emax
                else:
                    self.emaxLine.setValue(self.thread.emax)
                    return
            else:
                return
            # self.plot.setXRange(self.thread.energy[ei[0]], self.thread.energy[ei[-1]], padding=0.1)
            if self.thread.flag:
                self.thread.edge_temp = ei
                self.thread.edge_change_flag = True
            else:
                self.thread.edge = ei
                self.thread.init(ratio_reset=False)

    def warning_window(self, massage):
        QMessageBox.critical(self, 'Error', massage)

    def show_massage(self, massage, ms):
        self.statusbar.showMessage(massage, ms)

    def choice_window(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Result loading')
        msg.setText('Result files exist, do you want to continue from the result?')
        # msg.setIcon(QMessageBox.question)
        msg.addButton('Continue', QMessageBox.ButtonRole.YesRole)
        msg.addButton('New', QMessageBox.ButtonRole.NoRole)
        msg.exec()
        return msg.clickedButton().text() == 'Continue'

    def reload_event(self):
        # target = self.sender()
        fname = QFileDialog.getOpenFileName(self, 'select ratio dat file...', self.thread.folder,
                                            f'{self.thread.fname}_ratio*.dat')[0]
        if fname == '':
            return
        self.statusbar.showMessage('Reading', 0)
        fname = fname.replace('\\', '/')
        self.thread.suffix = fname.split('.')[0].split('_ratio')[1]
        self.thread.folder = os.path.dirname(fname)
        self.thread.load()
        self.reload()
        self.thread.init(ratio_reset=False)
        self.plot.setXRange(self.thread.energy[self.thread.edge[0]],
                            self.thread.energy[self.thread.edge[-1]], padding=0.1)
        self.statusbar.showMessage('Done!', 3000)

    def contourMap(self):
        # rn_u, rn_i, rn_c = np.unique(np.round(self.thread.ratio * 10, 0) / 10, axis=0, return_inverse=True,
        #                              return_counts=True)
        # selected = np.argsort(rn_c[rn_i])
        _ratio = self.thread.ratio
        _ratio_clip = np.where(_ratio <= 1e-2, 0, _ratio)
        nonzero = ~((_ratio_clip == 0).all(0))
        _ratio_clip = _ratio_clip[:, nonzero][:, :4]
        # densities = rn_c[rn_i[selected]]
        dist = cdist(_ratio, _ratio)
        densities = np.sum(dist < 0.1, axis=1)
        fig = plt.figure(figsize=(10, 9))
        if _ratio_clip.shape[1] == 2:
            ax = fig.add_subplot()
            x = _ratio_clip[:, 0]
            y = _ratio_clip[:, 1]
            sc = ax.scatter(x, y, c=densities, s=100, cmap='plasma')
            ax.set_xlabel('%s' % self.thread.species[self.thread.std[nonzero][0]], fontsize=12)
            ax.set_ylabel('%s' % self.thread.species[self.thread.std[nonzero][1]], fontsize=12)
        elif _ratio_clip.shape[1] >= 3:
            # alpha = ((densities - densities.min())/(densities.max() - densities.min())).clip(0.3, 1)
            # s = ((densities - densities.min()) / (densities.max() - densities.min()))*100+50
            ax = fig.add_axes([0.05, 0.15, 0.75, 0.75], projection='3d', computed_zorder=False)
            x = _ratio_clip[:, 0]
            y = _ratio_clip[:, 1]
            z = _ratio_clip[:, 2]
            if _ratio_clip.shape[1] >= 4:
                cmap_edge = colormaps['inferno']
                norm_edge = Normalize(vmin=densities.min(), vmax=densities.max())
                edge_colors = cmap_edge(norm_edge(densities))
                w = _ratio_clip[:, 3]
                cmap_core = colormaps['viridis']
                norm_core = Normalize(vmin=w.min(), vmax=w.max())
                core_colors = cmap_core(norm_core(w))
                sc = ax.scatter(x, y, z, c=core_colors, edgecolors=edge_colors, s=100)
            else:
                sc = ax.scatter(x, y, z, c=densities, s=100, cmap='plasma', alpha=1)
            xmin, xmax = x.min() - (x.max() - x.min()) * 0.1, x.max() + (x.max() - x.min()) * 0.1
            ymin, ymax = y.min() - (y.max() - y.min()) * 0.1, y.max() + (y.max() - y.min()) * 0.1
            zmin, zmax = z.min() - (z.max() - z.min()) * 0.1, z.max() + (z.max() - z.min()) * 0.1
            ax.scatter(x, y, zs=zmin, zdir='z', c='k')
            ax.scatter(x, z, zs=ymax, zdir='y', c='k')
            ax.scatter(y, z, zs=xmax, zdir='x', c='k')
            ax.plot([x.min(), x.max()], [y.min(), y.min()], [zmin, zmin], c='k')
            ax.plot([x.min(), x.min()], [y.min(), y.max()], [zmin, zmin], c='k')
            ax.plot([x.min(), x.max()], [y.max(), y.max()], [zmin, zmin], c='k')
            ax.plot([x.max(), x.max()], [y.min(), y.max()], [zmin, zmin], c='k')
            ax.plot([x.min(), x.max()], [ymax, ymax], [z.min(), z.min()], c='k')
            ax.plot([x.min(), x.min()], [ymax, ymax], [z.min(), z.max()], c='k')
            ax.plot([x.min(), x.max()], [ymax, ymax], [z.max(), z.max()], c='k')
            ax.plot([x.max(), x.max()], [ymax, ymax], [z.min(), z.max()], c='k')
            ax.plot([xmax, xmax], [y.min(), y.max()], [z.min(), z.min()], c='k')
            ax.plot([xmax, xmax], [y.min(), y.min()], [z.min(), z.max()], c='k')
            ax.plot([xmax, xmax], [y.min(), y.max()], [z.max(), z.max()], c='k')
            ax.plot([xmax, xmax], [y.max(), y.max()], [z.min(), z.max()], c='k')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            ax.set_xlabel('%s' % self.thread.species[self.thread.std[nonzero][0]], fontsize=12)
            ax.set_ylabel('%s' % self.thread.species[self.thread.std[nonzero][1]], fontsize=12)
            ax.set_zlabel('%s' % self.thread.species[self.thread.std[nonzero][2]], fontsize=12)

        if _ratio_clip.shape[1] >= 4:
            cax1 = fig.add_axes([0.82, 0.25, 0.022, 0.5])
            cax2 = fig.add_axes([0.22, 0.08, 0.55, 0.022])
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm_core, cmap=cmap_core), cax=cax1, pad=0.1)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm_edge, cmap=cmap_edge), cax=cax2, pad=0.1,
                                 orientation="horizontal")
            cbar2.set_label('Density[edgecolor]', fontsize=12, labelpad=10)
        else:
            cbar = plt.colorbar(sc)
        cbar.set_label('Density' if _ratio_clip.shape[1] < 4 else f'{self.thread.species[self.thread.std[nonzero][3]]}',
                       fontsize=12, labelpad=10)
        plt.show()

    def analysis_event(self):
        ref = self.thread.ref0[self.thread.std]
        ratio_sum = self.thread.ratio.sum(axis=1, keepdims=True)
        ratio_sum = np.where(ratio_sum > 0, ratio_sum, 1)
        for i in range(self.thread.std.size):
            rdf = np.unique(np.round(self.thread.ratio[:, i], 1), return_counts=True)
            plt.subplot(2, 3, int(self.thread.std[i] + 1))
            plt.bar(rdf[0], rdf[1], width=0.05, color='k')
            plt.xlabel('ratio[%s]' % self.thread.species[self.thread.std[i]])
            plt.ylabel('count')

        plt.subplot(234)
        ratio_sum_stat = self.thread.ratio.sum(axis=1)
        aver = ratio_sum_stat[np.round(ratio_sum_stat, 1) > 0].mean()
        devi = ratio_sum_stat[np.round(ratio_sum_stat, 1) > 0].std()
        ratio_round = np.round(self.thread.ratio.sum(axis=1) * 20) / 20
        u, indices = np.unique(ratio_round, return_inverse=True)
        u, height = np.unique(ratio_round, return_counts=True)
        plt.bar(u, height, width=0.025, color='k')
        plt.xlabel('$α_1+α_2+α_3$')
        plt.ylabel('count')
        count = []
        for i in range(u.size):
            if not np.where(self.thread.ratio[indices == i].sum(axis=1, keepdims=True) == 0)[0].size > 0:
                ratio_norm = self.thread.ratio[indices == i] / self.thread.ratio[indices == i].sum(axis=1,
                                                                                                   keepdims=True)
                count.append(ratio_norm.mean(axis=0) / ratio_norm.mean(axis=0).sum())
            else:
                ratio_norm = self.thread.ratio[indices == i]
                count.append(ratio_norm.mean(axis=0))

        count = np.asarray(count).T
        x = np.arange(100) / 100 * (u.max() - u.min()) + u.min()
        print(aver, devi, height.max(), ratio_sum[np.round(ratio_sum, 1) > 0].size / ratio_sum.size)
        for i in np.arange(self.thread.std.size):
            plt.bar(u, height * count[i:].sum(axis=0), width=0.025, label=self.thread.species[self.thread.std[i]])
        plt.legend()

        ratio_norm = self.thread.ratio / ratio_sum
        # for j in range(std.size):
        #     plt.subplot(2, 3, 4 + std[j])
        #     com = (np.round(ratio_sum_stat[ratio_sum_stat > 0]*20) +
        #            np.round(ratio_norm[:, j][ratio_sum_stat > 0]*20) * 1j) / 20
        #     com_u, com_c = np.unique(com, return_counts=True)
        #
        #     finite = np.isfinite(com_u)
        #     com_u = com_u[finite]
        #     com_c = com_c[finite]
        #     x, xi = np.unique(com_u.real, return_inverse=True)
        #     y, yi = np.unique(com_u.imag, return_inverse=True)
        #     x = np.repeat(x, 2) if x.size == 1 else x
        #     y = np.repeat(y, 2) if y.size == 1 else y
        #     z = np.zeros((y.size, x.size))
        #     for i in range(com_c.size):
        #         z[yi[i]][xi[i]] += com_c[i]
        #     xx, yy = np.meshgrid(x, y)
        #     ax = plt.contourf(xx, yy, z, cmap=plt.get_cmap('rainbow', 256))
        #     plt.colorbar(ax)#, orientation='horizontal')
        #     plt.xlabel('sum of ratio')
        #     plt.ylabel('$ratio_{norm}$[%s]'%ref_name[std[j]])

        # plt.subplot(235)
        # count_ratio = np.array([(height * count.sum(axis=0))[np.where((0 < u) & (u < 0.5))[0]].sum(),
        #                         (height * count.sum(axis=0))[np.where((u > 0.5) & (u < 10))[0]].sum()])
        # tm_ratio = np.where(ratio_sum_stat == 0)[0].size
        # ratio = np.array([(height * count * u)[:, np.where((0 < u) & (u < 0.5))[0]].sum(axis=1),
        #                   (height * count * u)[:, np.where((u > 0.5) & (u < 10))[0]].sum(axis=1)
        #                   ])
        # ratio[count_ratio>0] = ratio[count_ratio>0] / count_ratio.reshape(*count_ratio.shape, 1)[count_ratio>0]
        # print(count_ratio, tm_ratio, ratio)
        # count_ratio = count_ratio / count_ratio.sum()
        # tm_ratio = tm_ratio / (ratio_sum_stat.size - tm_ratio)
        # fit_group = (np.exp((ratio.reshape(*ratio.shape, 1) * ref.reshape(1, *ref.shape)).sum(1)) * count_ratio.reshape(
        #     *count_ratio.shape, 1))
        # fit_group = fit_group.clip(0, None)
        # fit = np.log(1 / fit_group.sum(0))
        # poly_pre = np.polyfit(self.thread.energy[self.thread.energy < (self.thread.eref - 20)], fit[self.thread.energy < (self.thread.eref - 20)], 1)
        # #fit = fit - self.thread.energy * poly_pre[0] - poly_pre[1]
        # #ei = np.where((self.thread.energy > self.thread.emin) & (self.thread.energy < self.thread.emax))[0]
        # plt.plot(self.thread.energy, self.thread.exper0, c='k', label='Experimental')
        # # plt.plot(energy, fit, c='g', label='construct')
        # for i in range(count_ratio.size):
        #     if count_ratio[i] > 0:
        #         poly_pre = np.polyfit(self.thread.energy[self.thread.energy < (self.thread.eref - 20)], np.log(1 / fit_group[i])[self.thread.energy < (self.thread.eref - 20)], 1)
        #         plt.plot(self.thread.energy, (np.log(1 / fit_group[i]) - self.thread.energy * poly_pre[0] - poly_pre[1]), label=f'group{i}')
        #     else:
        #         plt.plot(self.thread.energy, np.zeros_like(self.thread.energy), label=f'group{i}')
        # recover = np.log(1 / (self.thread.exper0 * (1 + tm_ratio) - tm_ratio))
        # plt.plot(self.thread.energy, recover, c='r', label='Recover')
        # np.savetxt(self.thread.file.split('.dat')[0] + r'_recover.dat', np.vstack((self.thread.energy, recover)).T, fmt='%.3f', header='')
        # # plt.ylim(recover.min(), recover.max())
        # plt.xlabel('Energy(eV)')
        # plt.ylabel(r'μt')
        # plt.legend()

        # contourf ratioA:ratioB:count
        ratio_contour_dim = 3
        ratio_contour_index = np.array([0, 2])
        if not self.thread.std.size == 1:
            ax = plt.subplot(236) if (ratio_contour_dim == 2 or self.thread.std.size < 3) else plt.subplot(236,
                                                                                                           projection='3d')
            if self.thread.std.size < 3:
                ratio_contour_index = np.arange(2)
            rn_u, rn_c = np.unique(np.round(ratio_norm * 40) / 40, axis=0, return_counts=True)
            finite = np.isfinite(rn_u).all(axis=1)
            rn_u = rn_u[finite]
            rn_c = np.asarray(rn_c)[finite]
            rn_sort = np.argsort(rn_c)
            if ratio_contour_dim == 3 and self.thread.std.size == 3:
                ax0 = ax.scatter(rn_u[rn_sort, 0], rn_u[rn_sort, 1], rn_u[rn_sort, 2], c=rn_c[rn_sort], s=150,
                                 alpha=0.8, cmap='inferno')
                ax.set_xlabel('%s' % self.thread.species[0])
                ax.set_ylabel('%s' % self.thread.species[1])
                ax.set_zlabel('%s' % self.thread.species[2])
                plt.colorbar(ax0)
            else:
                x, xi = np.unique(rn_u[:, ratio_contour_index[0]], return_inverse=True)
                y, yi = np.unique(rn_u[:, ratio_contour_index[1]], return_inverse=True)
                x = np.repeat(x, 2) if x.size == 1 else x
                y = np.repeat(y, 2) if y.size == 1 else y
                z = np.zeros((y.size, x.size))
                for i in range(rn_c.size):
                    z[yi[i]][xi[i]] += rn_c[i]
                xx, yy = np.meshgrid(x, y)
                ax = plt.contourf(xx, yy, z, cmap=plt.get_cmap('rainbow', 256))
                plt.colorbar(ax)  # , orientation='horizontal')
                plt.xlabel('%s$_{norm}$' % self.thread.species[self.thread.std[ratio_contour_index[0]]])
                plt.ylabel('%s$_{norm}$' % self.thread.species[self.thread.std[ratio_contour_index[1]]])

        plt.show()

if __name__ == '__main__':
    app = QApplication(argv)
    # ratiop = np.array([0.4, 0.4, 0.2])
    # ratioa = np.array([[0.600, 0.000, 1.100, 0.000],
    #                   [1.200, 0.000, 0.700, 1.800],
    #                   [0.000, 0.000, 0.000, 0.000]])
    # ratiop = np.array([0.7, 0.3])
    # ratioa = np.array([[1.8, 0.9, 0.0],
    #                    [0.0, 0.3, 0.5]])
    if not os.path.exists(os.getcwd() + '\\ref'):
        QMessageBox.information(None, 'Error', '/ref folder not found.', QMessageBox.StandardButton.Ok)
        exit(0)
    else:
        if len(os.listdir(os.getcwd() + '\\ref')) == 0:
            QMessageBox.information(None, 'Error', 'No element found folder in /ref.', QMessageBox.StandardButton.Ok)
            exit(0)
    if not os.path.exists(os.getcwd() + '\\data'):
        os.mkdir(os.getcwd() + '\\data')
    dialog = Intro(file=os.getcwd() + '\\data\\block.dat')
    if dialog.exec() == QDialog.DialogCode.Accepted:
        if dialog.element == '':
            main = MainWindow(file=dialog.selected_file)
        else:
            print(dialog.ratiop @ dialog.ratioa, dialog.snr)
            main = MainWindow(file=os.getcwd() + '\\data\\model\\new', element=dialog.element,
                              ratio_plane=dialog.ratiop, ratio_axis=dialog.ratioa, snr=dialog.snr)
    exit(app.exec())