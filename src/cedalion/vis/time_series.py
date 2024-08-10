# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:52:04 2024

@author: ahns97
"""

import cedalion
import sys
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
import warnings
import time
warnings.simplefilter("ignore")


class Main(QtWidgets.QMainWindow):
    def __init__(self, snirfRec = None):
        # Initialize
        super().__init__()
        self.snirfRec = snirfRec
        
        # Set central widget
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        
        # Initialize layout
        window_layout = QtWidgets.QVBoxLayout(self._main)
        window_layout.setContentsMargins(10,0,10,10)
        window_layout.setSpacing(10)
        
        # Set Minimum Size
        self.setMinimumSize(1000,850)
        
        # Set Window Title
        self.setWindowTitle("Time Series")
        
        # Create Status Bar
        self.statbar = self.statusBar()
        self.statbar.showMessage("Ready to Load SNIRF File!")
        
        
        # Filler plot for now
        self.plots = FigureCanvas(Figure(figsize=(30,8)))
        self.plots.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.plots.setFocus()
        
        (self._dataTimeSeries_ax, self._optode_ax) = self.plots.figure.subplots(1, 2, width_ratios=[2,1])
        self._optode_ax.axis('off')
        self._dataTimeSeries_ax.grid("True",axis="y")
        window_layout.addWidget(NavigationToolbar(self.plots,self),stretch=1)
        window_layout.addWidget(self.plots, stretch=8)
        
        # Connect Plots
        self.shift_pressed = False
        self.plots.mpl_connect('key_press_event', self.shift_is_pressed)
        self.plots.mpl_connect('key_release_event', self.shift_is_released)
        self.plots.mpl_connect('pick_event', self.optode_picked)
        
        
        # Create Control Panel
        control_panel = QtWidgets.QGroupBox("Control Panel")
        control_panel_layout = QtWidgets.QHBoxLayout()
        control_panel_layout.setSpacing(20)
        control_panel.setLayout(control_panel_layout)
        window_layout.addWidget(control_panel, stretch=1)
        
        
        # Create Timeseries Controls Layout
        ts_layout = QtWidgets.QGridLayout()
        ts_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(ts_layout,)
        
        ## Create Timeseries Controls
        self.ts = QtWidgets.QListWidget()
        self.ts.addItems(["None"])
        self.ts.setCurrentRow(0)
        self.ts.setFixedHeight(60)
        self.ts.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ts.currentTextChanged.connect(self.ts_changed)
        ts_layout.addWidget(QtWidgets.QLabel("Timeseries:"), 0,0)
        ts_layout.addWidget(self.ts, 0,1)
        
        
        # Create Aux selector Layout
        aux_layout = QtWidgets.QGridLayout()
        aux_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(aux_layout)
        
        ## Aux Selector
        self.auxs = QtWidgets.QComboBox()
        self.auxs.addItems(["None"])
        self.auxs.setCurrentIndex(0)
        self.auxs.currentTextChanged.connect(self.aux_changed)
        aux_layout.addWidget(QtWidgets.QLabel("Aux:"), 0,0)
        aux_layout.addWidget(self.auxs, 0,1)
        
        ## Aux Window Creator 
        self.aux_window = QtWidgets.QLineEdit()
        self.aux_window.setText('0')
        validator = QtGui.QDoubleValidator()
        validator.setRange(0,100)
        validator.setDecimals(3)
        self.aux_window.setValidator(validator)
        self.aux_window.textChanged.connect(self.aux_rect)
        aux_layout.addWidget(QtWidgets.QLabel("Aux window:"), 1,0)
        aux_layout.addWidget(self.aux_window, 1,1)
        
        
        # Create Wavelength Controls Layout
        wv_layout = QtWidgets.QGridLayout()
        wv_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(wv_layout,)
        
        ## Create Wavelength / Concentration Controls
        self.wv = QtWidgets.QListWidget()
        self.wv.setFixedHeight(45)
        self.wv.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.wv.currentTextChanged.connect(self.wv_changed)
        self.wv_label = QtWidgets.QLabel("Wavelength/Concentration:")
        wv_layout.addWidget(self.wv_label, 0,0)
        wv_layout.addWidget(self.wv, 0,1)
        
        
        # Create Opt2Circ Button
        self.opt2circ = QtWidgets.QCheckBox("View optodes as circles")
        self.opt2circ.stateChanged.connect(self._toggle_circles)
        control_panel_layout.addWidget(self.opt2circ,alignment=QtCore.Qt.AlignTop)
        
        ## Spacer
        control_panel_layout.addStretch()
        
        
        # Create button action for opening file
        open_btn = QtWidgets.QAction("Open...", self)
        open_btn.setStatusTip("Open SNIRF file")
        open_btn.triggered.connect(self.open_dialog)
        
        ## Create menu
        menu = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu)
       
        ## Populate menu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
         
        file_menu = menu.addMenu("&File")
        file_menu.addAction(open_btn)
        
        # In case there is snirfRec
        if self.snirfRec is not None:
            self.init_calc()
            
        
        
    def open_dialog(self):
        # Grab the appropriate SNIRF file
        self._fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "SNIRF Files (*.snirf)",
        )[0]
        self.statbar.showMessage("Loading SNIRF File...")
        t0 = time.time()
        cread = cedalion.io.read_snirf(self._fname)
        self.snirfRec = cread[0]
        t1 = time.time()
        self.statbar.showMessage(f'File Loaded in {t1 - t0:.2f} seconds!')
        self.auxs.setCurrentIndex(0)
        self.ts.setCurrentRow(0)
        self.aux_window.setText('0')
        self.init_calc()
        
    def init_calc(self):
        # Extract necessary data
        self.snirfData = self.snirfRec.timeseries[list(self.snirfRec.timeseries.keys())[0]]
        
        self.sPos = self.snirfRec.geo2d.sel(label = ["S" in str(s.values) for s in self.snirfRec.geo2d.label])
        self.dPos = self.snirfRec.geo2d.sel(label = ["D" in str(s.values) for s in self.snirfRec.geo2d.label])
        self.sPosVal = self.sPos.values
        self.dPosVal = self.dPos.values
        
        self.slabel = self.sPos.label.values
        self.dlabel = self.dPos.label.values
        self.clabel = self.snirfData.channel.values
        self.opt_label = np.append(self.slabel, self.dlabel)
        
        # Extract lengths
        self.no_channels = len(self.snirfData.channel)
        self.no_wvls = len(self.snirfData.wavelength)
        
        # Index channels and optodes
        self.channel_idx = np.arange(0,self.no_channels)
        self.src_idx = [np.arange(0,len(self.sPos))[self.sPos.label == src][0] for src in self.snirfData.source]
        self.det_idx = [np.arange(0,len(self.dPos))[self.dPos.label == det][0] for det in self.snirfData.detector]
        
        # Extract the x-y coordinates of the optodes
        self.sx = self.sPosVal[:,0]
        self.sy = self.sPosVal[:,1]
        self.dx = self.dPosVal[:,0]
        self.dy = self.dPosVal[:,1]
        
        self.sdx = np.append(self.sx, self.dx)
        self.sdy = np.append(self.sy, self.dy)
        
        # Initialize holders to control each part of the plot
        self.src_label = [0]*len(self.sx)
        self.det_label = [0]*len(self.dx)
        self.selected = []
        self.aux_sel = []
        self.aux_rect_width = 0
        
        # Create timeseries picker
        for i_ts, timser in enumerate(self.snirfRec.timeseries.keys()):
            self.ts.insertItem(i_ts+1,str(timser))
        
        # Create aux channels
        for i_a, aux_type in enumerate(self.snirfRec.aux_ts.keys()):
            self.auxs.insertItem(i_a+1, aux_type)
        
        self.snirfData = None
        self.draw_optodes()
        
        
    def draw_optodes(self):
        self._optode_ax.clear()
        
        self.picker = self._optode_ax.scatter(self.sdx, self.sdy,color=[[0,0,0,0]]*(len(self.sx)+len(self.dx)), zorder=3,picker=3)
        
        self.optodes = self._optode_ax.scatter(self.sdx,
                                           self.sdy,
                                           color=['r']*len(self.sx)+['b']*len(self.dx),
                                           zorder=2,
                                           visible=False
                                           )
        
        for idx, source in enumerate(self.sPos.label):
            self.src_label[idx] = self._optode_ax.text(self.sx[idx],self.sy[idx],f"{source.values}", color="r", fontsize=8, ha='center', va='center', zorder=1, clip_on=True)
        for idx, detector in enumerate(self.dPos.label):
            self.det_label[idx] = self._optode_ax.text(self.dx[idx],self.dy[idx],f"{detector.values}", color="b", fontsize=8, ha='center', va='center', zorder=1, clip_on=True)
        
        for i_ch in range(self.no_channels):
            si = self.src_idx[i_ch]
            di = self.det_idx[i_ch]
            
            self._optode_ax.plot([self.sx[si],self.dx[di]],
                             [self.sy[si],self.dy[di]],
                             '-',
                             color=[0.8,0.8,0.8],
                             zorder=0,
                             )
            
        self._optode_ax.set_aspect('equal')
        self._optode_ax.axis('off')
        self._optode_ax.figure.tight_layout()
        self._optode_ax.figure.canvas.draw()


    def shift_is_pressed(self, event):
        if self.shift_pressed == False and event.key == "shift":
            self.shift_pressed = True
        else:
            return
    
    
    def shift_is_released(self, event):
        if self.shift_pressed == True and event.key == "shift":
            self.shift_pressed = False
        else:
            return


    def optode_picked(self, event):      
        if self.ts.currentItem().text() == "None":
            return
        
        if not self.shift_pressed:
            self.selected = []
        
        if event.artist != self.picker:
            return
        
        N = len(event.ind)
        if not N:
            return
        
        # Click location
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        
        distances = np.hypot(x - self.sdx[event.ind], y - self.sdy[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]
        
        self.selected.append(dataind)
        self.draw_timeseries()
        
        
    def _toggle_circles(self):
        if self.opt2circ.isChecked():            
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_visible(False)
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_visible(False)
                
            self.optodes.set_visible(True)
        else:
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_visible(True)
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_visible(True)
                
            self.optodes.set_visible(False)
                
        self._optode_ax.figure.canvas.draw()

        
    
    def wv_changed(self, s):
        self.draw_timeseries()
        

    def ts_changed(self, s):
        # Extract data
        if s == "None":
            self.snirfData = None
            self._dataTimeSeries_ax.clear()
            return

        self.snirfData = self.snirfRec.timeseries[s]
        
        # Determine wavelength/concentration
        if "wavelength" in self.snirfData.dims:
            self.wv_label.setText("Wavelength:")
            self.wv.clear()
            
            for i_w, wvl in enumerate(self.snirfData.wavelength.values):
                self.wv.insertItem(i_w,str(wvl))
            self.wv.setCurrentRow(0)
            
        elif "chromo" in self.snirfData.dims:
            self.wv_label.setText("Concentration:")
            self.wv.clear()
            
            for i_w, wvl in enumerate(self.snirfData.chromo.values):
                self.wv.insertItem(i_w,f"[{str(wvl)}]")
            self.wv.setCurrentRow(0)

        self.draw_timeseries()


    def draw_timeseries(self):
        self._dataTimeSeries_ax.clear()
        
        if self.snirfData is None:
            return
        if len(self.selected) == 0:
            return
        
        # Extract time information
        self.t = self.snirfData.time.values
        
        opt_sel = self.opt_label[self.selected]
        chan_sel = []
        
        x_opt_sel = []
        y_opt_sel = []
        
        # Grab relevant data
        for opt in opt_sel:
            if 'S' in opt:
                chan_sel += self.snirfData.source[self.snirfData.source == opt].channel.values.tolist()
                x_opt_sel.append(self.sx[self.slabel == opt])
                y_opt_sel.append(self.sy[self.slabel == opt])
            elif 'D' in opt:
                chan_sel += self.snirfData.detector[self.snirfData.detector == opt].channel.values.tolist()
                x_opt_sel.append(self.dx[self.dlabel == opt])
                y_opt_sel.append(self.dy[self.dlabel == opt])
        
        chan_sel = np.unique(chan_sel)
        chan_sel_idx = [np.arange(0,self.no_channels)[self.clabel == chan][0] for chan in chan_sel]
        
        ## Grab coordinates
        x_chan_sel = [[],[]]
        y_chan_sel = [[],[]]
        
        for i_ch in chan_sel_idx:
            if not np.isnan(self.snirfData.isel(channel=i_ch)[0][0]):
                x_chan_sel[0].append(self.sx[self.slabel == self.snirfData.isel(channel=i_ch).source.values][0])
                x_chan_sel[1].append(self.dx[self.dlabel == self.snirfData.isel(channel=i_ch).detector.values][0])
                y_chan_sel[0].append(self.sy[self.slabel == self.snirfData.isel(channel=i_ch).source.values][0])
                y_chan_sel[1].append(self.dy[self.dlabel == self.snirfData.isel(channel=i_ch).detector.values][0])
            else:
                x_chan_sel[0].append(np.nan)
                x_chan_sel[1].append(np.nan)
                y_chan_sel[0].append(np.nan)
                y_chan_sel[1].append(np.nan)
        
        wvl_idx = self.wv.currentRow()
        wvl_ls = ['-', ':']
        

        # Highlight channels
        self.draw_optodes()
        self.chan_highlight = self._optode_ax.plot(x_chan_sel,y_chan_sel)
        self._optode_ax.figure.canvas.draw()
        
        # Plot timeseries
        if "wavelength" in self.snirfData.dims:
            self.timeSeries = self._dataTimeSeries_ax.plot(self.t, self.snirfData.isel(channel=chan_sel_idx,wavelength=wvl_idx).T,ls=wvl_ls[wvl_idx],zorder=5)
        elif "chromo" in self.snirfData.dims:
            self.timeSeries = self._dataTimeSeries_ax.plot(self.t, self.snirfData.isel(channel=chan_sel_idx)[wvl_idx].T,ls=wvl_ls[wvl_idx],zorder=5)
        
        # Plot lines or rectangles of aux
        if len(self.aux_sel) != 0:
            aux_on = np.append(0,self.aux_sel[self.aux_sel == 1].time.values)
            aux_ondiff = np.array([aux_on[i+1] - aux_on[i] > 1 for i in range(len(aux_on) - 1)])
            aux_ondiff = np.append([False],aux_ondiff)
            aux_marks = aux_on[aux_ondiff] + 1
            
            if self.aux_rect_width == 0:
                for rx in aux_marks:
                    self._dataTimeSeries_ax.axvline(rx,c="k",lw=1,zorder=0)
            else:
                if "wavelength" in self.snirfData.dims:
                    y_val = self.snirfData[chan_sel_idx,wvl_idx].values.ravel()
                    y_val = y_val[~np.isnan(y_val)]
                elif "chromo" in self.snirfData.dims:
                    y_val = self.snirfData.isel(channel=chan_sel_idx)[wvl_idx].values.ravel()
                    y_val = y_val[~np.isnan(y_val)]
                
                ry = max(y_val)
                ry2 = min(y_val)
                ry2 = min(ry2,0)
                
                for rx in aux_marks:
                    rx2 = rx + self.aux_rect_width
                    self._dataTimeSeries_ax.fill(
                                                 [rx, rx, rx2, rx2],
                                                 [ry2, ry, ry, ry2],
                                                 color=[0.7,0.7,0.7,0.4],
                                                 zorder=0,
                                                 )
        
        self._dataTimeSeries_ax.grid("True",axis="y")
        self._dataTimeSeries_ax.figure.canvas.draw()
        
        self.statbar.showMessage("Timeseries Drawn!")
    
    def aux_changed(self,s): # TODO
        if s == 'None':
            return
        
        if s == 'dark signal':
            return
        elif s == 'digital':
            self.aux_sel = self.snirfRec.aux_ts[s]
        else:
            return
            
        self.draw_timeseries()
        
    def aux_rect(self,s):
        self.aux_rect_width = float(s)
        
        self.draw_timeseries()
    
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     main_gui = Main()
#     main_gui.show()
#     sys.exit(app.exec())


def run_vis(snirfRec = None):
    """
    Parameters
    ----------
    snirfRec : Recording, optional
        Pass the cedalion Recording. The default is None.

    Opens a gui that loads the recording, if given.

    """
    if type(snirfRec) is not cedalion.dataclasses.recording.Recording:
        app = QtWidgets.QApplication(sys.argv)
        main_gui = Main()
        main_gui.show()
        sys.exit(app.exec())
    else:
        app = QtWidgets.QApplication(sys.argv)
        main_gui = Main(snirfRec = snirfRec)
        main_gui.show()
        sys.exit(app.exec())
















