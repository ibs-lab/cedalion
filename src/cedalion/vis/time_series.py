# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:52:04 2024

@author: ahns97
"""

import cedalion
import cedalion.dataclasses as cdc
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
        self._auxTimeSeries_ax = self._dataTimeSeries_ax.twinx()
        self.plots.figure.tight_layout()
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
        
        
        # Create Wavelength Controls Layout
        wv_layout = QtWidgets.QGridLayout()
        wv_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(wv_layout,)
        
        ## Create Wavelength / Concentration Controls
        self.wv = QtWidgets.QListWidget()
        self.wv.setFixedHeight(45)
        self.wv.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.wv.itemSelectionChanged.connect(self.wv_changed)
        self.wv_label = QtWidgets.QLabel("Wavelength/Concentration:")
        wv_layout.addWidget(self.wv_label, 0,0)
        wv_layout.addWidget(self.wv, 0,1)
        
        
        # Create Optional Controls Layout
        opt_layout = QtWidgets.QVBoxLayout()
        opt_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(opt_layout,)
        
        ## Create Opt2Circ Button
        self.opt2circ = QtWidgets.QCheckBox("View optodes as circles")
        self.opt2circ.stateChanged.connect(self._toggle_circles)
        opt_layout.addWidget(self.opt2circ)
        
        ## Create Stim Plot button
        self.stim_togg = QtWidgets.QCheckBox("Plot stims")
        self.stim_togg.stateChanged.connect(self._toggle_stims)
        self.stim_togg.setCheckable(False)
        opt_layout.addWidget(self.stim_togg)
        
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
        self.stim_togg.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.init_calc()
        
    def init_calc(self):
        # Extract necessary data
        self.optodes_drawn = False
        
        self.snirfData = self.snirfRec.timeseries[list(self.snirfRec.timeseries.keys())[0]]
        
        self.sPos = self.snirfRec.geo2d.sel(label = ["S" in str(s.values) for s in self.snirfRec.geo2d.label])
        self.dPos = self.snirfRec.geo2d.sel(label = ["D" in str(s.values) for s in self.snirfRec.geo2d.label])
        self.sPosVal = self.sPos.values
        self.dPosVal = self.dPos.values
        
        self.slabel = self.sPos.label.values
        self.dlabel = self.dPos.label.values
        # self.clabel = self.snirfData.channel.values
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
        self.chan_highlight = []
        self.aux_sel = []
        self.auxplot = []
        self.aux_type = None
        self.aux_rect_width = 0
        self.plot_stims = 0
        
        # Create timeseries picker
        for i_ts, timser in enumerate(self.snirfRec.timeseries.keys()):
            self.ts.insertItem(i_ts+1,str(timser))
        
        # Create aux channels
        for i_a, aux_type in enumerate(self.snirfRec.aux_ts.keys()):
            self.auxs.insertItem(i_a+1, aux_type)
            
        if len(self.snirfRec.stim):
            self.stim_togg.setCheckable(True)
        else:
            self.stim_togg.setCheckable(False)
        
        self.snirfData = None
        self.draw_optodes()
        
        
    def draw_optodes(self):
        if self.optodes_drawn:
            return
        
        self._optode_ax.clear()
        
        self.picker = self._optode_ax.scatter(self.sdx, 
                                              self.sdy, 
                                              color=[[0,0,0,0]]*(len(self.sx)+len(self.dx)), 
                                              zorder=3,
                                              picker=3
                                              )
        
        self.optodes = self._optode_ax.scatter(self.sdx,
                                           self.sdy,
                                           color=['r']*len(self.sx)+['b']*len(self.dx),
                                           zorder=2,
                                           visible=False
                                           )
        
        for idx, source in enumerate(self.sPos.label):
            self.src_label[idx] = self._optode_ax.text(self.sx[idx],
                                                       self.sy[idx],
                                                       f"{source.values}", 
                                                       color="r", 
                                                       fontsize=8, 
                                                       ha='center', 
                                                       va='center', 
                                                       zorder=1, 
                                                       clip_on=True
                                                       )
            
        for idx, detector in enumerate(self.dPos.label):
            self.det_label[idx] = self._optode_ax.text(self.dx[idx],
                                                       self.dy[idx],
                                                       f"{detector.values}", 
                                                       color="b", 
                                                       fontsize=8, 
                                                       ha='center', 
                                                       va='center', 
                                                       zorder=1, 
                                                       clip_on=True
                                                       )
        
        for i_ch in range(self.no_channels):
            si = self.src_idx[i_ch]
            di = self.det_idx[i_ch]
            
            self._optode_ax.plot([self.sx[si],self.dx[di]],
                             [self.sy[si],self.dy[di]],
                             '-',
                             color=[0.8,0.8,0.8],
                             zorder=0,
                             )
            
        self.optodes_drawn = True
            
        self._optode_ax.set_aspect('equal')
        self._optode_ax.axis('off')
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

    def _toggle_stims(self, s):
        self.plot_stims = s
        self.draw_timeseries()
        
    
    def wv_changed(self):
        self.draw_timeseries()
        

    def ts_changed(self, s):
        # Extract data
        if s == "None":
            self.snirfData = None
            self._dataTimeSeries_ax.clear()
            return

        self.snirfData = self.snirfRec.timeseries[s]
        self.ts_sel = s
        
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
        
        ## Grab coordinates
        nempty_chan_sel = []
        x_chan_sel = [[],[]]
        y_chan_sel = [[],[]]
        
        for chan in chan_sel:
            if not np.isnan(self.snirfData.sel(channel=chan)[0][0]):
                x_chan_sel[0].append(self.sx[self.slabel == self.snirfData.sel(channel=chan).source.values][0])
                x_chan_sel[1].append(self.dx[self.dlabel == self.snirfData.sel(channel=chan).detector.values][0])
                y_chan_sel[0].append(self.sy[self.slabel == self.snirfData.sel(channel=chan).source.values][0])
                y_chan_sel[1].append(self.dy[self.dlabel == self.snirfData.sel(channel=chan).detector.values][0])
                nempty_chan_sel.append(chan)
        
        wvl_idx = self.wv.selectedItems()
        wvl_idx = [foo.text() for foo in wvl_idx]
        wvl_ls = ['-', ':']
        
        ## Grab timeseries y-label
        ylabel = self.ts_sel
        if "amp" in ylabel:
            ylabel = "amp (A.U.)"
        elif "od" in ylabel:
            ylabel = r"$\Delta$ OD (A.U.)"
        elif "conc" in ylabel or "chromo" in self.snirfData.dims:
            ylabel = r"$\Delta$ Concentration ($\mu$M)"

        # Highlight channels
        chan_col = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]
        
        for line in self.chan_highlight:
            line.remove()
            del line
        
        self.chan_highlight = [0]*len(nempty_chan_sel)
        
        for i_ch in range(len(nempty_chan_sel)):
            self.chan_highlight[i_ch], = self._optode_ax.plot([x_chan_sel[0][i_ch],x_chan_sel[1][i_ch]],
                                                             [y_chan_sel[0][i_ch],y_chan_sel[1][i_ch]], 
                                                             color=chan_col[i_ch%len(chan_col)])
        self._optode_ax.figure.canvas.draw()
        
        # Plot lines of aux
        if len(self.aux_sel):
            self.auxplot = self._auxTimeSeries_ax.plot(self.aux_sel.time, self.aux_sel, alpha=0.3, zorder=2)
            
        self._auxTimeSeries_ax.set_ylabel(self.aux_type, rotation=270, ha="right")
        self._auxTimeSeries_ax.yaxis.set_label_position("right")
        self._auxTimeSeries_ax.figure.canvas.draw()
        
        ymin = 100
        ymax = -100
        
        # Plot timeseries
        if "wavelength" in self.snirfData.dims:
            for sel_wv in wvl_idx:
                idx = self.snirfData.wavelength.values
                idx = [str(foo) for foo in idx]
                idx = idx.index(sel_wv)
                for i_ch, chan in enumerate(nempty_chan_sel):
                    self.timeSeries = self._dataTimeSeries_ax.plot(
                                                                    self.t,
                                                                    self.snirfData.sel(channel=chan,wavelength=sel_wv).T,
                                                                    ls=wvl_ls[idx],
                                                                    zorder=5,
                                                                    color=chan_col[i_ch%len(chan_col)],
                                                                  )
                    
                    ymin = min(ymin, min(self.snirfData.sel(channel=chan,wavelength=sel_wv).values.ravel()))
                    ymax = max(ymax, max(self.snirfData.sel(channel=chan,wavelength=sel_wv).values.ravel()))
                
        elif "chromo" in self.snirfData.dims:
            for sel_wv in wvl_idx:
                idx = self.snirfData.chromo.values
                idx = [str(foo) for foo in idx]
                idx = idx.index(sel_wv[1:-1])
                
                if "[" in sel_wv:
                    sel_wv = sel_wv[1:-1]
                
                for i_ch, chan in enumerate(nempty_chan_sel):
                    self.timeSeries = self._dataTimeSeries_ax.plot(
                                                                    self.t,
                                                                    self.snirfData.sel(channel=chan,chromo=sel_wv).T,
                                                                    ls=wvl_ls[idx],
                                                                    zorder=5,
                                                                    color=chan_col[i_ch%len(chan_col)],
                                                                  )
                
                    ymin = min(ymin, min(self.snirfData.sel(channel=chan,chromo=sel_wv).values.ravel()))
                    ymax = max(ymax, max(self.snirfData.sel(channel=chan,chromo=sel_wv).values.ravel()))
        
        # Plot stims
        stim_col = ["#648FFF","#DC267F","#FFB000","#785EF0","#FE6100"]
        if self.plot_stims:
            ymax = ymax + (.05*(ymax-ymin))
            ymin = ymin - (.05*(ymax-ymin))
            
            for i_t, tt in enumerate(np.unique(self.snirfRec.stim.trial_type)):
                label_on = True
                for i_r, dat in self.snirfRec.stim.loc[self.snirfRec.stim['trial_type'] == tt].iterrows():
                    if label_on:
                        self._dataTimeSeries_ax.axvline(dat.onset,
                                                        ls="--", 
                                                        lw=1, 
                                                        zorder=1,
                                                        c=stim_col[i_t%5],
                                                        label=tt
                                                        )
                        self._dataTimeSeries_ax.fill(
                                                     [dat.onset, dat.onset, dat.onset+dat.duration, dat.onset+dat.duration],
                                                     [ymin, ymax, ymax, ymin],
                                                     color=stim_col[i_t%5]+"22",
                                                     zorder=1,
                                                     )
                    else:
                        self._dataTimeSeries_ax.axvline(dat.onset,
                                                        ls="--", 
                                                        lw=1, 
                                                        zorder=1,
                                                        c=stim_col[i_t%5]
                                                        )
                        # self._dataTimeSeries_ax.axvline(dat.onset+dat.duration, 
                        #                                 ls="--", 
                        #                                 lw=1, 
                        #                                 zorder=1, 
                        #                                 c="gray"
                        #                                 )
                        self._dataTimeSeries_ax.fill(
                                                     [dat.onset, dat.onset, dat.onset+dat.duration, dat.onset+dat.duration],
                                                     [ymin, ymax, ymax, ymin],
                                                     color=stim_col[i_t%5]+"22",
                                                     zorder=1,
                                                     )
                    
                    label_on=False
            
            self._dataTimeSeries_ax.legend(loc="best")
        
        self._dataTimeSeries_ax.set_ylabel(ylabel)
        self._dataTimeSeries_ax.grid("True",axis="y")
        self._dataTimeSeries_ax.figure.canvas.draw()
        
        self.statbar.showMessage("Timeseries Drawn!")
    
    def aux_changed(self,s): # TODO
        self._auxTimeSeries_ax.clear()
        
        if s == 'None' or s == 'dark signal':
            self.aux_sel = []
            self.aux_type = None
            for line in self.auxplot:
                line.remove()
                del line
        # elif s == 'dark signal':
        #     return
        else:
            self.aux_sel = self.snirfRec.aux_ts[s]
            self.aux_type = s
            
        self.draw_timeseries()


def run_vis(snirfRec = None):
    """
    Parameters
    ----------
    snirfRec : Recording, optional
        Pass the cedalion Recording. The default is None.

    Opens a gui that loads the recording, if given.

    """
    if type(snirfRec) is not cdc.recording.Recording:
        app = QtWidgets.QApplication(sys.argv)
        main_gui = Main()
        main_gui.show()
        sys.exit(app.exec())
    else:
        app = QtWidgets.QApplication(sys.argv)
        main_gui = Main(snirfRec = snirfRec)
        main_gui.show()
        sys.exit(app.exec())
















