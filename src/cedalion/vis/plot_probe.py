# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:34 2024

@author: ahns97
"""

import cedalion
import sys
import numpy as np
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import warnings
import time
warnings.simplefilter("ignore")


class Main(QtWidgets.QMainWindow):
    def __init__(self, snirfData = None, geo2d = None, geo3d = None):
        # Initialize
        super().__init__()
        self.snirfData = snirfData
        self.geo2d = geo2d
        self.geo3d = geo3d
        
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
        self.setWindowTitle("Plot Probe")
        
        
        # Filler plot for now
        self.plotprobe = FigureCanvas(Figure(figsize=(10,10)))
        self._ax = self.plotprobe.figure.subplots()
        self._ax.axis('off')
        window_layout.addWidget(NavigationToolbar(self.plotprobe,self),stretch=1)
        window_layout.addWidget(self.plotprobe, stretch=8)
        
        # Create Control Panel
        control_panel = QtWidgets.QGroupBox("Control Panel")
        control_panel_layout = QtWidgets.QHBoxLayout()
        control_panel.setLayout(control_panel_layout)
        window_layout.addWidget(control_panel, stretch=1)
        
        
        # Create Activity Display Controls
        display_activity = QtWidgets.QGroupBox("Display Activity")
        display_activity_layout = QtWidgets.QVBoxLayout()
        display_activity.setLayout(display_activity_layout)
        
        ## Condition Selector
        self.conditions = QtWidgets.QListWidget()
        self.conditions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.conditions.insertItem(0, "-- Select Condition --")
        self.conditions.currentTextChanged.connect(self._condition_changed)
        display_activity_layout.addWidget(self.conditions)
        
        ## Add Display Activity
        control_panel_layout.addWidget(display_activity, stretch=1)
        
        
        # Create Plot Scale Controls
        plot_scale = QtWidgets.QGroupBox("Plot Scale")
        plot_scale_layout = QtWidgets.QVBoxLayout()
        plot_scale_layout.setSpacing(15)
        plot_scale.setLayout(plot_scale_layout)
        
        ## X Scaler
        x_scale_layout = QtWidgets.QHBoxLayout()
        self.x_scale = QtWidgets.QDoubleSpinBox()
        self.x_scale.setValue(1)
        self.x_scale.setSingleStep(0.2)
        self.x_scale.valueChanged.connect(self._xscale_changed)
        x_scale_layout.addWidget(QtWidgets.QLabel("X scale"))
        x_scale_layout.addWidget(self.x_scale)
        plot_scale_layout.addLayout(x_scale_layout)
        
        ## Y Scaler
        y_scale_layout = QtWidgets.QHBoxLayout()
        self.y_scale = QtWidgets.QDoubleSpinBox()
        self.y_scale.setValue(1)
        self.y_scale.setSingleStep(0.2)
        self.y_scale.valueChanged.connect(self._yscale_changed)
        y_scale_layout.addWidget(QtWidgets.QLabel("Y scale"))
        y_scale_layout.addWidget(self.y_scale)
        plot_scale_layout.addLayout(y_scale_layout)
        
        # ## Axis Image
        # axim_btn = QtWidgets.QCheckBox("Axis Image")
        # plot_scale_layout.addWidget(axim_btn)
        
        ## Add Scaler
        control_panel_layout.addWidget(plot_scale, stretch=1)
        
        
        # Create Prune Channels Control
        prune_channels = QtWidgets.QGroupBox("Prune Channels")
        prune_channels_layout = QtWidgets.QGridLayout()
        prune_channels_layout.setSpacing(10)
        prune_channels.setLayout(prune_channels_layout)
        
        ## Set up Prune Channels controllers
        self.mindist = QtWidgets.QDoubleSpinBox()
        self.mindist.setValue(15)
        self.mindist.setSingleStep(3)
        self.mindist.valueChanged.connect(self._mindist_changed)
        self.maxdist = QtWidgets.QDoubleSpinBox()
        self.maxdist.setValue(45)
        self.mindist.setRange(15,self.maxdist.value())
        self.maxdist.setRange(self.mindist.value(),45)
        self.maxdist.setSingleStep(3)
        self.maxdist.valueChanged.connect(self._maxdist_changed)
        self.ssfade = QtWidgets.QDoubleSpinBox()
        self.ssfade.setValue(15)
        self.ssfade.setRange(15,45)
        self.ssfade.setSingleStep(3)
        self.ssfade.valueChanged.connect(self._ssfade_changed)
        
        ## Populate Prune Channels
        prune_channels_layout.addWidget(QtWidgets.QLabel("Min dist"), 0,0)
        prune_channels_layout.addWidget(self.mindist, 0,1)
        prune_channels_layout.addWidget(QtWidgets.QLabel("Max dist"), 1,0)
        prune_channels_layout.addWidget(self.maxdist, 1,1)
        prune_channels_layout.addWidget(QtWidgets.QLabel("SS fade thresh"), 2,0)
        prune_channels_layout.addWidget(self.ssfade, 2,1)
        
        ## Add Prune Channels
        control_panel_layout.addWidget(prune_channels, stretch=1)
        
        
        # Create Probe Control
        probe_control = QtWidgets.QGroupBox("Probe")
        probe_control_layout = QtWidgets.QVBoxLayout()
        probe_control_layout.setSpacing(5)
        probe_control.setLayout(probe_control_layout)
        
        ## Set up Probe Control controllers
        self.opt2circ = QtWidgets.QCheckBox("View optodes as circles")
        self.opt2circ.stateChanged.connect(self._toggle_circles)
        self.measline = QtWidgets.QCheckBox("Display Measurement Line")
        self.measline.stateChanged.connect(self._toggle_measline)
        
        # sigact = QtWidgets.QCheckBox("Display significant activation")
        # pval = QtWidgets.QLineEdit()
        # pval.setInputMask('0.00;_')
        # pval.setText("0.05")
        
        ## Populate Probe Control
        probe_control_layout.addWidget(self.opt2circ)
        probe_control_layout.addWidget(self.measline)
        # probe_control_layout.addWidget(sigact)
        pval_layout = QtWidgets.QHBoxLayout()
        pval_layout.setSpacing(10)
        # pval_layout.addWidget(QtWidgets.QLabel("p-val level of sig"))
        # pval_layout.addWidget(pval)
        probe_control_layout.addLayout(pval_layout)
        
        ## Add Probe Control
        control_panel_layout.addWidget(probe_control, stretch=1)
        
        
        # # Create Reference Points Control
        # ref_point = QtWidgets.QGroupBox("Reference Points")
        # ref_point_layout = QtWidgets.QVBoxLayout()
        # ref_point_layout.setSpacing(10)
        # ref_point.setLayout(ref_point_layout)
        
        # ## Set up and populate selectors
        # label_btn = QtWidgets.QRadioButton("Labels")
        # circ_btn = QtWidgets.QRadioButton("Circles")
        # ref_point_layout.addWidget(label_btn,stretch=1)
        # ref_point_layout.addWidget(circ_btn,stretch=1)
        # # ref_point_layout.addWidget(QtWidgets.QLabel(),stretch=2)
        
        # ## Add Reference Points Control
        # control_panel_layout.addWidget(ref_point,stretch=1)
        
        
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
        
        if self.snirfData is not None:
            
            if np.shape(self.snirfData)[1] != len(self.snirfData.channel):
                self.snirfData = self.snirfData.transpose("trial_type", "channel", "chromo", "reltime")
            
            self.sPos = self.geo2d.sel(label = ["S" in str(s.values) for s in self.geo2d.label])
            self.dPos = self.geo2d.sel(label = ["D" in str(s.values) for s in self.geo2d.label])
            
            self.sourcePos3D = self.geo3d.sel(label = ["S" in str(s.values) for s in self.geo3d.label])
            self.detectorPos3D = self.geo3d.sel(label = ["D" in str(s.values) for s in self.geo3d.label])
            
            print("starting calculations!")
            self.init_calc()
        
    
    def open_dialog(self):
        # Grab the appropriate SNIRF file
        self._fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "SNIRF Files (*.snirf)",
        )[0]
        print('Loading SNIRF...')
        t0 = time.time()
        self.snirfObj = cedalion.io.read_snirf(self._fname)
        t1 = time.time()
        print(f'SNIRF Loaded in {t1 - t0:.2f} seconds!')
        
        # Extract necessary data
        self.snirfData = self.snirfObj[0].data[0]

        self.sPos = self.snirfObj[0].geo2d.sel(label = ["S" in str(s.values) for s in self.snirfObj[0].geo2d.label])
        self.dPos = self.snirfObj[0].geo2d.sel(label = ["D" in str(s.values) for s in self.snirfObj[0].geo2d.label])

        self.sourcePos3D = self.snirfObj[0].geo3d.sel(label = ["S" in str(s.values) for s in self.snirfObj[0].geo3d.label])
        self.detectorPos3D = self.snirfObj[0].geo3d.sel(label = ["D" in str(s.values) for s in self.snirfObj[0].geo3d.label])
        
        self.init_calc()
    
    def init_calc(self):
        t0 = time.time()
        
        # Initialize certain values to begin
        self.x_scale.setValue(1)
        self.plot_Xscale = 1
        self.x_scale.setValue(1)
        self.plot_Yscale = 1
        
        self.mindist.setValue(15)
        self.channel_min_dist = 15
        self.maxdist.setValue(45)
        self.channel_max_dist = 45
        self.ssfade.setValue(15)
        self.ssFadeThres = 15
        self.fade_factor = 0.3 ##### Connect?
        self.lineWidth = 0.7 ##### Connect?
        
        self.conditions.clear()
        self.opt2circ.setChecked(False)
        self.measline.setChecked(False)
        
        # Color for conditions
        self.color_HbO = [0.862, 0.078, 0.235] ##### Connect?
        self.color_HbR = [0,     0,     0.8  ] ##### Connect?
        self.chrom = {0: self.color_HbO, 1: self.color_HbR}
        
        self.sPosVal = self.sPos.values
        self.dPosVal = self.dPos.values
        
        self.channel_idx = np.arange(0,len(self.snirfData.channel))
        self.src_idx = [np.arange(0,len(self.sPos))[self.sPos.label == src][0] for src in self.snirfData.source]
        self.det_idx = [np.arange(0,len(self.dPos))[self.dPos.label == det][0] for det in self.snirfData.detector]
        
        # Find Channel distances
        self.chan_dist = np.sqrt(np.sum((self.sourcePos3D.values[self.src_idx] - self.detectorPos3D.values[self.det_idx])**2,1))
        self.dist = np.sqrt(np.sum((self.sPosVal[self.src_idx] - self.dPosVal[self.det_idx])**2,1))
        
        # Find the extreme coordinates of the optodes
        self.sdMin = np.array([min([min(self.sPos[:,0]),min(self.dPos[:,0])]).values,min([min(self.sPos[:,1]),min(self.dPos[:,1])]).values])
        self.sdMin = self.sdMin - np.mean(self.chan_dist)
        self.sdMax = np.array([max([max(self.sPos[:,0]),max(self.dPos[:,0])]).values,max([max(self.sPos[:,1]),max(self.dPos[:,1])]).values])
        self.sdMax = self.sdMax + np.mean(self.chan_dist)
        
        # Find the scaling factors for the plot
        self.sdWid = self.sdMax[0] - self.sdMin[0]
        self.sdHgt = self.sdMax[1] - self.sdMin[1]
        
        # Find the axis scale
        self.sd2axScl = max(self.sdWid,self.sdHgt)
        
        # Scale the optode coordinates by the scale
        self.sPosVal /= self.sd2axScl
        self.dPosVal /= self.sd2axScl
        
        # Calculate the scaling/translation factors
        self.nAcross = len(np.unique(np.append(self.sPosVal,self.dPosVal,axis=0)[:,0])) + 1
        self.axXoff = np.mean(np.append(self.sPosVal,self.dPosVal,axis=0)[:,0]) - 0.5
        self.nUp = len(np.unique(np.append(self.sPosVal,self.dPosVal,axis=0)[:,1])) + 1
        self.axYoff = np.mean(np.append(self.sPosVal,self.dPosVal,axis=0)[:,1]) - 0.5
        
        # Calculate the size of each HRF
        self.axWid = self.plot_Xscale/self.nAcross
        self.axHgt = self.plot_Yscale/self.nUp
        
        # Extract the x-y coordinates of the optodes
        self.sx = self.sPosVal[:,0] - self.axXoff
        self.sy = self.sPosVal[:,1] - self.axYoff
        self.dx = self.dPosVal[:,0] - self.axXoff
        self.dy = self.dPosVal[:,1] - self.axYoff
        
        # Extract time information
        try:
            self.t = self.snirfData.time.values
        except:
            pass
        
        try:
            self.t = self.snirfData.reltime.values
        except:
            pass
        
        self.minT = min(self.t)
        self.maxT = max(self.t)
        
        # Extract lengths
        self.trial_types = len(self.snirfData.trial_type)
        self.channels = len(self.snirfData.channel)
        self.chromophores = len(self.snirfData.chromo)
        
        # Initialize holders to control each part of the plot
        self.src_label = [0]*len(self.sx)
        self.det_label = [0]*len(self.dx)
        self.meas_line = [0]*self.channels
        self.hrf = [0]*self.chromophores*self.channels*self.trial_types # access via: condition*no_chans + channel*no_chrom + chromophore
        
        # Calculate the HRF plot coordinates
        self.xa = (self.sx[self.src_idx] + self.dx[self.det_idx])/2
        self.ya = (self.sy[self.src_idx] + self.dy[self.det_idx])/2
        self.hrf_val = [self.snirfData.sel(trial_type=i).values for i in self.snirfData.trial_type]
        self.ya = np.array([[[a]*len(self.t)]*self.chromophores for a in self.ya])
        
        self.cmin = [0]*self.trial_types
        self.cmax = [0]*self.trial_types
        
        for trial in range(self.trial_types):
            self.cmin[trial] = np.min(np.nan_to_num(self.hrf_val[trial].ravel(),10))
            self.cmax[trial] = np.max(np.nan_to_num(self.hrf_val[trial].ravel(),-10))
        
        self.cmin = min(self.cmin)
        self.cmax = max(self.cmax)
        
        self.xT = [xa1 - self.axWid/8 + (1/4)*self.axWid*((self.t - self.minT)/(self.maxT-self.minT)) for xa1 in self.xa]
        self.hrfT = [0]*self.trial_types # access via: condition, channel, chromophore
        for trial in range(self.trial_types):
            self.hrfT[trial] = self.ya - self.axHgt/8 + (1/4)*self.axHgt*((self.hrf_val[trial] - self.cmin)/(self.cmax - self.cmin))
        
        # Update Conditions in list widget
        for tidx,trial in enumerate(self.snirfData.trial_type.values):
            self.conditions.insertItem(tidx, str(trial))
            
        t1 = time.time()
        print(f'Calculations complete in {t1-t0:.2f} seconds!')
        self.draw_hrf()
        self.conditions.setCurrentRow(0)
        
    def _change_hrf_vis(self):
        for i_con in range(self.trial_types):
            if i_con == self.conditions.currentRow():
                for i_ch in range(self.channels):
                    if self.chan_dist[i_ch] >= self.channel_min_dist and self.chan_dist[i_ch] <= self.ssFadeThres:
                        for i_col in range(self.chromophores):
                            self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col].set_color(self.chrom[i_col] + [self.fade_factor])
                    elif self.chan_dist[i_ch] >= self.ssFadeThres and self.chan_dist[i_ch] <= self.channel_max_dist:
                        for i_col in range(self.chromophores):
                            self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col].set_color(self.chrom[i_col] + [1])
                    else:
                        for i_col in range(self.chromophores):
                            self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col].set_color(self.chrom[i_col] + [0])
            else:
                for i_ch in range(self.channels):
                    for i_col in range(self.chromophores):
                        self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col].set_color(self.chrom[i_col] + [0])
        
        self._ax.figure.canvas.draw()
        
    def _redraw_hrf(self):
        for i_con in range(self.trial_types):
            for i_ch in range(self.channels):
                for i_col in range(self.chromophores):
                    self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col].set_data(self.xT[i_ch],self.hrfT[i_con][i_ch][i_col])
        
        self._ax.figure.canvas.draw()
        
    def _condition_changed(self, s):
        # Pass the new condition and draw hrf again
        if self.conditions.currentItem() is None:
            pass
        elif self.conditions.currentItem().text() == "-- Select Condition --":
            pass
        else:
            self._change_hrf_vis()
        
    def _toggle_circles(self):
        if self.opt2circ.isChecked():
            self.src_optodes.set_color([1,0,0])
            self.det_optodes.set_color([0,0,1])
            
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_color([1,0,0,0])
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_color([0,0,1,0])
        else:
            self.src_optodes.set_color([1,0,0,0])
            self.det_optodes.set_color([0,0,1,0])
            
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_color([1,0,0,1])
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_color([0,0,1,1])
                
        self._ax.figure.canvas.draw()
        
    def _toggle_measline(self):
        if self.measline.isChecked():
            for i_ch in range(self.channels):
                self.meas_line[i_ch].set_color([0.8,0.8,0.8,1])
        else:
            for i_ch in range(self.channels):
                self.meas_line[i_ch].set_color([0.8,0.8,0.8,0])
        
        self._ax.figure.canvas.draw()
        
    def _xscale_changed(self, i):
        # Pass the new xscale and draw hrf again
        self.plot_Xscale = i
        # print(f"Changing x-scale to {self.plot_Xscale}!")
        self.axWid = self.plot_Xscale/self.nAcross
        self.xT = [xa1 - self.axWid/8 + (1/4)*self.axWid*((self.t - self.minT)/(self.maxT-self.minT)) for xa1 in self.xa]
        
        self._redraw_hrf()
        
    def _yscale_changed(self, i):
        # Pass the new yscale and draw hrf again
        self.plot_Yscale = i
        # print(f"Changing y-scale to {self.plot_Yscale}!")
        self.axHgt = self.plot_Yscale/self.nUp
        for trial in range(self.trial_types):
            self.hrfT[trial] = self.ya - self.axHgt/8 + (1/4)*self.axHgt*((self.hrf_val[trial] - self.cmin)/(self.cmax - self.cmin))
        
        self._redraw_hrf()
        # print("HRFs should have changed!")
        
    def _mindist_changed(self, i):
        # Pass the new minimum channel distance and draw hrf again
        self.channel_min_dist = i
        
        if self.ssfade.value() < i:
            self.ssfade.setValue(i)
        else:
            self._change_hrf_vis()
        
    def _maxdist_changed(self, i):
        # Pass the new maximum channel distance and draw hrf again
        self.channel_max_dist = i
        self._change_hrf_vis()
        
    def _ssfade_changed(self, i):
        # Pass the fade amount and draw hrf again
        self.ssFadeThres = i
        self._change_hrf_vis()
        
    def draw_hrf(self):
        print("Plotting Optodes!")
        t0 = time.time()
        self._ax.clear()
        
        # Plot optode dots transparently
        self.src_optodes, = self._ax.plot(self.sx, self.sy, 'o', markersize=5, color=[1,0,0,0])
        self.det_optodes, = self._ax.plot(self.dx, self.dy, 'o', markersize=5, color=[0,0,1,0])
        
        # Plot optode labels
        for idx2, source in enumerate(self.sPos.label):
            self.src_label[idx2] = self._ax.text(self.sx[idx2], self.sy[idx2], f"{source.values}", fontsize=8, ha='center', va='center', clip_on=True)
            self.src_label[idx2].set_color([1,0,0,1])

        for idx2, detector in enumerate(self.dPos.label):
            self.det_label[idx2] = self._ax.text(self.dx[idx2], self.dy[idx2], f"{detector.values}", fontsize=8, ha='center', va='center', clip_on=True)
            self.det_label[idx2].set_color([0,0,1,1])
        
        print("Plotting HRFs!")
        
        for i_con in range(self.trial_types):
            for i_ch in range(self.channels):
                for i_col in range(self.chromophores):
                    self.hrf[i_con*self.channels*self.chromophores + i_ch*self.chromophores + i_col], = self._ax.plot(
                        self.xT[i_ch],
                        self.hrfT[i_con][i_ch][i_col],
                        lw = self.lineWidth,
                        zorder=2 - i_col,
                        color=self.chrom[i_col] + [0],
                        )
        
        for i_ch in range(self.channels):
            si = self.src_idx[i_ch]
            di = self.det_idx[i_ch]
            
            self.meas_line[i_ch], = self._ax.plot(
                [self.sx[si],self.dx[di]],
                [self.sy[si],self.dy[di]],
                '--',
                color=[0.8,0.8,0.8,0],
                zorder=0,
                )

        self._ax.set_aspect('equal')
        self._ax.axis('off')
        self._ax.figure.tight_layout()
        self._ax.figure.canvas.draw()
        
        t1 = time.time()
        print(f"Everything plotted in {t1-t0:.2f} seconds!")


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     main_gui = Main()
#     main_gui.show()
#     sys.exit(app.exec())


def run_vis(snirfData = None, geo2d = None, geo3d = None):
    """
    Parameters
    ----------
    snirfData : xarray, optional
        Pass the xarray with the HRF data. The default is None.
    geo2d : xarray, optional
        Pass the xarray with the 2d probe geometry data. The default is None.
    geo3d : xarray, optional
        Pass the xarray with the 3d probe geometry data. The default is None.

    Opens a gui that loads the HRF, if given.

    Raises
    ------
    Exception
        If the argument passed is not valid, raises an exception.

    """    
    
    if snirfData is not None and type(snirfData) != xr.core.dataarray.DataArray:
        raise Exception("Please provide valid snirfData!")
    elif geo2d is not None and type(geo2d) != xr.core.dataarray.DataArray or np.shape(geo2d)[1] != 2:
        raise Exception("Please provide valide geo2d!")
    elif geo3d is not None and type(geo3d) != xr.core.dataarray.DataArray or np.shape(geo3d)[1] != 3:
        raise Exception("Please provide valid geo3d!")
    else:
        app = QtWidgets.QApplication(sys.argv)
        main_gui = Main(snirfData = snirfData, geo2d = geo2d, geo3d = geo3d)
        main_gui.show()
        sys.exit(app.exec())











