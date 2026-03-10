"""Interactive GUI to view the HRF in probe space after data has been processed.

Based on Homer3 v1.80.2 "PlotProbe2.m" (:cite:t:`Huppert2009`)
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

Initial Contributors:
    - Sung Ahn | ahnsm@bu.edu | 2024
"""

from __future__ import annotations
import sys
import time
import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
import xarray as xr
import matplotlib.colors as mcolors


import cedalion
import cedalion.typing as cdt

warnings.simplefilter("ignore")


class _MAIN_GUI(QtWidgets.QMainWindow):
    def __init__(self, snirfData=None, geo2d=None, geo3d=None, stderr=None, reject=None, 
                 chan_roi_df=None, roi_color_map=None, roi_alpha=0.25, roi_marker_size=90):
        # Initialize
        super().__init__()
        self.snirfData = snirfData
        self.stderr = stderr  
        self.geo2d = geo2d
        self.geo3d = geo3d
        self.reject = reject  # reject parameter

        # for plotting ROIs
        self.chan_roi_df = chan_roi_df
        self.roi_color_map = roi_color_map or {}
        self.roi_alpha = roi_alpha
        self.roi_marker_size = roi_marker_size
        self.roi_scatter = None
        self.ch_to_roi = {}
        # allow for adjustment of ROI alpha values via sliders in the control panel
        self.roi_alphas = {}          # e.g., {"IFG_L": 0.20, "SFG_R": 0.10, ...}
        self.roi_alpha_sliders = {}   # roi -> slider widget
        self.roi_alpha_widgets = {}  # roi -> (slider, spinbox, value_label)

        # Set central widget
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Initialize layout
        window_layout = QtWidgets.QVBoxLayout(self._main)
        window_layout.setContentsMargins(10, 0, 10, 10)
        window_layout.setSpacing(10)

        # Set Minimum Size
        self.setMinimumSize(800, 600)

        # Set Window Title
        self.setWindowTitle("Plot Probe")

        # Filler plot for now
        self.plotprobe = FigureCanvas(Figure(figsize=(10, 10)))
        self._ax = self.plotprobe.figure.subplots()
        self._ax.axis("off")
        window_layout.addWidget(NavigationToolbar(self.plotprobe, self), stretch=1)
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
        self.mindist.setRange(15, self.maxdist.value())
        self.maxdist.setRange(self.mindist.value(), 45)
        self.maxdist.setSingleStep(3)
        self.maxdist.valueChanged.connect(self._maxdist_changed)
        self.ssfade = QtWidgets.QDoubleSpinBox()
        self.ssfade.setValue(15)
        self.ssfade.setRange(15, 45)
        self.ssfade.setSingleStep(3)
        self.ssfade.valueChanged.connect(self._ssfade_changed)

        ## Populate Prune Channels
        prune_channels_layout.addWidget(QtWidgets.QLabel("Min dist"), 0, 0)
        prune_channels_layout.addWidget(self.mindist, 0, 1)
        prune_channels_layout.addWidget(QtWidgets.QLabel("Max dist"), 1, 0)
        prune_channels_layout.addWidget(self.maxdist, 1, 1)
        prune_channels_layout.addWidget(QtWidgets.QLabel("SS fade thresh"), 2, 0)
        prune_channels_layout.addWidget(self.ssfade, 2, 1)

        ## Add Prune Channels
        control_panel_layout.addWidget(prune_channels, stretch=1)

        # Create significance filter control - only if reject is provided
        if self.reject is not None:  
            sig_control = QtWidgets.QGroupBox("Significance Filter")
            sig_control_layout = QtWidgets.QVBoxLayout()
            sig_control_layout.setSpacing(10)
            sig_control.setLayout(sig_control_layout)
            
            ## Set up significance filter checkbox
            self.show_sig_only = QtWidgets.QCheckBox("Show only channels significant in both HbO and HbR")
            self.show_sig_only.setChecked(False)
            self.show_sig_only.stateChanged.connect(self._sig_filter_changed)
            sig_control_layout.addWidget(self.show_sig_only)
            
            ## Add significance control
            control_panel_layout.addWidget(sig_control, stretch=1)

        # Create t-stat thresh control - only if standard error is provided
        if self.stderr is not None:  
            tstat_control = QtWidgets.QGroupBox("T-Stat Threshold")
            tstat_control_layout = QtWidgets.QVBoxLayout()
            tstat_control_layout.setSpacing(10)
            tstat_control.setLayout(tstat_control_layout)
            
            ## Set up T-stat threshold controller
            tstat_threshold_layout = QtWidgets.QHBoxLayout()
            self.tstat_threshold = QtWidgets.QDoubleSpinBox()
            self.tstat_threshold.setValue(0)  # Default: no threshold
            self.tstat_threshold.setRange(-100, 100)
            self.tstat_threshold.setSingleStep(0.5)
            self.tstat_threshold.setDecimals(2)
            self.tstat_threshold.valueChanged.connect(self._tstat_threshold_changed)
            tstat_threshold_layout.addWidget(QtWidgets.QLabel("Threshold"))
            tstat_threshold_layout.addWidget(self.tstat_threshold)
            tstat_control_layout.addLayout(tstat_threshold_layout)
            
            ## Add T-stat control
            control_panel_layout.addWidget(tstat_control, stretch=1)

        ## Create Probe Control
        probe_control = QtWidgets.QGroupBox("Probe")
        probe_control_layout = QtWidgets.QVBoxLayout()
        probe_control_layout.setSpacing(5)
        probe_control.setLayout(probe_control_layout)

        ## Create ROI Alpha Controls (one slider per ROI)
        roi_alpha_group = QtWidgets.QGroupBox("ROI Alpha")
        roi_alpha_layout = QtWidgets.QVBoxLayout()
        roi_alpha_group.setLayout(roi_alpha_layout)
        # A scroll area so it doesn't explode the GUI if you have lots of ROIs
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(6)
        scroll.setWidget(scroll_content)
        roi_alpha_layout.addWidget(scroll)
        # Add ROI Alpha controls panel to the main control panel row
        control_panel_layout.addWidget(roi_alpha_group, stretch=1)
        self._roi_alpha_scroll_layout = scroll_layout  # store for later population


        ## Set up Probe Control controllers
        self.opt2circ = QtWidgets.QCheckBox("View optodes as circles")
        self.opt2circ.stateChanged.connect(self._toggle_circles)
        self.measline = QtWidgets.QCheckBox("Display Measurement Line")
        self.measline.stateChanged.connect(self._toggle_measline)

        # toggle for ROIs
        self.show_rois = QtWidgets.QCheckBox("Show ROI overlay")
        self.show_rois.setChecked(True)
        self.show_rois.stateChanged.connect(self._toggle_rois)
        probe_control_layout.addWidget(self.show_rois)

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
        open_btn = QtGui.QAction("Open...", self)
        open_btn.setStatusTip("Open SNIRF file")
        open_btn.triggered.connect(self._open_dialog)

        ## Create menu
        menu = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu)

        ## Populate menu
        file_menu = menu.addMenu("&File")
        file_menu.addAction(open_btn)

        if self.snirfData is not None:
            time_dim = 'reltime' if 'reltime' in self.snirfData.dims else 'time' # Detect time dimension
            if np.shape(self.snirfData)[1] != len(self.snirfData.channel):
                self.snirfData = self.snirfData.transpose(
                    "trial_type", "channel", "chromo", time_dim
                )

            self.sPos = self.geo2d.sel(
                label=["S" in s for s in self.geo2d.label.values]
            )
            self.dPos = self.geo2d.sel(
                label=["D" in s for s in self.geo2d.label.values]
            )

            self.sourcePos3D = self.geo3d.sel(
                label=["S" in str(s.values) for s in self.geo3d.label]
            )
            self.detectorPos3D = self.geo3d.sel(
                label=["D" in str(s.values) for s in self.geo3d.label]
            )

            print("starting calculations!")
            self._init_calc()

    def _open_dialog(self):
        # Grab the appropriate SNIRF file
        self._fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "SNIRF Files (*.snirf)",
        )[0]
        print("Loading SNIRF...")
        t0 = time.time()
        self.snirfObj = cedalion.io.read_snirf(self._fname)
        t1 = time.time()
        print(f"SNIRF Loaded in {t1 - t0:.2f} seconds!")

        # Extract necessary data
        self.snirfData = self.snirfObj[0].data[0]

        self.sPos = self.snirfObj[0].geo2d.sel(
            label=["S" in str(s.values) for s in self.snirfObj[0].geo2d.label]
        )
        self.dPos = self.snirfObj[0].geo2d.sel(
            label=["D" in str(s.values) for s in self.snirfObj[0].geo2d.label]
        )

        self.sourcePos3D = self.snirfObj[0].geo3d.sel(
            label=["S" in str(s.values) for s in self.snirfObj[0].geo3d.label]
        )
        self.detectorPos3D = self.snirfObj[0].geo3d.sel(
            label=["D" in str(s.values) for s in self.snirfObj[0].geo3d.label]
        )

        self._init_calc()

    def _init_calc(self):
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
        self.fade_factor = 0.3  ##### Connect?
        self.lineWidth = 0.9  ##### Connect?

        if 'reltime' in self.snirfData.dims: # handle 'time' or 'reltime'
            self.time_dim = 'reltime'
        elif 'time' in self.snirfData.dims:
            self.time_dim = 'time'
        else:
            raise ValueError("Data must have either 'time' or 'reltime' dimension")

        # Process reject data to find channels significant in BOTH HbO and HbR
        if self.reject is not None:
            self.sig_channels_both = {}  # Dictionary: trial_type -> list of channel indices
            
            for trial_idx, trial_type in enumerate(self.reject.trial_type.values):
                # Get significance for HbO and HbR
                sig_HbO = self.reject.sel(trial_type=trial_type, chromo='HbO').values
                sig_HbR = self.reject.sel(trial_type=trial_type, chromo='HbR').values
                
                # Find channels where EITHER/AND are significant
                sig_both = sig_HbO | sig_HbR
                
                # Store channel indices that are significant in both
                self.sig_channels_both[str(trial_type)] = np.where(sig_both)[0]
                
                print(f"Trial {trial_type}: {len(self.sig_channels_both[str(trial_type)])} channels significant in either HbO and HbR")
        else:
            self.sig_channels_both = None


        # T-stat calculation
        self.tstat_thresh = 0  # Initialize threshold
        if self.stderr is not None:
            # Calculate t-statistic: mean / stderr
            self.tstat = self.snirfData / self.stderr
            # Get max absolute t-stat per channel across time for thresholding
            self.tstat_max = np.abs(self.tstat).max(dim=self.time_dim)  # Max across time
        else:
            self.tstat = None
            self.tstat_max = None

        self.conditions.clear()
        self.opt2circ.setChecked(False)
        self.measline.setChecked(False)

        # Color for conditions
        self.color_HbO = [0.862, 0.078, 0.235]  ##### Connect?
        self.color_HbR = [0, 0, 0.8]  ##### Connect?
        self.chrom = {0: self.color_HbO, 1: self.color_HbR}

        self.sPosVal = self.sPos.values
        self.dPosVal = self.dPos.values

        self.src_idx = [
            np.arange(0, len(self.sPos))[self.sPos.label == src][0]
            for src in self.snirfData.source
        ]
        self.det_idx = [
            np.arange(0, len(self.dPos))[self.dPos.label == det][0]
            for det in self.snirfData.detector
        ]

        # Find Channel distances
        self.chan_dist = np.sqrt(
            np.sum(
                (
                    self.sourcePos3D.values[self.src_idx]
                    - self.detectorPos3D.values[self.det_idx]
                )
                ** 2,
                1,
            )
        )

        # Find the extreme coordinates of the optodes
        self.sdMin = np.array(
            [
                min([min(self.sPos[:, 0]), min(self.dPos[:, 0])]).values,
                min([min(self.sPos[:, 1]), min(self.dPos[:, 1])]).values,
            ]
        )
        self.sdMin = self.sdMin - np.mean(self.chan_dist)
        self.sdMax = np.array(
            [
                max([max(self.sPos[:, 0]), max(self.dPos[:, 0])]).values,
                max([max(self.sPos[:, 1]), max(self.dPos[:, 1])]).values,
            ]
        )
        self.sdMax = self.sdMax + np.mean(self.chan_dist)

        # Find the scaling factors for the plot
        self.sdWid = self.sdMax[0] - self.sdMin[0]
        self.sdHgt = self.sdMax[1] - self.sdMin[1]

        # Find the axis scale
        self.sd2axScl = max(self.sdWid, self.sdHgt)

        # Scale the optode coordinates by the scale
        self.sPosVal /= self.sd2axScl
        self.dPosVal /= self.sd2axScl

        # Calculate the scaling/translation factors
        self.nAcross = (
            len(np.unique(np.append(self.sPosVal, self.dPosVal, axis=0)[:, 0])) + 1
        )
        self.axXoff = np.mean(np.append(self.sPosVal, self.dPosVal, axis=0)[:, 0]) - 0.5
        self.nUp = (
            len(np.unique(np.append(self.sPosVal, self.dPosVal, axis=0)[:, 1])) + 1
        )
        self.axYoff = np.mean(np.append(self.sPosVal, self.dPosVal, axis=0)[:, 1]) - 0.5

        # Calculate the size of each HRF
        self.axWid = self.plot_Xscale / self.nAcross
        self.axHgt = self.plot_Yscale / self.nUp

        # Extract the x-y coordinates of the optodes
        self.sx = self.sPosVal[:, 0] - self.axXoff
        self.sy = self.sPosVal[:, 1] - self.axYoff
        self.dx = self.dPosVal[:, 0] - self.axXoff
        self.dy = self.dPosVal[:, 1] - self.axYoff

        # Extract time information
        try:
            self.t = self.snirfData[self.time_dim].values  # CHANGE to use self.time_dim
        except Exception:
            # Fallback to trying both
            try:
                self.t = self.snirfData.time.values
            except Exception:
                self.t = self.snirfData.reltime.values
        # try:
        #     self.t = self.snirfData.time.values
        # except Exception:
        #     pass
        # try:
        #     self.t = self.snirfData.reltime.values
        # except Exception:
        #     pass

        self.minT = min(self.t)
        self.maxT = max(self.t)

        # Extract lengths
        self.trial_types = len(self.snirfData.trial_type)
        self.channels = len(self.snirfData.channel)
        self.chromophores = len(self.snirfData.chromo)

        # Initialize holders to control each part of the plot
        self.src_label = [0] * len(self.sx)
        self.det_label = [0] * len(self.dx)
        self.meas_line = [0] * self.channels
        self.hrf = (
            [0] * self.chromophores * self.channels * self.trial_types
        )  # access via: condition*no_chans + channel*no_chrom + chromophore

        # Calculate the HRF plot coordinates
        self.xa = (self.sx[self.src_idx] + self.dx[self.det_idx]) / 2
        self.ya_mid = (self.sy[self.src_idx] + self.dy[self.det_idx]) / 2  # <-- save this!
        self.hrf_val = [
            self.snirfData.sel(trial_type=i).values for i in self.snirfData.trial_type
        ]
        self.ya = np.array([[[a] * len(self.t)] * self.chromophores for a in self.ya_mid])

        self.cmin = [0] * self.trial_types
        self.cmax = [0] * self.trial_types

        for trial in range(self.trial_types):
            self.cmin[trial] = np.min(np.nan_to_num(self.hrf_val[trial].ravel(), 10))
            self.cmax[trial] = np.max(np.nan_to_num(self.hrf_val[trial].ravel(), -10))

        self.cmin = min(self.cmin)
        self.cmax = max(self.cmax)

        self.xT = [
            xa1
            - self.axWid / 8
            + (1 / 4) * self.axWid * ((self.t - self.minT) / (self.maxT - self.minT))
            for xa1 in self.xa
        ]
        self.hrfT = [
            0
        ] * self.trial_types  # access via: condition, channel, chromophore
        for trial in range(self.trial_types):
            self.hrfT[trial] = (
                self.ya
                - self.axHgt / 8
                + (1 / 4)
                * self.axHgt
                * ((self.hrf_val[trial] - self.cmin) / (self.cmax - self.cmin))
            )

        # Update Conditions in list widget
        for tidx, trial in enumerate(self.snirfData.trial_type.values):
            self.conditions.insertItem(tidx, str(trial))


        self.channel_ids = self.snirfData.channel.values # store channel ids # These are typically strings like "S1", "S10", and "D1", "D87"
        self.src_labels = [str(s) for s in self.snirfData.source.values]
        self.det_labels = [str(d) for d in self.snirfData.detector.values]
         # Build channel->ROI lookup (only if provided)
        self.ch_to_roi = {}
        if self.chan_roi_df is not None:
            df = self.chan_roi_df.copy() # Expect columns: 'channel' and 'ROI'
            df = df.dropna(subset=["ROI"]) # drop NaN ROI rows
            # Build mapping from channel coordinate value -> ROI
            # (Works whether your channel labels are 0..N-1 or other ints)
            # self.ch_to_roi = dict(zip(df["channel"].astype(int).values, df["ROI"].astype(str).values))
            self.ch_to_roi = dict(zip(df["channel"].astype(str).values, df["ROI"].astype(str).values))
        print("ROI overlay mapping size:", len(self.ch_to_roi))  # helpful debug

        # Build ROI alpha sliders
        self._init_roi_alpha_controls()

        t1 = time.time()
        print(f"Calculations complete in {t1-t0:.2f} seconds!")
        self._draw_hrf()
        self.conditions.setCurrentRow(0)


    # def _change_hrf_vis(self):  # orig
    #     for i_con in range(self.trial_types):
    #         if i_con == self.conditions.currentRow():
    #             for i_ch in range(self.channels):
    #                 if (
    #                     self.chan_dist[i_ch] >= self.channel_min_dist
    #                     and self.chan_dist[i_ch] <= self.ssFadeThres
    #                 ):
    #                     for i_col in range(self.chromophores):
    #                         self.hrf[
    #                             i_con * self.channels * self.chromophores
    #                             + i_ch * self.chromophores
    #                             + i_col
    #                         ].set_color(self.chrom[i_col] + [self.fade_factor])
    #                 elif (
    #                     self.chan_dist[i_ch] >= self.ssFadeThres
    #                     and self.chan_dist[i_ch] <= self.channel_max_dist
    #                 ):
    #                     for i_col in range(self.chromophores):
    #                         self.hrf[
    #                             i_con * self.channels * self.chromophores
    #                             + i_ch * self.chromophores
    #                             + i_col
    #                         ].set_color(self.chrom[i_col] + [1])
    #                 else:
    #                     for i_col in range(self.chromophores):
    #                         self.hrf[
    #                             i_con * self.channels * self.chromophores
    #                             + i_ch * self.chromophores
    #                             + i_col
    #                         ].set_color(self.chrom[i_col] + [0])
    #         else:
    #             for i_ch in range(self.channels):
    #                 for i_col in range(self.chromophores):
    #                     self.hrf[
    #                         i_con * self.channels * self.chromophores
    #                         + i_ch * self.chromophores
    #                         + i_col
    #                     ].set_color(self.chrom[i_col] + [0])

    #     self._ax.figure.canvas.draw()
    
    def _change_hrf_vis(self):
        for i_con in range(self.trial_types):
            if i_con == self.conditions.currentRow():
                current_trial_type = str(self.snirfData.trial_type.values[i_con])
                
                for i_ch in range(self.channels):
                    # Check if channel is significant in both HbO and HbR
                    is_sig_both = False
                    if self.sig_channels_both is not None:
                        is_sig_both = i_ch in self.sig_channels_both.get(current_trial_type, [])
                    
                    # Determine base alpha based on distance
                    if (
                        self.chan_dist[i_ch] >= self.channel_min_dist
                        and self.chan_dist[i_ch] <= self.ssFadeThres
                    ):
                        base_alpha = self.fade_factor
                    elif (
                        self.chan_dist[i_ch] >= self.ssFadeThres
                        and self.chan_dist[i_ch] <= self.channel_max_dist
                    ):
                        base_alpha = 1
                    else:
                        base_alpha = 0
                    
                    # Apply significance filter if enabled
                    if hasattr(self, 'show_sig_only') and self.show_sig_only.isChecked():
                        if not is_sig_both and base_alpha > 0:
                            base_alpha = base_alpha * 0.15  # Heavily fade non-significant channels
                    
                    # Set color for each chromophore
                    for i_col in range(self.chromophores):
                        self.hrf[
                            i_con * self.channels * self.chromophores
                            + i_ch * self.chromophores
                            + i_col
                        ].set_color(self.chrom[i_col] + [base_alpha])
            else:
                for i_ch in range(self.channels):
                    for i_col in range(self.chromophores):
                        self.hrf[
                            i_con * self.channels * self.chromophores
                            + i_ch * self.chromophores
                            + i_col
                        ].set_color(self.chrom[i_col] + [0])

        self._ax.figure.canvas.draw()

    # def _change_hrf_vis(self): # TSTAT
    #     for i_con in range(self.trial_types):
    #         if i_con == self.conditions.currentRow():
    #             for i_ch in range(self.channels):
    #                 # Check if channel meets t-stat threshold
    #                 meets_tstat = True
    #                 if self.tstat_max is not None:
    #                     # Check if ANY chromophore meets threshold for this channel
    #                     meets_tstat = any(
    #                         self.tstat_max.sel(trial_type=self.snirfData.trial_type[i_con]).values[i_ch, i_col] >= self.tstat_thresh
    #                         for i_col in range(self.chromophores)
    #                     )
                    
    #                 # Determine alpha based on distance and t-stat
    #                 if (
    #                     self.chan_dist[i_ch] >= self.channel_min_dist
    #                     and self.chan_dist[i_ch] <= self.ssFadeThres
    #                 ):
    #                     base_alpha = self.fade_factor
    #                 elif (
    #                     self.chan_dist[i_ch] >= self.ssFadeThres
    #                     and self.chan_dist[i_ch] <= self.channel_max_dist
    #                 ):
    #                     base_alpha = 1
    #                 else:
    #                     base_alpha = 0
                    
    #                 # Apply t-stat threshold: fade if doesn't meet threshold
    #                 if not meets_tstat and base_alpha > 0:
    #                     base_alpha = base_alpha * 0.3  # Further fade channels below threshold
                    
    #                 # Set color for each chromophore
    #                 for i_col in range(self.chromophores):
    #                     self.hrf[
    #                         i_con * self.channels * self.chromophores
    #                         + i_ch * self.chromophores
    #                         + i_col
    #                     ].set_color(self.chrom[i_col] + [base_alpha])
    #         else:
    #             for i_ch in range(self.channels):
    #                 for i_col in range(self.chromophores):
    #                     self.hrf[
    #                         i_con * self.channels * self.chromophores
    #                         + i_ch * self.chromophores
    #                         + i_col
    #                     ].set_color(self.chrom[i_col] + [0])

        # self._ax.figure.canvas.draw()

    def _re_draw_hrf(self):
        for i_con in range(self.trial_types):
            for i_ch in range(self.channels):
                for i_col in range(self.chromophores):
                    self.hrf[
                        i_con * self.channels * self.chromophores
                        + i_ch * self.chromophores
                        + i_col
                    ].set_data(self.xT[i_ch], self.hrfT[i_con][i_ch][i_col])

        self._ax.figure.canvas.draw()

    def _condition_changed(self, s):
        # Pass the new condition and draw hrf again
        if self.conditions.currentItem() is None:
            pass
        elif self.conditions.currentItem().text() == "-- Select Condition --":
            pass
        else:
            self._change_hrf_vis()

    def _sig_filter_changed(self):
        # Toggle significance filter
        self._change_hrf_vis()

    def _toggle_circles(self):
        if self.opt2circ.isChecked():
            self.src_optodes.set_color([1, 0, 0])
            self.det_optodes.set_color([0, 0, 1])
            self.src_optodes.set_markersize(3)  # ADD THIS - Make circles smaller
            self.det_optodes.set_markersize(3)  # ADD THIS - Make circles smaller

            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_color([1, 0, 0, 0])
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_color([0, 0, 1, 0])
        else:
            self.src_optodes.set_color([1, 0, 0, 0])
            self.det_optodes.set_color([0, 0, 1, 0])
            self.src_optodes.set_markersize(5)  # ADD THIS - Reset to original size
            self.det_optodes.set_markersize(5)  # ADD THIS - Reset to original size

            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_color([1, 0, 0, 1])
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_color([0, 0, 1, 1])

        self._ax.figure.canvas.draw()

    def _toggle_measline(self):
        if self.measline.isChecked():
            for i_ch in range(self.channels):
                self.meas_line[i_ch].set_color([0.8, 0.8, 0.8, 1])
        else:
            for i_ch in range(self.channels):
                self.meas_line[i_ch].set_color([0.8, 0.8, 0.8, 0])

        self._ax.figure.canvas.draw()

    def _xscale_changed(self, i):
        # Pass the new xscale and draw hrf again
        self.plot_Xscale = i
        # print(f"Changing x-scale to {self.plot_Xscale}!")
        self.axWid = self.plot_Xscale / self.nAcross
        self.xT = [
            xa1
            - self.axWid / 8
            + (1 / 4) * self.axWid * ((self.t - self.minT) / (self.maxT - self.minT))
            for xa1 in self.xa
        ]

        self._re_draw_hrf()

    def _yscale_changed(self, i):
        # Pass the new yscale and draw hrf again
        self.plot_Yscale = i
        # print(f"Changing y-scale to {self.plot_Yscale}!")
        self.axHgt = self.plot_Yscale / self.nUp
        for trial in range(self.trial_types):
            self.hrfT[trial] = (
                self.ya
                - self.axHgt / 8
                + (1 / 4)
                * self.axHgt
                * ((self.hrf_val[trial] - self.cmin) / (self.cmax - self.cmin))
            )

        self._re_draw_hrf()
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

    def _tstat_threshold_changed(self, i):
        # Pass the new t-stat threshold and update HRF visibility
        self.tstat_thresh = i
        self._change_hrf_vis()

    def _draw_hrf(self):
        print("Plotting Optodes!")
        t0 = time.time()
        self._ax.clear()

        # Plot optode dots transparently
        (self.src_optodes,) = self._ax.plot(
            self.sx, self.sy, "o", markersize=5, color=[1, 0, 0, 0]
        )
        (self.det_optodes,) = self._ax.plot(
            self.dx, self.dy, "o", markersize=5, color=[0, 0, 1, 0]
        )

        # Plot optode labels
        for idx2, source in enumerate(self.sPos.label):
            self.src_label[idx2] = self._ax.text(
                self.sx[idx2],
                self.sy[idx2],
                f"{source.values}",
                fontsize=8,
                ha="center",
                va="center",
                clip_on=True,
            )
            self.src_label[idx2].set_color([1, 0, 0, 1])

        for idx2, detector in enumerate(self.dPos.label):
            self.det_label[idx2] = self._ax.text(
                self.dx[idx2],
                self.dy[idx2],
                f"{detector.values}",
                fontsize=8,
                ha="center",
                va="center",
                clip_on=True,
            )
            self.det_label[idx2].set_color([0, 0, 1, 1])

        self._draw_roi_overlay() # Draw ROI overlay if applicable

        print("Plotting HRFs!")

        for i_con in range(self.trial_types):
            for i_ch in range(self.channels):
                for i_col in range(self.chromophores):
                    (
                        self.hrf[
                            i_con * self.channels * self.chromophores
                            + i_ch * self.chromophores
                            + i_col
                        ],
                    ) = self._ax.plot(
                        self.xT[i_ch],
                        self.hrfT[i_con][i_ch][i_col],
                        lw=self.lineWidth,
                        zorder=2 - i_col,
                        color=self.chrom[i_col] + [0],
                    )

        for i_ch in range(self.channels):
            si = self.src_idx[i_ch]
            di = self.det_idx[i_ch]

            (self.meas_line[i_ch],) = self._ax.plot(
                [self.sx[si], self.dx[di]],
                [self.sy[si], self.dy[di]],
                "--",
                color=[0.8, 0.8, 0.8, 0],
                zorder=0,
            )


        self._ax.set_aspect("equal")
        self._ax.axis("off")
        self._ax.figure.tight_layout()
        self._ax.figure.canvas.draw()

        t1 = time.time()
        print(f"Everything plotted in {t1-t0:.2f} seconds!")

    def _toggle_rois(self):
        # simplest: just redraw the whole probe
        self._draw_hrf()
        self._change_hrf_vis()  # restore HRF visibility for current condition

    def _draw_roi_overlay(self):
        """Overlay semi-transparent ROI circles at channel midpoints."""

        if hasattr(self, "show_rois") and (not self.show_rois.isChecked()):
            return

        # remove old overlay if it exists
        if self.roi_scatter is not None:
            try:
                self.roi_scatter.remove()
            except Exception:
                pass
            self.roi_scatter = None

        if not self.ch_to_roi:
            return

        xs = []
        ys = []
        cs = []

        for i_ch in range(self.channels):
            # ch_id = int(self.channel_ids[i_ch])  # channel coordinate value
            # roi = self.ch_to_roi.get(ch_id, None)
            chan_key = f"{self.src_labels[i_ch]}{self.det_labels[i_ch]}"  # e.g. "S10D87"
            roi = self.ch_to_roi.get(chan_key, None)

            if roi is None:
                continue

            color = self.roi_color_map.get(roi, None)
            if color is None:
                # if ROI exists but no color defined, skip (or pick a default)
                continue

            xs.append(self.xa[i_ch])
            ys.append(self.ya_mid[i_ch])
            # cs.append(color) # to draw one alpha 
            rgba = mcolors.to_rgba(color)  # (r,g,b,a_base)
            a = float(self.roi_alphas.get(roi, self.roi_alpha))  # per-ROI alpha fallback
            cs.append((rgba[0], rgba[1], rgba[2], a))

        if not xs:
            return

        # Put circles above meas lines (zorder=1) but below HRFs (your HRFs use zorder ~ 1-2)
        # self.roi_scatter = self._ax.scatter(   # for only 1 alpha value for all rois
        #     xs, ys,
        #     s=self.roi_marker_size,
        #     c=cs,
        #     alpha=self.roi_alpha,
        #     edgecolors="none",
        #     zorder=1.2
        # )

        self.roi_scatter = self._ax.scatter(
            xs, ys,
            s=self.roi_marker_size,
            c=cs,              # cs already includes alpha
            edgecolors="none",
            zorder=1.2
        )


    def _init_roi_alpha_controls(self):
        """Create/refresh per-ROI alpha sliders based on roi_color_map."""
        # Clear any existing sliders in the scroll layout
        layout = getattr(self, "_roi_alpha_scroll_layout", None)
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.roi_alpha_sliders.clear()

        # Decide which ROIs to include:
        # - Only those with colors AND that appear in channel mapping (recommended)
        rois_in_data = set(self.ch_to_roi.values()) if self.ch_to_roi else set()
        rois = [r for r in self.roi_color_map.keys() if r in rois_in_data]

        # If you prefer: show all ROIs in roi_color_map regardless of whether used:
        # rois = list(self.roi_color_map.keys())

        rois = sorted(rois)

        # Initialize default per-ROI alpha if missing
        for roi in rois:
            if roi not in self.roi_alphas:
                self.roi_alphas[roi] = float(self.roi_alpha)  # start from global default

        # Add a slider row per ROI
        for roi in rois:
            row = QtWidgets.QHBoxLayout()

            lab = QtWidgets.QLabel(roi)
            lab.setMinimumWidth(70)
            row.addWidget(lab)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)  # 0..100 -> 0.00..1.00
            slider.setValue(int(round(self.roi_alphas[roi] * 100)))
            row.addWidget(slider, stretch=1)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(float(self.roi_alphas[roi]))
            spin.setFixedWidth(70)
            row.addWidget(spin)

            val_lab = QtWidgets.QLabel(f"{self.roi_alphas[roi]:.2f}")
            val_lab.setMinimumWidth(40)
            row.addWidget(val_lab)

            # --- connect signals (with signal blocking to avoid recursion) ---
            slider.valueChanged.connect(lambda v, r=roi: self._roi_alpha_changed_from_slider(r, v))
            spin.valueChanged.connect(lambda v, r=roi: self._roi_alpha_changed_from_spinbox(r, v))

            self.roi_alpha_widgets[roi] = (slider, spin, val_lab)

            container = QtWidgets.QWidget()
            container.setLayout(row)
            self._roi_alpha_scroll_layout.addWidget(container)


    def _roi_alpha_changed(self, roi, slider_val):
        # Handle slider to update alpha and redraw overlay
        a = float(slider_val) / 100.0
        self.roi_alphas[roi] = a

        # update label next to slider
        if roi in self.roi_alpha_sliders:
            _, val_lab = self.roi_alpha_sliders[roi]
            val_lab.setText(f"{a:.2f}")

        # redraw only overlay (fast) + canvas
        self._draw_roi_overlay()
        self._ax.figure.canvas.draw_idle()

    def _roi_alpha_changed_from_slider(self, roi, slider_val: int):
        a = float(slider_val) / 100.0
        self.roi_alphas[roi] = a

        if roi in self.roi_alpha_widgets:
            slider, spin, val_lab = self.roi_alpha_widgets[roi]
            spin.blockSignals(True)
            spin.setValue(a)
            spin.blockSignals(False)
            val_lab.setText(f"{a:.2f}")

        self._draw_roi_overlay()
        self._ax.figure.canvas.draw_idle()


    def _roi_alpha_changed_from_spinbox(self, roi, a: float):
        a = float(np.clip(a, 0.0, 1.0))
        self.roi_alphas[roi] = a

        if roi in self.roi_alpha_widgets:
            slider, spin, val_lab = self.roi_alpha_widgets[roi]
            slider.blockSignals(True)
            slider.setValue(int(round(a * 100)))
            slider.blockSignals(False)
            val_lab.setText(f"{a:.2f}")

        self._draw_roi_overlay()
        self._ax.figure.canvas.draw_idle()


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     main_gui = _MAIN_GUI()
#     main_gui.show()
#     sys.exit(app.exec())


# def run_vis(
#     blockaverage: cdt.NDTimeSeries,
#     geo2d: cdt.LabeledPoints,
#     geo3d: cdt.LabeledPoints,
#     stderr: cdt.NDTimeSeries = None,  # optional standerr input 
#     reject: xr.DataArray = None,
# ):
#     """Opens the visualization GUI.

#     Args:
#         blockaverage: The blockaveraged HRF data.
#         geo2d: The 2d probe geometry data.
#         geo3d: The 3d probe geometry data.
#     """

#     app = QtWidgets.QApplication(sys.argv)
#     #main_gui = _MAIN_GUI(snirfData=blockaverage, geo2d=geo2d, geo3d=geo3d)
#     main_gui = _MAIN_GUI(snirfData=blockaverage, geo2d=geo2d, geo3d=geo3d, stderr=stderr, reject=reject)
#     main_gui.show()
#     sys.exit(app.exec())



def run_vis(blockaverage, geo2d, geo3d, stderr=None, reject=None,
            chan_roi_df=None, roi_color_map=None, roi_alpha=0.25, roi_marker_size=90):

    # app = QtWidgets.QApplication(sys.argv)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    main_gui = _MAIN_GUI(
        snirfData=blockaverage,
        geo2d=geo2d,
        geo3d=geo3d,
        stderr=stderr,
        reject=reject,
        chan_roi_df=chan_roi_df,
        roi_color_map=roi_color_map,
        roi_alpha=roi_alpha,
        roi_marker_size=roi_marker_size,
    )
    main_gui.show()
    sys.exit(app.exec())

