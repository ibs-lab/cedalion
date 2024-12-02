import cedalion.nirs
import xarray as xr
import cedalion.pipelines
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.sigproc.TDDR

import cedalion.pipelines.pipeline

units = cedalion.units

class motion_splineSG(cedalion.pipelines.pipeline.cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "Spline based motion-correction"
        self._cite=  None
        self.options={'frame_size',10 * units.s}
        self.inputName='last'
        self.outputName='last'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):

        if(self.inputName=='last'):
            inputName= list(rec.timeseries.keys())[-1]
        else:
            inputName=self.inputName
        if(self.outputName=='last'):
            outputName= list(rec.timeseries.keys())[-1]
        else:
            outputName=self.outputName    

        rec[outputName] = motion_correct.motion_correct_splineSG(rec[inputName],
                                                                      frame_size=self.options['frame_size'])
        return rec
    

class TDDR(cedalion.pipelines.pipeline.cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "TDDR"
        self._cite='Fishburn, Frank A., Ludlum, Ruth S., Vaidya, Chandan J., and Medvedev, Andrei V. Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS. NeuroImage 184 (2019): 171-179.'
        self.options={'split_PosNeg':True,
                      'usePCA':True}
        self.inputName='last'
        self.outputName='last'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):

        if(self.inputName=='last'):
            inputName= list(rec.timeseries.keys())[-1]
        else:
            inputName=self.inputName
        if(self.outputName=='last'):
            outputName= list(rec.timeseries.keys())[-1]
        else:
            outputName=self.outputName

        Fs=1/(rec[inputName].time[1]-rec[inputName].time[0])

        rec[outputName] = cedalion.sigproc.TDDR.TDDR(rec[inputName],Fs=Fs,usePCA=self.options['usePCA'],
                                                     split_PosNeg=self.options['split_PosNeg'])
        return rec