import cedalion.nirs
import xarray as xr
import cedalion.pipelines

import cedalion.sigproc.frequency as freq
import cedalion.pipelines.pipeline

units = cedalion.units

class bandpass_filter(cedalion.pipelines.pipeline.cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "Band-pass filter"
        self._cite=None
        self.options={'fmax':1 *units.Hz,
                      'fmin':0.016*units.Hz,
                      'butter_order':4}
        self.inputName='od'
        self.outputName='od'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        rec[self.outputName] = freq.freq_filter(rec[self.inputName],
                                                fmin=self.options['fmin'],
                                                fmax=self.options['fmax'],
                                                butter_order=self.options['butter_order'])
                                                
        return rec
    
