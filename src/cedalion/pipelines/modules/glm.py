import cedalion.nirs
import xarray as xr
import cedalion.pipelines

import cedalion.models.glm
import cedalion.pipelines.pipeline

units = cedalion.units

class GLM(cedalion.pipelines.pipeline.cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "GLM Model"
        self._cite=None
        self.options={'noise_model':'ols',
                      'ar_order':30,
                      'max_jobs':1,
                      'basis_function':cedalion.models.glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
                      'short_channel_method': None,
                      'drift_order':0,
                      'verbose':True}
        self.inputName='conc'
        self.outputName='stats'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        

        design_matrix,channel_wise_regressors =  cedalion.models.glm.make_design_matrix(
            ts_long=rec[self.inputName],
            ts_short=None,
            stim=rec.stim,
            geo3d=rec.geo3d,
            basis_function=self.options['basis_function'],
            drift_order=self.options['drift_order'],
            short_channel_method=self.options['short_channel_method'])

       
        rec[self.outputName] = cedalion.models.glm.fit(rec[self.inputName],
                                        design_matrix=design_matrix,
                                        channel_wise_regressors=channel_wise_regressors,
                                        ar_order=self.options['ar_order'],
                                        noise_model=self.options['noise_model'],
                                        max_jobs=self.options['max_jobs'],
                                        verbose=self.options['verbose'])
                                                
        return rec
    
