import cedalion.testing.simData as simdata
import cedalion.pipelines.modules.preproccessing as prep
import cedalion.pipelines.modules.glm as stats
import numpy as np
import matplotlib.pyplot as plt

def PipelineList(steps):
    job = None
    for step in steps:
        job=step(job)
    
    return job

class ChannelROC:
    def __init__(self):
        self.iterations=0
        self.data_simulation_function=simdata.Data
        self.data_simulation_args={'snr':5}
        self.pipeline=PipelineList([prep.intensity_opticaldensity,
                                    prep.mbll,
                                    stats.GLM])
        self.pipeline_args={'noise_model':'ar_irls'}
        self._truth=[]
        self._pvals=[]
        return
    
    def run(self,num_iterations):
        self.pipeline.set_all_options(self.pipeline_args)
        for iter in range(num_iterations):
            print(f"Interation {self.iterations}")
            rec,truth=self.data_simulation_function(**self.data_simulation_args)
            rec=self.pipeline.run(rec)
            val=rec['stats'].pvalue['HRF A'].to_numpy().reshape((2,len(rec['conc'].channel))).T
            self._pvals.append(val)
            self._truth.append(truth.data)
            self.iterations+=1
        return

    def reset(self):
        self._truth=[]
        self._pvals=[]
        self.iterations=0
        return
    
    def results(self):
        rr=np.concat(self._pvals)
        tt=np.concat(self._truth)

        tp=[]
        fp=[]
        th=[]
        for i in range(rr.shape[1]):
            t,f,h=self._roc_values(tt[:,i],rr[:,i])
            tp.append(t)
            fp.append(f)
            th.append(h)
        
        
        return tp, fp, th


    def draw(self):
        tp, fp, th = self.results()
        plt.subplot(121)
        for i in range(len(tp)):
            plt.plot(fp[i],tp[i])
        plt.title('Sensivity-Specificity')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.subplot(122)
        for i in range(len(tp)):
            plt.plot(th[i],fp[i])
        plt.title('Type-I error')
        plt.ylabel('False Positive Rate')
        plt.xlabel('Estimated FPR (p-value)')


    def _roc_values(self,truth, pval):
        nan_indices = np.isnan(truth)
        pval = pval[~nan_indices]
        truth = truth[~nan_indices]

        lst = np.where(truth)[0]
        lstN = np.where(~truth)[0]
    
        if len(lst) != len(lstN) and not np.all(truth == 0):
            n = min(len(lst), len(lstN))
            lst = np.random.choice(lst, n, replace=False)
            lstN = np.random.choice(lstN, n, replace=False)
            pval = np.concatenate((pval[lst], pval[lstN]))
            truth = np.concatenate((truth[lst], truth[lstN]))

        th, I = np.sort(pval), np.argsort(pval)
        tp = np.cumsum(truth[I]) / np.sum(truth)
        fp = np.cumsum(~truth[I]) / np.sum(~truth)

        return tp, fp, th
    
    

