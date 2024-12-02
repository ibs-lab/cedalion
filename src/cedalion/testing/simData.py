import cedalion
import cedalion.models
import cedalion.dataclasses as cdc
import cedalion.testing
import cedalion.testing.simData
import numpy as np
import xarray as xr
from scipy.linalg import toeplitz
import random
import scipy.signal
import cedalion.models.glm as glm
import cedalion.math
import cedalion.math.ar_model
import cedalion.pipelines
import cedalion.pipelines.modules.preproccessing as prep
import cedalion.testing.simEvents 
import copy

units = cedalion.units

def randAR( P=5):
    """
    Function to generate random P-th order AR coef for generating data with serial-correlations   
    Inputs: 
        P: int {default=5}
    Outputs:
        a: numpy array of ar coefficients 
    """ 
    a = [random.random() for _ in range(P)]
    a=np.flipud(np.cumsum(a))
    a=a / a.sum() * 0.99
    return a




def defaultProbe2D():
    """       
    This function returns a default simple 2D probe.  This is the same default used in the NIRS-toolbox 

    S1  S2  S3  S4  S5  S6  S7  S8  S9
     \ / \ / \ / \ / \ / \ / \ / \ /
      D1  D2  D3  D4  D5  D6  D7  D8

      Inputs:   None
      Outputs:  geo2d structure
                Dictionary containing:
                        sourceLabels:  List[np._str]
                        detectorLabels: List[np._str]
                        channel: xr.DataArray
                        source: xr.DataArray
                        detector: xr.DataArray
                        wavelength: List[float]
    
    TODO:   Add geo3d structure
            Add input arguements to all customization                      
    """

    dims = ["label", "pos"]
    attrs = {"units": "mm"}

    sourceLabels=[np.str_('S1'),np.str_('S2'),np.str_('S3'),
                  np.str_('S4'),np.str_('S5'),np.str_('S6'),
                  np.str_('S7'),np.str_('S8'),np.str_('S9')]
    detectorLabels=[np.str_('D1'),np.str_('D2'),np.str_('D3'),
                    np.str_('D4'),np.str_('D5'),np.str_('D6'),
                    np.str_('D7'),np.str_('D8')]
    landmarkLabels=[]
    types = (
        [cdc.PointType.SOURCE] * len(sourceLabels)
        + [cdc.PointType.DETECTOR] * len(detectorLabels)
        + [cdc.PointType.LANDMARK] * len(landmarkLabels)
        )

    labels = np.hstack([sourceLabels, detectorLabels]) #, landmarkLabels])

    sourcePos=np.ones((9,2))
    sourcePos[:,0]=[-80,-60,-40,-20,0,20,40,60,80]
    sourcePos[:,1]=0

    detectorPos=np.ones((8,2))
    detectorPos[:,0]=[-70,-50,-30,-10,10,30,50,70]
    detectorPos[:,1]=25


    positions = np.vstack([sourcePos, detectorPos]) #, landmarkPos])


    coords = {"label": ("label", labels), "type": ("label", types)}
    geo2d = xr.DataArray(positions, coords=coords, dims=dims, attrs=attrs)

    geo2d = geo2d.set_xindex("type")
    geo2d = geo2d.pint.quantify()

    # Now make the measurement list
    ml = [[0,0],[1,0],[1,1],[2,1],[2,2],[3,2],
          [3,3],[4,3],[4,4],[5,4],[5,5],[6,5],
          [6,6],[7,6],[7,7],[8,7]]

    channels=[]
    source=[]
    detector=[]
    for s,d in ml:
        channels.append(sourceLabels[s]+detectorLabels[d])
        source.append(sourceLabels[s])
        detector.append(detectorLabels[d])


    channels = xr.DataArray(data=channels,dims=['channel'],
                            coords={'channel':channels})
    channels=channels.assign_coords(source=('channel',source))
    channels=channels.assign_coords(detector=('channel',detector))
    
    measList={'sourceLabels':sourceLabels,
              'detectorLabels':detectorLabels,
              'channel':channels,
              'source':source,
              'detector':detector,
              'wavelength':[760.0,850.0]}

    return geo2d,measList




def ARnoise(geo2D=defaultProbe2D()[0],measList=defaultProbe2D()[1],
            t=np.round([(j+1)*.1 for j in range(0,3000)],3), P=10, sigma=0.33):
    """ This function will simulate fNIRS CW raw amp data with AR noise
    This matches the defaults for the nirs toolbox nirs.modules.simARNoise
    
    Inputs:
        geo2d.  2d probe structure.     {Default calls defaultProbe()}
        measList: Dictionary containing {Default calls defaultProbe()}
                channel: xr.DataArray
                wavelength: List[float]
        t:      time variable.  np.array {Default 0-300s @ 10Hz}
        P:      AR model order (calls randAR(P) to generate) {default 10}
        sigma:  Spatial covariance noise prior {default 0.33}

    Outputs:
        rec:    Cedalion recording structure 
    
    TODO:
        Test with probes other than the defaultProbe()
        Add cedalion verifier checks 

    """
 

    numchan = len(measList['channel'])*len(measList['wavelength'])
    
    nsamples = len(t)

    # Create the time dataarray
    t = xr.DataArray(data=t,dims=['time'],coords={'time':t})
    t.assign_coords(samples=('time',np.arange(nsamples)))

    # noise mean and spatial covariance
    mu = np.zeros(numchan);
    S = toeplitz([1]+ [sigma]*(numchan-1))
    e =  np.random.multivariate_normal(mu, S, nsamples)
    # This noise has spatial covariance (set by sigma) but no temporal correlation yet

    # add temporal covariance
    for i in range(numchan):
        a = randAR( P )  # This generates a random set of AR coeffients to smooth the data
        e[:,i] = scipy.signal.lfilter(np.ones(P+1), np.append(1,-a), e[:,i])
    
    # Convert to raw signal
    data = 100*np.exp(-e*0.005)
    data = np.reshape(data,(nsamples,len(measList['channel']),len(measList['wavelength'])))
  
    # Create the recording class
    rec=cedalion.dataclasses.recording.Recording()
    rec.geo2d = geo2D

    # Make a fake 3D geom
    rec.geo3d = xr.DataArray(data=np.concat((geo2D.data,np.zeros((geo2D.data.shape[0],1))),axis=1),
                             dims=geo2D.dims,coords=geo2D.coords,attrs=geo2D.attrs)
    #TODO generate a real 3D probe to use


    ts = cdc.build_timeseries(data, dims=["time", "channel", "wavelength"],
                            time=t.data,
                            channel=measList['channel'].data,
                            value_units=units.V,
                            time_units=units.s,
                            other_coords={"wavelength": measList['wavelength']})
    
    ts=ts.assign_coords(channel=measList['channel'])

    ts.time.attrs["units"] = cedalion.units.s
    ts.attrs['data_type_group']='unprocessed raw'
    rec['amp']=ts

    return rec





def Data(noise=None,stim=None,snr=0.5,
         channels=None,basis=glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
         modifiers=None):
    """
        Function for generating simulated recording with HRF added

        Inputs:
            noise:  cedalion recording class.   
                This is the data to use to add the synthetic HRF to.  This may be either a simulated noise recording or
                real data.  {Default use cedalion.testing.simData.ARnoise()}
            stim:   stim events pd.DataFrame
                This is the timing to use for generating the HRF.  {Default use simEvents.rand_stim_design()}
            snr:    float {default 0.5}
                SNR to use for generating HRF.  SNR is defined from the whitened innovations of the recording data
            channels:  np.array of indices (matching recording) to add events to
                If not provides {default None}, then randomly 1/2 of the channels will be selected and used
            basis:  glm basis set to use {default Gamma(tau=0s,sigma=3s,T=3s)}

        Outputs:
            rec:    recording with the HRF added to the amp (raw) timecourse 
            truth:  xr.DataArray indicating (binary) where the simulated HRF was added
                Note- the HRF is added in chromo space and passed back to OD and raw timecourses 
    """
    
    # I could not figure out why, but when I pass this directly as an input, it ran into wierd referencing issues 
    # where the stim kept propogating into later calls of the function. I was unable to figure out why since IMO, 
    # this should be the same as calling on the function definition
    if(noise is None):
        noise=cedalion.testing.simData.ARnoise()
    if(stim  is None):
        stim=cedalion.testing.simEvents.rand_stim_design()

    # We will add the HRF in chromo space, convert back to raw, and then copy the raw
    # back to the recording.  
    # Make a deep copy of the recording to avoid adding additional fields to it
    rec = copy.deepcopy(noise)
    rec.stim=stim
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    rec=job.run(rec)

    # Make a quick design model from the stim design
    dm,_ =  glm.make_design_matrix(
            ts_long=rec['conc'],
            ts_short=None,
            stim=rec.stim,
            geo3d=None,
            basis_function=basis,
            drift_order=0,
            short_channel_method=None)
    
    # If no channels were defined as "truth", then randomly select 1/2 the data to add
    if(channels is None):
        numchannels = len(rec['conc'].channel)
        # Generate a random permutation
        perm = np.random.permutation(np.arange(numchannels ))
        channels=np.zeros((numchannels,2));
        channels[perm[:np.uint8(np.round(numchannels/2))],0]=1
        channels[perm[:np.uint8(np.round(numchannels/2))],1]=channels[perm[:np.uint8(np.round(numchannels/2))],0]

    # Compute the autoregressivly whitened innovations to use in the calculation of SNR 
    inn = cedalion.math.ar_model.ar_filter(rec['conc'],pmax=10)

    # beta amplitudes to use per channel based on provided SNR 
    b=snr*np.std(inn.to_numpy(),axis=0)

    # Let's just do this to get the structure of beta.  (TODO- this is me being lazy)
    stats = glm.fit(rec['conc'], dm, None, noise_model="ols",max_jobs=1,verbose=False)
    betas=stats.results
    betas[betas.regressor=='HRF A',:,0]=b[:,0]   # TODO- this assumes the default probe
    betas[betas.regressor=='HRF A',:,1]=-b[:,1]
    betas[betas.regressor=='Drift 0',:,:]=0
    
    #Zero out the beta's for the negative channels
    betas[:,channels[:,0]==0,0]=0
    betas[:,channels[:,1]==0,1]=0

    # Add the HRF responses to the conc timecourse
    data=rec['conc']
    data=data.transpose('time','channel','chromo')
    hrf=glm.predict(data,betas,dm,None)
    hrf=hrf.transpose('time','channel','chromo')
    data.pint.magnitude[:]=data.pint.magnitude[:]+hrf.pint.magnitude[:]
    rec['conc']=data
    

    # Now, convert back to OD and then raw
    job=prep.conc2od()
    job=prep.opticaldensity_intensity(job)
    rec=job.run(rec)
    rec['amp']=rec['amp'].transpose('time','channel','wavelength')

    if(modifiers is not None):
        for mod in modifiers:
            rec['amp']=mod(rec['amp'])


    # Finally, copy the raw data structure back to the original recording
    noise['amp']=rec['amp']
    noise.stim=stim

    # Create the truth table denoting where the HRF was added
    truth=(channels==1)
    truth=xr.DataArray(data=truth,dims=['channel','chromo'],
                       coords={'channel':data.channel,
                               'chromo':['HbO','HbR']})

    return noise,truth



def simMotionArtifact(data,spikes_per_minute=2, shifts_per_minute=0.5, motionMask=None):
    
    if(hasattr(data,'wavelength')):
        data=data.transpose('time','channel','wavelength')
    else:
        data=data.transpose('time','channel','chromo')

    shp=data.shape
    dd=np.reshape(data.as_numpy().data,(shp[0],shp[1]*shp[2]))
    time=data.time.to_numpy()

    if motionMask is None:
        motionMask = np.ones(len(time), dtype=bool)

    num_spikes = np.int64(spikes_per_minute * (time[-1] - time[0]) / 60)
    num_shifts = np.int64(shifts_per_minute * (time[-1] - time[0]) / 60)

    lst = np.where(motionMask)[0]
    nsamp = len(lst)

    spike_inds = np.random.randint(1, nsamp - 1, size=num_spikes)
    shift_inds = np.random.randint(1, nsamp - 1, size=num_shifts)

    spike_amp_Z = 10 * np.random.randn(num_spikes)
    shift_amp_Z = 10 * np.random.randn(num_shifts)

    mu = np.mean(dd, axis=0)
    stds = np.std(dd, axis=0)

    for i in range(num_spikes):
        width = 9.9 * np.random.rand() + 0.1  # Spike duration of 0.1-10 seconds

        t_peak = time[spike_inds[i]]
        t_start = t_peak - width / 2
        t_inds = np.where((time > t_start) & (time <= t_peak))[0]
        spike_time = np.concatenate((time[t_inds], time[t_inds[-2::-1]])) - t_start

        amp=np.abs(spike_amp_Z[i])
        amp = amp+0.25 * np.abs(amp) * np.random.randn(len(stds))
        amp *= stds
        amp = amp+(2 * np.random.rand(len(spike_time), len(stds)) - 1) * 0.5 * amp

        tau = width / 2
        spike_dd = amp ** np.expand_dims((spike_time / tau),axis=1)*np.ones((1,amp.shape[1]))

        t_inds = np.arange(t_inds[0], min(t_inds[0] + spike_dd.shape[0], dd.shape[0]))

        dd[t_inds, :] += spike_dd[:len(t_inds), :]

    for i in range(num_shifts):
        shift_amt = shift_amp_Z[i] * stds
        dd[shift_inds[i]:, :] += shift_amt

    # Restore original mean intensity
    dd += (mu - np.mean(dd, axis=0))

    # Prevent negative intensities
    while np.any(dd <= 0):
        has_neg = np.any(dd < 0, axis=0)
        dd[:, has_neg] += np.random.rand(1, np.sum(has_neg)) * np.std(dd[:, has_neg], axis=0)

    data.data=np.reshape(dd,shp)
    return data