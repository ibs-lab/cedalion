
import pandas as pd
import numpy as np

""" 
This is a collection of functions for generating stimulus stiming information used in the simData function

    TODO Add BlockedDesign simulator

"""

def rand_stim_design(t=np.round([(j+1)*.1 for j in range(0,3000)],3), 
                     stim_dur=2, stim_space=7, ncond=1):
    """
        This function generates a random event-related design
        
        Inputs:
            t: time np.array {default 0-300s @ 10Hz}
            stim_dur: duration of events (float; seconds) {default 2s}
            stim_space: (float seconds) {default 7s}.  This is the average time between the onsets of events
            ncond:  Number of conditions to simulate {default 1}. Events are named "A","B",...

        Outputs:
            stim design: pandas.DataFrame
    """
    
    # min max times
    tmin = np.min(t) + 1 * stim_dur
    tmax = np.max(t) - 2 * stim_dur

    # number of stims onsets
    nrnd = round(2 * (tmax - tmin) / stim_space)

    # random times between tasks
    dt = stim_space / 2 + np.random.exponential(stim_space / 2, nrnd)

    # onsets
    onset = tmin + np.cumsum(np.concatenate(([0], dt)))
    onset = onset[onset < tmax]

    # durations
    dur = stim_dur * np.ones_like(onset)

    # amplitude
    amp = np.ones_like(dur)

    # output
    stim = {}
    r = np.random.randint(1, ncond + 1)

    stim=pd.DataFrame()
    for i in range(1, ncond + 1):
        lst = np.arange((i + r) % ncond + 1, len(dur) + 1, ncond) - 1

        stim = pd.concat([stim,
            pd.DataFrame({"onset": o, "trial_type": chr(65 + i - 1)} for o in onset[lst])] # Named "A","B", etc
        )
    stim["value"] = amp
    stim["duration"] = np.ones(len(dur))*stim_dur

    return stim
