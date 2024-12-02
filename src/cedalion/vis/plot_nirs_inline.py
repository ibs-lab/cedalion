import matplotlib.pyplot as plt
import numpy as np


def to_string(arr):
    # Convert to list of integers   
    return arr.values.astype(str).tolist()



def plot(rec,type='amp'):

    fig, ax = plt.subplots(1,2,figsize=(12,4),width_ratios=[1,3])

    if(len(rec.geo2d)==0):
        geo2d=rec.geo3d
    else:
        geo2d=rec.geo2d


    data=rec[type]
    mllines=[]
    for chan in data.channel:
        sdstr=to_string(chan)
        source=sdstr[:sdstr.find("D")]
        detector=sdstr[sdstr.find("D"):]

        srcpos=geo2d[geo2d.label==source].to_numpy()
        detpos=geo2d[geo2d.label==detector].to_numpy()
        ll,= ax[0].plot([srcpos[0,0],detpos[0,0]],[srcpos[0,1],detpos[0,1]],'k')
        ax[0].text(srcpos[0,0], srcpos[0,1], source, fontsize=12, ha='center', va='center')
        ax[0].text(detpos[0,0],detpos[0,1], detector, fontsize=12, ha='center', va='center')
        ll.set_color('k')  
        mllines.append(ll)


    mllines[0].set_color('r') 

    optodes=geo2d.to_numpy()
    s=(optodes.max()-optodes.min())/10
    ax[0].set_ylim(optodes[:,1].min()-s, optodes[:,1].max()+s)
    ax[0].set_xlim(optodes[:,0].min()-s, optodes[:,0].max()+s)
    ax[0].set_axis_off()

    selchann=to_string(data.channel[0])
    if(hasattr(data,'wavelength')):
        line0=ax[1].plot(data.time,data.sel(channel=selchann, wavelength="850"), "r-", label="850nm")
        line1=ax[1].plot(data.time,data.sel(channel=selchann, wavelength="760"), "b-", label="760nm")
        plt.legend(["850nm","760nm"])
    else:
        line0=ax[1].plot(data.time,data.sel(channel=selchann, chromo="HbO"), "r-", label="HbO")
        line1=ax[1].plot(data.time,data.sel(channel=selchann, chromo="HbR"), "b-", label="HbR")
        plt.legend(["HbO","HbR"])

    ax[1].set_title( selchann)
    ax[1].set_xlabel("time / s")
    ax[1].set_ylabel("Signal intensity / a.u.")
    ax[1].set_ylim(data.to_numpy().min(), data.to_numpy().max())
    ax[1].set_xlim(data.time.min(), data.time.max())



    def on_click(event):
        if event.inaxes is not None:
            distances = []
            for line in mllines:
                xdata, ydata = line.get_data()
                linelength = ((xdata[0] - xdata[-1])**2 + (ydata[0] - ydata[-1])**2)**0.5   
                lineseg1   = ((xdata[0] - event.xdata)**2 + (ydata[0] - event.ydata)**2)**0.5 
                lineseg2   = ((xdata[-1] - event.xdata)**2 + (ydata[-1] - event.ydata)**2)**0.5 
                distances.append(np.abs(linelength-lineseg1-lineseg2))
                line.set_color('k')  

            min_index = distances.index(min(distances))
            mllines[min_index].set_color('r')  
            selchann=to_string(data.channel[min_index])
            if(hasattr(data,'wavelength')):
                line0[0].set_ydata(data.sel(channel=selchann, wavelength="850"))
                line1[0].set_ydata(data.sel(channel=selchann, wavelength="760"))
            else:
                line0[0].set_ydata(data.sel(channel=selchann, chromo="HbO"))
                line1[0].set_ydata(data.sel(channel=selchann, chromo="HbR"))
            ax[1].set_title( selchann)

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()