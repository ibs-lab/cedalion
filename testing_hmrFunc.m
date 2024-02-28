% testing hmrR_tInc_baselineshift_Ch_Nirs

DATADIR = "/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/Data/OLD_FT/BIDS-NIRS-Tapping/";


file_path = char(strjoin([DATADIR, "sub-01/nirs/sub-01_task-tapping_nirs.snirf"],''));

snirf = SnirfLoad(file_path);

%%

[tIncCh, tInc] = hmrR_tInc_baselineshift_Ch_Nirs(snirf.data.dataTimeSeries, snirf.data.time);
%%
x = snirf.data.dataTimeSeries(:,end);
[row, col] = find(tInc == 1);
tInc_inv = abs(tInc - 1);

plot(x)
hold on 
plot(tInc_inv*max(x), 'LineWidth',3)    


