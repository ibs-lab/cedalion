from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
from cedalion.typing import LabeledPointCloud, NDTimeSeries

import copy
import statsmodels
import statsmodels.regression
import statsmodels.robust
import statsmodels.regression.recursive_ls
import statsmodels.regression.linear_model
import statsmodels.robust.robust_linear_model
import statsmodels.stats.contrast
import cedalion.math.stats_helpers

models=[statsmodels.regression.recursive_ls.RecursiveLSResultsWrapper,
            statsmodels.regression.linear_model.RegressionResultsWrapper,
            statsmodels.robust.robust_linear_model.RLMResultsWrapper]

@dataclass
class Statistics:
    """Main container for statistical results.

    The `Statistics` class holds stats adjunct objects.
    
    Attributes:
        models (Pandas DataFrame): colection of statsmodel results
        coloring_matrix: xr.DataArray
        masks (OrderedDict[str, xr.DataArray]): A dictionary of masks. The keys are the
            names of the masks.
        geo3d (LabeledPointCloud): A labeled point cloud representing the 3D geometry of
            the recording.
        geo2d (LabeledPointCloud): A labeled point cloud representing the 2D geometry of
            the recording.
        aux_obj (OrderedDict[str, Any]): A dictionary of auxiliary objects.
        head_model (Optional[Any]): A head model object.
        meta_data (OrderedDict[str, Any]): A dictionary of meta data.
    """

    description: str =field(default_factory=str)
    models: pd.DataFrame= field(default_factory=pd.DataFrame)
    coloring_matrix: xr.DataArray = field(default_factory=xr.DataArray)
    masks: OrderedDict[str, xr.DataArray] = field(default_factory=OrderedDict)
    geo3d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    geo2d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    head_model: Optional[Any] = None
    meta_data: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    # these are the loaded ML from the snirf file.
    _measurement_lists: OrderedDict[str, pd.DataFrame] = field(
        default_factory=OrderedDict
    )

    def __repr__(self):
        """Return a string representation of the Recording object."""
        return (
            f"<Statistics | "
            f"{self.description}, "
            f" masks: {list(self.masks.keys())}, "
            )

    def get_mask(self, key: Optional[str] = None) -> xr.DataArray:
        """Get a mask by key.

        Args:
            key (Optional[str]): The key of the mask to retrieve. If None, the last
                mask is returned.

        Returns:
            xr.DataArray: The requested mask.
        """
        if not self.masks:
            raise ValueError("masks dict is empty.")

        if key:
            return self.masks[key]
        else:
            last_key = list(self.masks.keys())[-1]

            return self.masks[last_key]

    def set_mask(self, key: str, value: xr.DataArray, overwrite: bool = False):
        """Set a mask.

        Args:
            key (str): The key of the mask to set.
            value (xr.DataArray): The mask to set.
            overwrite (bool): Whether to overwrite an existing mask with the same key.
                Defaults to False.
        """
        if (overwrite is False) and (key in self.masks):
            raise ValueError(f"a mask with key '{key}' already exists!")

        self.masks[key] = value
    



    @property
    def betas(self):
        params=[]
        if(self.models.models[0].__class__ in models):
            for m in self.models.models:
                params.append(m.params)
        elif(self.models.models[0].__class__ ==statsmodels.stats.contrast.ContrastResults):
            for m in self.models.models:
                coef=pd.DataFrame(m.effect)
                coef.columns=m.c_names
                params.append(coef.to_dict(orient='records')[0])
        else:
            NotImplementedError
        
        theta=pd.DataFrame(params)
        theta.insert(0,'channel',self.models.channel)
        theta.insert(1,'type',self.models.type)
        
        return theta

    @property
    def tvalue(self):
        params=[]
        dof=[]
        if(self.models.models[0].__class__ in models):
            for m in self.models.models:
                params.append(m.tvalues)
                dof.append(m.df_resid)
        elif(self.models.models[0].__class__ ==statsmodels.stats.contrast.ContrastResults):
            for m in self.models.models:
                coef=pd.DataFrame(m.tvalue)
                coef.columns=m.c_names
                params.append(coef.to_dict(orient='records')[0])
                
                dof.append(m.df_denom)
        else:
            NotImplementedError   

        ttest=pd.DataFrame(params)
        ttest.insert(0,'channel',self.models.channel)
        ttest.insert(1,'type',self.models.type)
        ttest.insert(len(params[0])+2,'dof',dof)
        
        return ttest
    
    @property
    def pvalue(self):
        params=[]
        dof=[]

        if(self.models.models[0].__class__ in models):
            for m in self.models.models:
                params.append(m.pvalues)
                dof.append(m.df_resid)
            pval=pd.DataFrame(params)    
        
        elif(self.models.models[0].__class__ ==statsmodels.stats.contrast.ContrastResults):
            for m in self.models.models:
                coef=pd.DataFrame(m.pvalue+[0]) # There is a bug in statsmodels that makes this a 0 size array
                coef.columns=m.c_names
                params.append(coef)
            
                dof.append(m.df_denom)

            pval=pd.DataFrame(np.array(params)[:,:,0])

        else:
            NotImplementedError 
        
        pval.insert(0,'channel',self.models.channel)
        pval.insert(1,'type',self.models.type)
        pval.insert(len(params[0])+2,'dof',dof)
        
        return pval
    @property
    def stderr(self):
        params=[]

        if(self.models.models[0].__class__ in models):
            for m in self.models.models:
                params.append(m.tvalues/m.params)
        elif(self.models.models[0].__class__ ==statsmodels.stats.contrast.ContrastResults):
            for m in self.models.models:
                coef=pd.DataFrame(m.sd)
                coef.columns=m.c_names
                params.append(coef.to_dict(orient='records')[0])
                
        else:
            NotImplementedError 
                
        stde=pd.DataFrame(params)
        stde.insert(0,'channel',self.models.channel)
        stde.insert(1,'type',self.models.type)
        
        return stde

    def ttest(self,cont):
        contstats=copy.deepcopy(self)
        if(self.models.models[0].__class__ in models):
            params=[]
            for m in self.models.models:
                params.append(m.t_test(cont))

            contstats.models.models=params
            contstats.description=f"T-test result from {cont}"
            return contstats
        else:
            NotImplementedError

    @property
    def table(self):
        tblB=self.betas
        tblT=self.tvalue
        tblP=self.pvalue
        
        conds=tblB.columns[2:]
        tbl=pd.DataFrame()
        for cond in conds:
            tmp=pd.concat([tblB[['channel','type']],tblB[cond],tblT[cond],tblP[cond],tblP['dof']],ignore_index=True,axis=1)
            tmp.insert(2,'cond',cond)
            tbl=pd.concat([tbl,tmp],ignore_index=True,axis=0)

        tbl.columns=['channel','type','cond','beta','tstat','pval','dof']
        qval = cedalion.math.stats_helpers.BenjaminiHochberg(tbl['pval'].to_numpy())
        tbl.insert(6,'qval',qval)

        return tbl
    @property
    def condnames(self):
        tblB=self.betas
        return tblB.columns[2:]
    
    @property
    def results(self):
        tbl = self.table
        if('HbO' in tbl.type.unique()):
            return xr.DataArray(tbl.beta.to_numpy().reshape(len(tbl.cond.unique()),len(tbl.channel.unique()),len(tbl.type.unique())),
                dims=['regressor','channel','chromo'],
                coords={
                    "channel": tbl.channel.unique(),
                    "regressor": tbl.cond.unique(),
                    "chromo": tbl.type.unique()},)
        else:
            return xr.DataArray(tbl.beta.to_numpy().reshape(len(tbl.cond.unique()),len(tbl.channel.unique()),len(tbl.type.unique())),
                dims=['regressor','channel','wavelength'],
                coords={
                    "channel": tbl.channel.unique(),
                    "regressor": tbl.cond.unique(),
                    "wavelength": tbl.type.unique()},)