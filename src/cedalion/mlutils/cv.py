import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from cedalion.models.glm.design_matrix import DesignMatrix
import cedalion
import cedalion.typing as cdt
from typing import Generator


def create_cv_splits(
    df_stim: pd.DataFrame, n_splits: int
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Split stimulus events into train and test sets for cross-validation.

    Args:
        df_stim: Stimulus events, sorted by onset times with ordered index.
        n_splits: number of folds

    Yields:
        For each fold, the stimulis data frame split into train and test set.

        The test trials are consecutive and not randomized.
    """
    assert (np.diff(df_stim["onset"]) > 0).all()
    assert (np.diff(df_stim.index) > 0).all()

    y = df_stim["trial_type"]
    skf = StratifiedKFold(n_splits=n_splits)

    for train_indices, test_indices in skf.split(np.zeros(len(y)), y):
        train_stim = df_stim.iloc[train_indices]
        test_stim = df_stim.iloc[test_indices]
        yield train_stim, test_stim


def mask_design_matrix(
    dms: DesignMatrix,
    df_stim_test: pd.DataFrame,
    before: cdt.QTime = 5 * cedalion.units.s,
    after: cdt.QTime = 20 * cedalion.units.s,
) -> DesignMatrix:
    """Mask a segment of the design matrix by setting it to zero.

    When using GLM parameters as features, the fit must not have access to the test
    trials. This function zeros out a contiguous segment of the design matrix, ensuring
    that the model cannot explain the time course in the masked segment for any choice
    of parameters. The segment extends from the earliest to the latest trial in
    `df_stim_test`, padded by additional time specified by the `before` and `after`
    parameters. Because the masked segment is continuous, the train-test split must be
    chosen such that the test trials are consecutive.

    Args:
        dms: The design matrix to mask
        df_stim_test: test set of stimulus events.
        before : time to pad before the earlist test trial
        after: time to pad after the latest test trial

    Returns:
        A copy of the design matrix with the masked segment set to zero.
    """
    if len(dms.channel_wise) > 0:
        raise NotImplementedError(
            "masking of channel-wise regressors is not implemented, yet."
        )

    # Identify the earliest and latest test stimulus onset times
    t_min = df_stim_test["onset"].min() * cedalion.units.s - before
    t_max = df_stim_test["onset"].max() * cedalion.units.s + after

    if "units" in dms.common.time.attrs:
        units = dms.common.time.attrs["units"]
    else:
        units = cedalion.units.s

    t_min = t_min.to(units).magnitude
    t_max = t_max.to(units).magnitude

    dms = dms.copy()
    dms.common.loc[{"time": slice(t_min, t_max)}] = 0.0

    return dms
