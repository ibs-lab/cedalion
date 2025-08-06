import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import cedalion.dataclasses as cdc
import cedalion.datasets
import cedalion.sigproc.quality as quality
from cedalion import units


@pytest.fixture
def rec():
    rec = cedalion.datasets.get_snirf_test_data()[0]
    rec["amp"] = rec["amp"].pint.dequantify().pint.quantify(units.V)
    return rec


def test_sci(rec):
    _, _ = quality.sci(rec["amp"], 5 * units.s, sci_thresh=0.7)


def test_psp(rec):
    _, _ = quality.psp(rec["amp"], 2 * units.s, psp_thresh=0.1)

def test_snr(rec):
    _, _ = quality.snr(rec["amp"], snr_thresh=2.0)

@pytest.mark.parametrize(
    "stat_type",
    [
        "default",
        "histogram_mode",
        "kdensity_mode",
        "parabolic_mode",
        "median",
        "mean",
        "MAD",
    ],
)
def test_gvtd(rec, stat_type):
    _, _ = quality.gvtd(rec["amp"], stat_type=stat_type)


def test_mean_amp(rec):
    amp_min = 0.5 * units.V
    amp_max = 1.0 * units.V

    _, _ = quality.mean_amp(rec["amp"], amp_range=(amp_min, amp_max))


def test_sd_dist(rec):
    # units are mm
    channel_distances = {
        "S3D3": 45.3,
        "S8D6": 47.5,
        "S8D12": 48.1,
        "S1D16": 7.1,
        "S15D23": 8.5,
    }

    metric, mask = quality.sd_dist(
        rec["amp"],
        rec.geo3d,
        sd_range=(1.5 * units.cm, 4.5 * units.cm),
    )

    dists = (
        metric.sel(channel=list(channel_distances.keys()))
        .pint.to("mm")
        .pint.dequantify()
    )

    assert_allclose(dists, list(channel_distances.values()), atol=0.1)
    for ch in channel_distances.keys():
        assert mask.sel(channel=ch).item() == quality.TAINTED


def test_id_motion(rec):
    rec["od"] = cedalion.nirs.int2od(rec["amp"])

    _ = quality.id_motion(rec["od"])


def test_id_motion_refine(rec):
    rec["od"] = cedalion.nirs.int2od(rec["amp"])

    ma_mask = quality.id_motion(rec["od"])

    for operator in ["by_channel", "all"]:
        _, _ = quality.id_motion_refine(ma_mask, operator)


def test_detect_outliers(rec):
    _ = quality.detect_outliers(rec["amp"], t_window_std=2 * units.s)


def test_detect_baselineshift(rec):
    outlier_mask = quality.detect_outliers(rec["amp"], t_window_std=2 * units.s)
    _ = quality.detect_baselineshift(rec["amp"], outlier_mask)



def test_stimulus_mask():
    t = np.arange(10)
    channel = ["S1D1", "S1D2", "S1D3"]
    source = ["S1", "S1", "S1"]
    detector = ["D1", "D2", "D3"]

    df_stim = pd.DataFrame(
        {
            "onset": [1.0, 5.0],
            "duration": [3.0, 3.0],
            "value": [1.0, 1.0],
            "trial_type": ["X", "X"],
        }
    )

    mask = cdc.build_timeseries(
        np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1], # stim 0 in channel 1 tainted
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1], # stim 1 in channel 2 tainted
        ]),
        dims=["channel", "time"],
        time=t,
        channel=channel,
        value_units="1",
        time_units="s",
        other_coords={"source": ("channel", source), "detector": ("channel", detector)},
    ).astype(bool)

    stim_mask = quality.stimulus_mask(df_stim, mask)

    assert stim_mask.dims == ("stim", "channel")
    assert stim_mask.sizes["stim"] == 2
    assert stim_mask.sizes["channel"] == 3

    assert all(stim_mask[0,:] == [True, False, True])
    assert all(stim_mask[1, :] == [True, True, False])
