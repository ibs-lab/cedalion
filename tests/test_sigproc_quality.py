import pytest
from numpy.testing import assert_allclose
import cedalion.sigproc.quality as quality
import cedalion.datasets
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

#def test_psp_alternative(rec):
#    _, _ = quality._psp_alternative(rec["amp"], 2 * units.s, psp_thresh=0.1)


def test_snr(rec):
    _, _ = quality.snr(rec["amp"], snr_thresh=2.0)

@pytest.mark.parametrize(
    "stat_type",
    [
        "default",
        "histogram_mode",
        "Kdensity_mode",
        "parabolic_mode",
        "median",
        "mean",
        "MAD",
    ],
)
def test_gvtd(rec, stat_type):
    _ = quality.gvtd(rec["amp"], statType=stat_type)


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
