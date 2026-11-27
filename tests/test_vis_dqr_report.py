"""Tests for Cedalion DQR assembly."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cedalion.vis.dqr as dqr


class FakeRecording:
    """Minimal recording object needed by generate_dqr()."""

    def __init__(
        self,
        wavelengths: list[float],
        *,
        with_stim: bool = False,
    ) -> None:
        """Create a minimal recording with amplitude data."""
        channels = [
            "S1D1",
            "S1D2",
            "S2D1",
        ]

        time = np.arange(
            5,
            dtype=float,
        )

        amp = xr.DataArray(
            np.ones(
                (
                    len(channels),
                    len(wavelengths),
                    len(time),
                )
            ),
            dims=(
                "channel",
                "wavelength",
                "time",
            ),
            coords={
                "channel": channels,
                "wavelength": wavelengths,
                "time": time,
            },
            name="amp",
        )

        self.timeseries = {
            "amp": amp,
        }

        self.geo3d = object()

        if with_stim:
            self.stim = pd.DataFrame(
                {
                    "onset": [1.0],
                    "duration": [0.5],
                    "trial_type": ["stim"],
                }
            )
        else:
            self.stim = None

    def __getitem__(
        self,
        name: str,
    ) -> xr.DataArray:
        """Return a named time series."""
        return self.timeseries[name]


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure tests never leak Matplotlib figures."""
    plt.close("all")

    yield

    plt.close("all")


def make_sidecar(
    wavelengths: list[float] | None = None,
    *,
    normalize_landmarks: bool = False,
) -> xr.Dataset:
    """Construct a minimal valid DQR sidecar."""
    if wavelengths is None:
        wavelengths = [
            760.0,
            850.0,
        ]

    channels = [
        "S1D1",
        "S1D2",
        "S2D1",
    ]

    time = np.arange(
        5,
        dtype=float,
    )

    nwavelengths = len(wavelengths)

    snr_values = np.array(
        [
            np.linspace(
                10.0,
                12.0,
                nwavelengths,
            ),
            np.linspace(
                4.0,
                6.0,
                nwavelengths,
            ),
            np.linspace(
                7.0,
                9.0,
                nwavelengths,
            ),
        ]
    )

    variance_values = np.array(
        [
            np.linspace(
                -4.0,
                -3.5,
                nwavelengths,
            ),
            np.linspace(
                -3.0,
                -2.5,
                nwavelengths,
            ),
            np.linspace(
                -2.0,
                -1.5,
                nwavelengths,
            ),
        ]
    )

    return xr.Dataset(
        {
            "gvtd_before": xr.DataArray(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                ],
                dims=("time",),
                coords={
                    "time": time,
                },
            ),
            "gvtd_before_threshold": (xr.DataArray(2.5)),
            "gvtd_after": xr.DataArray(
                [
                    0.8,
                    1.2,
                    1.8,
                    1.1,
                    0.7,
                ],
                dims=("time",),
                coords={
                    "time": time,
                },
            ),
            "gvtd_after_threshold": (xr.DataArray(1.5)),
            "amp_pruned_reason": (
                xr.DataArray(
                    [
                        0,
                        3,
                        1,
                    ],
                    dims=("channel",),
                    coords={
                        "channel": channels,
                    },
                )
            ),
            "amp_pruned_snr": (
                xr.DataArray(
                    snr_values,
                    dims=(
                        "channel",
                        "wavelength",
                    ),
                    coords={
                        "channel": channels,
                        "wavelength": wavelengths,
                    },
                )
            ),
            "amp_pruned_snr_mask": (
                xr.DataArray(
                    snr_values > 5.0,
                    dims=(
                        "channel",
                        "wavelength",
                    ),
                    coords={
                        "channel": channels,
                        "wavelength": wavelengths,
                    },
                )
            ),
            "amp_pruned_sd_dist": xr.DataArray(
                [
                    30.0,
                    35.0,
                    42.0,
                ],
                dims=("channel",),
                coords={
                    "channel": channels,
                },
            ),
            "amp_pruned_time_clean_fraction": xr.DataArray(
                [
                    0.95,
                    0.55,
                    0.80,
                ],
                dims=("channel",),
                coords={
                    "channel": channels,
                },
            ),
            "amp_pruned_mean_amp": xr.DataArray(
                np.array(
                    [
                        np.linspace(
                            0.20,
                            0.25,
                            nwavelengths,
                        ),
                        np.linspace(
                            0.10,
                            0.15,
                            nwavelengths,
                        ),
                        np.linspace(
                            0.30,
                            0.35,
                            nwavelengths,
                        ),
                    ]
                ),
                dims=(
                    "channel",
                    "wavelength",
                ),
                coords={
                    "channel": channels,
                    "wavelength": wavelengths,
                },
            ),
            "od_variance_corrected": (
                xr.DataArray(
                    variance_values,
                    dims=(
                        "channel",
                        "wavelength",
                    ),
                    coords={
                        "channel": channels,
                        "wavelength": wavelengths,
                    },
                )
            ),
        },
        attrs={
            "preprocess": json.dumps(
                {
                    "steps": [
                        {
                            "name": "amp_pruned",
                            "method": "prune",
                            "params": {
                                "snr_thresh": 5,
                            },
                        },
                    ],
                    "keep_intermediate": False,
                    "normalize_landmarks": normalize_landmarks,
                },
                sort_keys=True,
                default=str,
            ),
        },
    )


def write_sidecar(
    tmp_path,
    dataset: xr.Dataset,
):
    """Write a temporary DQR sidecar."""
    path = tmp_path / "sidecar.nc"

    dataset.to_netcdf(path)

    return path


def install_recording(
    monkeypatch,
    wavelengths: list[float],
    *,
    with_stim: bool = False,
) -> FakeRecording:
    """Replace SNIRF loading with a minimal recording."""
    rec = FakeRecording(
        wavelengths,
        with_stim=with_stim,
    )

    monkeypatch.setattr(
        dqr.cedalion.io,
        "read_snirf",
        lambda *args, **kwargs: [rec],
    )

    return rec


def add_dummy_artist(
    ax,
    label: str,
) -> None:
    """Add an empty labeled artist so legend calls remain valid."""
    ax.plot(
        [],
        [],
        label=label,
    )


def test_prune_display_metric_maps_all_categories():
    """Pruning reason codes use legacy DQR display positions."""
    reason = xr.DataArray(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        dims=("channel",),
        coords={
            "channel": [
                "c0",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
            ],
        },
    )

    actual = dqr._prune_display_metric(reason)

    expected = np.array(
        [
            0.58,
            0.08,
            0.24,
            0.40,
            0.76,
            0.92,
        ]
    )

    np.testing.assert_allclose(
        actual.values,
        expected,
    )


def test_prune_display_metric_rejects_unknown_code():
    """Unknown pruning reason codes are not silently plotted."""
    reason = xr.DataArray(
        [
            0,
            99,
        ],
        dims=("channel",),
        coords={
            "channel": [
                "c0",
                "c1",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match=("Unknown pruning reason codes"),
    ):
        dqr._prune_display_metric(reason)


def test_generate_dqr_builds_expected_panels(
    tmp_path,
    monkeypatch,
):
    """DQR assembly preserves the expected legacy semantics."""
    wavelengths = [
        760.0,
        850.0,
    ]

    sidecar = write_sidecar(
        tmp_path,
        make_sidecar(wavelengths),
    )

    install_recording(
        monkeypatch,
        wavelengths,
        with_stim=True,
    )

    gvtd_calls = []
    metric_calls = []
    histogram_calls = []
    stim_calls = []

    def fake_plot_gvtd(
        values,
        ax,
        **kwargs,
    ):
        gvtd_calls.append(kwargs)

        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    def fake_plot_channel_metric(
        ts,
        geo3d,
        metric,
        ax,
        **kwargs,
    ):
        metric_calls.append(kwargs)

    def fake_plot_histogram(
        values,
        ax,
        **kwargs,
    ):
        histogram_calls.append(kwargs)

        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    def fake_plot_stim_markers(
        ax,
        stim,
        **kwargs,
    ):
        stim_calls.append(kwargs)

    monkeypatch.setattr(
        dqr,
        "plot_gvtd",
        fake_plot_gvtd,
    )

    monkeypatch.setattr(
        dqr,
        "plot_channel_metric",
        fake_plot_channel_metric,
    )

    monkeypatch.setattr(
        dqr,
        "plot_gvtd_histogram",
        fake_plot_histogram,
    )

    monkeypatch.setattr(
        dqr,
        "plot_stim_markers",
        fake_plot_stim_markers,
    )

    output_dqr = tmp_path / "dqr.png"

    output_hist = tmp_path / "hist.png"

    output_quality = tmp_path / "quality.png"

    dqr.generate_dqr_plots(
        input_snirf=(tmp_path / "input.snirf"),
        input_sidecar=sidecar,
        output_dqr=output_dqr,
        output_gvtd_histogram=(output_hist),
        output_channel_quality=(output_quality),
        title="test",
    )

    assert output_dqr.exists()
    assert output_dqr.stat().st_size > 0

    assert output_hist.exists()
    assert output_hist.stat().st_size > 0

    assert output_quality.exists()
    assert output_quality.stat().st_size > 0

    assert len(gvtd_calls) == 2

    assert gvtd_calls[0]["color"] == "blue"

    assert gvtd_calls[0]["threshold_color"] == "blue"

    assert gvtd_calls[1]["color"] == "#ff4500"

    assert gvtd_calls[1]["threshold_color"] == "#ff4500"

    assert len(stim_calls) == 1

    assert stim_calls[0]["y"] == 1.0

    prune_calls = [
        call for call in metric_calls if call["title"].startswith("Pruned Channels")
    ]

    assert len(prune_calls) == 1

    prune_call = prune_calls[0]

    assert prune_call["vmin"] == 0

    assert prune_call["vmax"] == 1

    assert prune_call["cb_ticks_labels"] == dqr._PRUNE_TICKS

    variance_calls = [
        call for call in metric_calls if call["title"].startswith("OD Variance")
    ]

    assert len(variance_calls) == 2

    variance_limits = {
        (
            call["vmin"],
            call["vmax"],
        )
        for call in variance_calls
    }

    assert len(variance_limits) == 1

    assert {call["wavelength"] for call in variance_calls} == {
        760.0,
        850.0,
    }

    snr_calls = [call for call in metric_calls if call["title"].startswith("SNR -")]

    assert len(snr_calls) == 2

    for call in snr_calls:
        assert call["vmin"] == 0

        assert call["vmax"] == 25

        assert "pass SNR > 5" in call["title"]

    sd_calls = [
        call
        for call in metric_calls
        if call["title"].startswith("Source-Detector Distance")
    ]

    assert len(sd_calls) == 1

    clean_fraction_calls = [
        call
        for call in metric_calls
        if call["title"].startswith("Clean Time Fraction")
    ]

    assert len(clean_fraction_calls) == 1

    mean_amp_calls = [
        call
        for call in metric_calls
        if call["title"].startswith("Mean Amplitude")
    ]

    assert len(mean_amp_calls) == 2

    assert {call["wavelength"] for call in mean_amp_calls} == {
        760.0,
        850.0,
    }

    assert len(histogram_calls) == 2

    assert all(call["threshold_color"] == "red" for call in histogram_calls)

    assert plt.get_fignums() == []


def test_generate_dqr_supports_more_than_two_wavelengths(
    tmp_path,
    monkeypatch,
):
    """Report layout expands for datasets with more wavelengths."""
    wavelengths = [
        690.0,
        760.0,
        850.0,
    ]

    sidecar = write_sidecar(
        tmp_path,
        make_sidecar(wavelengths),
    )

    install_recording(
        monkeypatch,
        wavelengths,
    )

    metric_calls = []

    def fake_plot_gvtd(
        values,
        ax,
        **kwargs,
    ):
        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    def fake_plot_histogram(
        values,
        ax,
        **kwargs,
    ):
        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    monkeypatch.setattr(
        dqr,
        "plot_gvtd",
        fake_plot_gvtd,
    )

    monkeypatch.setattr(
        dqr,
        "plot_gvtd_histogram",
        fake_plot_histogram,
    )

    monkeypatch.setattr(
        dqr,
        "plot_channel_metric",
        lambda *args, **kwargs: metric_calls.append(kwargs),
    )

    output_dqr = tmp_path / "dqr.png"

    output_hist = tmp_path / "hist.png"

    dqr.generate_dqr_plots(
        input_snirf=(tmp_path / "input.snirf"),
        input_sidecar=sidecar,
        output_dqr=output_dqr,
        output_gvtd_histogram=(output_hist),
    )

    # One pruning panel,
    # three variance panels,
    # three SNR panels.
    assert len(metric_calls) == 7

    assert output_dqr.exists()
    assert output_hist.exists()

    assert plt.get_fignums() == []


def test_generate_dqr_reports_missing_sidecar_variable(
    tmp_path,
    monkeypatch,
):
    """Missing report checkpoints produce a clear error."""
    wavelengths = [
        760.0,
        850.0,
    ]

    dataset = make_sidecar(wavelengths).drop_vars("od_variance_corrected")

    sidecar = write_sidecar(
        tmp_path,
        dataset,
    )

    install_recording(
        monkeypatch,
        wavelengths,
    )

    with pytest.raises(
        ValueError,
        match=("od_variance_corrected"),
    ):
        dqr.generate_dqr_plots(
            input_snirf=(tmp_path / "input.snirf"),
            input_sidecar=sidecar,
            output_dqr=(tmp_path / "dqr.png"),
            output_gvtd_histogram=(tmp_path / "hist.png"),
        )


def test_generate_dqr_normalizes_landmarks_from_sidecar(
    tmp_path,
    monkeypatch,
):
    """DQR applies landmark normalization recorded in sidecar provenance."""
    wavelengths = [
        760.0,
        850.0,
    ]

    sidecar = write_sidecar(
        tmp_path,
        make_sidecar(
            wavelengths,
            normalize_landmarks=True,
        ),
    )

    rec = install_recording(
        monkeypatch,
        wavelengths,
    )

    original_geo3d = rec.geo3d
    normalized_geo3d = object()
    normalization_calls = []

    def fake_normalize_landmarks_labels(geo3d):
        normalization_calls.append(geo3d)
        return normalized_geo3d

    def fake_plot_gvtd(
        values,
        ax,
        **kwargs,
    ):
        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    def fake_plot_histogram(
        values,
        ax,
        **kwargs,
    ):
        add_dummy_artist(
            ax,
            kwargs.get(
                "label",
                "GVTD",
            ),
        )

    monkeypatch.setattr(
        dqr,
        "normalize_landmarks_labels",
        fake_normalize_landmarks_labels,
    )

    monkeypatch.setattr(
        dqr,
        "plot_gvtd",
        fake_plot_gvtd,
    )

    monkeypatch.setattr(
        dqr,
        "plot_gvtd_histogram",
        fake_plot_histogram,
    )

    monkeypatch.setattr(
        dqr,
        "plot_channel_metric",
        lambda *args, **kwargs: None,
    )

    dqr.generate_dqr_plots(
        input_snirf=(tmp_path / "input.snirf"),
        input_sidecar=sidecar,
        output_dqr=(tmp_path / "dqr.png"),
        output_gvtd_histogram=(tmp_path / "hist.png"),
    )

    assert normalization_calls == [original_geo3d]
    assert rec.geo3d is normalized_geo3d
