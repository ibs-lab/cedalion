import numpy as np
from cedalion.sigproc.bvp_wav_ana_v12 import (
                interpft,
                peakseek,
                bvp_single_ch,
                wct,
                extract_bvp,
                extract_waveforms,
                remove_artifact_waveforms,
                classify_waveforms,)
from cedalion.dataclasses import build_timeseries
from cedalion.dataclasses.bvp_container import BVP_Container


def test_interpft():
    """Tests Fourier interpolation by upsampling and downsampling.

    A periodic multiharmonic signal with a known analytical representation
    is first upsampled from 50 to 100 samples. The result is compared with
    the signal evaluated directly at the finer sampling points.

    The upsampled signal is subsequently downsampled to verify that the
    original signal can be reconstructed.
    """

    # --- Define coefficients of the multiharmonic signal ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    # --- Create original and upsampled time vectors ---
    n_samples = 50
    n_upsampled = 100
    t_samples = np.arange(n_samples) / n_samples
    t_upsampled = np.arange(n_upsampled) / n_upsampled

    k = np.arange(1, 9)[:, None]

    angle_samples = 2 * np.pi * k * t_samples
    angle_upsampled = 2 * np.pi * k * t_upsampled

    # --- Generate the original multiharmonic signal ---
    x = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    # --- Upsample the signal using Fourier interpolation ---
    upsampled = interpft(x, n_upsampled)

    # Calculate the expected signal directly at the finer sampling points.
    expected_upsampled = np.sum(
        a[:, None] * np.sin(angle_upsampled)
        + b[:, None] * (np.cos(angle_upsampled) - 1),
        axis=0)

    # --- Check the upsampled signal ---
    assert upsampled.shape == (n_upsampled,)
    assert np.isrealobj(upsampled)
    np.testing.assert_allclose(
        upsampled,
        expected_upsampled,
        atol=1e-12)

    # --- Downsample to the original number of samples ---
    downsampled = interpft(upsampled, n_samples)

    # --- Check reconstruction of the original signal ---
    assert downsampled.shape == x.shape
    np.testing.assert_allclose(
        downsampled,
        x,
        atol=1e-12)

def test_peakseek():
    """Tests peak detection using simple and multiharmonic signals.

    The first signal contains clearly defined peaks and verifies the minimum
    peak distance and minimum peak height parameters.

    The second signal represents a realistic multiharmonic pulse waveform
    and verifies peak locations and amplitudes for a more complex signal.
    """

    # --- Test a simple signal with known peak locations ---
    simple_signal = np.array([
        10.0, 0.0, 2.0, 0.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 0.0, 10.0,
    ])

    locations, peaks = peakseek(
        simple_signal,
        minpeakdist=3,
        minpeakh=1.5,
    )

    # Peaks at the signal boundaries are not considered local maxima.
    np.testing.assert_array_equal(
        locations,
        np.array([4, 9]))
    np.testing.assert_array_equal(
        peaks,
        np.array([3.0, 4.0]))

    # --- Define coefficients of the multiharmonic signal ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    # --- Generate a multiharmonic pulse waveform ---
    n_samples = 80
    t_samples = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angle_samples = 2 * np.pi * k * t_samples

    multiharmonic_signal = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    # --- Detect peaks in the multiharmonic waveform ---
    locations, peaks = peakseek(multiharmonic_signal)

    # --- Check detected peak locations and amplitudes ---
    np.testing.assert_array_equal(
        locations,
        np.array([23, 35]))
    np.testing.assert_allclose(
        peaks,
        np.array([0.9698297535333608, 0.9508981042738599]),
        atol=1e-12)

def test_bvp_single_ch():
    """Tests BVP extraction from a single-channel concentration signal.

    Two test cases are evaluated:

    1. A linear concentration change containing only a low-frequency trend.
       The extracted BVP should therefore be zero.
    2. A physiological multiharmonic pulse superimposed on a slowly varying
       trend. The extracted trend and BVP are compared with their known
       analytical components.
    """

    # --- Test a signal containing only a linear trend ---
    concentration = np.linspace(5.0, 15.0, 100)
    original_concentration = concentration.copy()
    fs = 10.0
    fs_new = 20.0

    bvp, resampled, trend = bvp_single_ch(
        concentration,
        fs,
        fs_new)

    expected_length = 200
    expected_resampled = np.linspace(
        0.0,
        10.0,
        expected_length)

    # --- Check output dimensions and input immutability ---
    assert bvp.shape == (expected_length,)
    assert resampled.shape == (expected_length,)
    assert trend.shape == (expected_length,)
    np.testing.assert_array_equal(
        concentration,
        original_concentration)

    # --- Check separation of the linear trend ---
    np.testing.assert_allclose(
        resampled,
        expected_resampled,
        atol=1e-12)
    np.testing.assert_allclose(
        trend,
        expected_resampled,
        atol=1e-10)
    np.testing.assert_allclose(
        bvp,
        np.zeros(expected_length),
        atol=1e-10)
    np.testing.assert_allclose(
        bvp,
        resampled - trend,
        atol=1e-12)

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Define sampling and signal parameters ---
    fs = 20.0
    fs_new = 50.0
    duration = 30.0
    pulse_frequency = 1.0
    n_samples = int(fs * duration)
    n_resampled = int((fs_new / fs) * n_samples)

    time = np.linspace(0.0, duration, n_samples)
    time_resampled = np.linspace(
        0.0,
        duration,
        n_resampled)

    k = np.arange(1, 9)[:, None]

    def make_pulse(actual_time):
        """Generate the physiological multiharmonic pulse waveform."""
        angles = 2 * np.pi * k * pulse_frequency * actual_time

        pulse = np.sum(
            a[:, None] * np.sin(angles)
            + b[:, None] * (np.cos(angles) - 1),
            axis=0)

        # The mean of the multiharmonic expression is -sum(b).
        return pulse + np.sum(b)

    def make_low_frequency_trend(actual_time):
        """Generate the known low-frequency concentration trend."""
        return (
            1.2
            + 0.25 * np.sin(2 * np.pi * 0.1 * actual_time)
            - 0.01 * actual_time)

    # --- Combine the pulse and low-frequency trend ---
    pulse = make_pulse(time)
    low_frequency_trend = make_low_frequency_trend(time)
    concentration = low_frequency_trend + pulse

    # --- Extract the BVP and trend components ---
    bvp, resampled, trend = bvp_single_ch(
        concentration,
        fs,
        fs_new)

    # --- Calculate the expected resampled components ---
    expected_pulse = make_pulse(time_resampled)
    expected_trend = (
        make_low_frequency_trend(time_resampled)
        - concentration[0]
    )
    expected_resampled = expected_trend + expected_pulse

    # --- Check output dimensions and signal reconstruction ---
    assert bvp.shape == (n_resampled,)
    assert resampled.shape == (n_resampled,)
    assert trend.shape == (n_resampled,)
    np.testing.assert_allclose(
        resampled,
        expected_resampled,
        atol=0.05)
    np.testing.assert_allclose(
        bvp,
        resampled - trend,
        atol=1e-12)

    # --- Exclude the unstable LOESS boundary regions ---
    edge = int(3.5 * fs_new)
    interior = slice(edge, -edge)

    # --- Calculate trend and pulse extraction quality ---
    trend_error = (
        trend[interior]
        - expected_trend[interior]
    )
    trend_rmse = np.sqrt(
        np.mean(trend_error**2)
    )
    trend_correlation = np.corrcoef(
        trend[interior],
        expected_trend[interior],
    )[0, 1]
    bvp_correlation = np.corrcoef(
        bvp[interior],
        expected_pulse[interior],
    )[0, 1]

    # --- Check trend and BVP extraction accuracy ---
    assert trend_rmse < 0.08
    assert trend_correlation > 0.98
    assert bvp_correlation > 0.95

def test_wct():
    """Tests wavelet coherence using two sinusoidal signals with a known
    phase shift.

    The signals have identical frequencies and amplitudes. Therefore, their
    coherence should be close to one at the signal frequency. The measured
    phase should correspond to the known phase shift.
    """  # noqa: D205

    # --- Create sinusoidal test signals ---
    fs = 50.0
    duration = 30.0
    signal_frequency = 1.25
    phase_shift = np.pi / 3

    time = np.arange(int(fs * duration)) / fs

    signal_1 = np.sin(
        2 * np.pi * signal_frequency * time
    )
    signal_2 = np.sin(
        2 * np.pi * signal_frequency * time + phase_shift
    )

    # --- Calculate wavelet coherence ---
    (
        cross_wavelet,
        power_1,
        power_2,
        coherence,
        phase,
        coi,
        frequencies,
        significance,
    ) = wct(
        signal_1,
        signal_2,
        dt=1.0 / fs,
        sig=False,
    )

    # --- Check output dimensions ---
    expected_shape = (len(frequencies), len(time))

    assert cross_wavelet.shape == expected_shape
    assert power_1.shape == expected_shape
    assert power_2.shape == expected_shape
    assert coherence.shape == expected_shape
    assert phase.shape == expected_shape
    assert coi.shape == time.shape
    np.testing.assert_array_equal(significance, np.array([0]))

    # --- Select scale closest to the signal frequency ---
    frequency_idx = np.argmin(
        np.abs(frequencies - signal_frequency)
    )

    # --- Exclude values outside the cone of influence ---
    signal_period = 1.0 / frequencies[frequency_idx]
    valid_idx = signal_period <= coi

    assert np.count_nonzero(valid_idx) > len(time) // 2

    # --- Check wavelet coherence ---
    mean_coherence = np.mean(
        coherence[frequency_idx, valid_idx]
    )

    assert mean_coherence > 0.95

    # --- Calculate circular mean phase ---
    mean_phase = np.angle(
        np.mean(
            np.exp(1j * phase[frequency_idx, valid_idx])
        )
    )

    # W1 * conj(W2) produces the negative phase difference.
    expected_phase = -phase_shift
    phase_error = np.angle(
        np.exp(1j * (mean_phase - expected_phase))
    )

    # --- Check phase difference ---
    assert abs(phase_error) < 0.15

def test_extract_bvp():
    """Tests multi-channel BVP extraction from an HbO time series.

    The test uses the linear trend and physiological pulse signals from
    test_bvp_single_ch. It verifies the output structure, channel metadata,
    physical units, and the channel-wise results.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Define sampling and signal parameters ---
    fs = 20.0
    fs_new = 50.0
    duration = 30.0
    pulse_frequency = 1.0
    n_samples = int(fs * duration)

    time = np.arange(n_samples) / fs
    k = np.arange(1, 9)[:, None]

    # --- Generate the physiological pulse waveform ---
    angles = 2 * np.pi * k * pulse_frequency * time
    pulse = np.sum(
        a[:, None] * np.sin(angles)
        + b[:, None] * (np.cos(angles) - 1),
        axis=0)
    pulse = pulse + np.sum(b)

    # --- Generate the concentration signals ---
    linear_concentration = np.linspace(
        5.0,
        15.0,
        n_samples)

    low_frequency_trend = (
        1.2
        + 0.25 * np.sin(2 * np.pi * 0.1 * time)
        - 0.01 * time)
    pulse_concentration = low_frequency_trend + pulse

    concentration = np.vstack([
        linear_concentration,
        pulse_concentration,
    ])

    # --- Build a two-channel HbO concentration time series ---
    channels = ["S1D1", "S2D2"]

    hbo_conc_ts = build_timeseries(
        concentration,
        ["channel", "time"],
        time,
        channels,
        "uM",
        "s",
        {
            "source": ("channel", ["S1", "S2"]),
            "detector": ("channel", ["D1", "D2"]),
        },
    )

    # --- Extract the BVP without opening the confirmation dialog ---
    result = extract_bvp(
        hbo_conc_ts,
        fs_new=fs_new,
        request=False)

    # --- Check dimensions, coordinates, metadata, and units ---
    assert result.dims == ("channel", "compound", "time")
    assert result.sizes["time"] == int((fs_new / fs) * n_samples)
    assert result.pint.units == hbo_conc_ts.pint.units

    np.testing.assert_array_equal(
        result.channel.values,
        channels)
    np.testing.assert_array_equal(
        result.compound.values,
        ["bvp_ts", "hbo_conc_ts_50hz", "low_freq_trend"])
    np.testing.assert_array_equal(
        result.source.values,
        ["S1", "S2"])
    np.testing.assert_array_equal(
        result.detector.values,
        ["D1", "D2"])

    # --- Check each channel against the single-channel implementation ---
    result_values = result.pint.dequantify()

    for channel, channel_concentration in zip(
            channels,
            concentration):

        expected_bvp, expected_resampled, expected_trend = (
            bvp_single_ch(
                channel_concentration,
                fs,
                fs_new)
        )

        np.testing.assert_allclose(
            result_values.sel(
                channel=channel,
                compound="bvp_ts"),
            expected_bvp,
            atol=1e-12)
        np.testing.assert_allclose(
            result_values.sel(
                channel=channel,
                compound="hbo_conc_ts_50hz"),
            expected_resampled,
            atol=1e-12)
        np.testing.assert_allclose(
            result_values.sel(
                channel=channel,
                compound="low_freq_trend"),
            expected_trend,
            atol=1e-12)

def test_extract_waveforms_gerneral():
    """Tests extraction and normalization of individual BVP waveforms.

    A periodic physiological pulse waveform from test_bvp_single_ch is
    provided in two channels with different amplitudes. The detected extrema,
    extracted waveform segments, and normalized waveform matrices are checked.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Generate thirty identical physiological pulse cycles ---
    fs = 50.0
    duration = 30.0
    pulse_frequency = 1.0
    n_samples = int(fs * duration)

    time = np.arange(n_samples) / fs
    k = np.arange(1, 9)[:, None]
    angles = 2 * np.pi * k * pulse_frequency * time

    pulse = np.sum(
        a[:, None] * np.sin(angles)
        + b[:, None] * (np.cos(angles) - 1),
        axis=0)

    # The mean of the multiharmonic expression is -sum(b).
    pulse = pulse + np.sum(b)

    # --- Create two channels with different pulse amplitudes ---
    channels = ["S1D1", "S2D2"]
    channel_scales = [1.0, 0.5]

    bvp_data = np.vstack([
        channel_scales[0] * pulse,
        channel_scales[1] * pulse,
    ])

    bvp_ts = build_timeseries(
        bvp_data,
        ["channel", "time"],
        time,
        channels,
        "uM",
        "s",
    )

    # --- Extract and normalize the individual pulse waveforms ---
    wav_storage_user, wav_storage_details = extract_waveforms(
        bvp_ts)

    # --- Define analytically known extrema ---
    expected_min_idx = np.arange(
        int(fs),
        n_samples,
        int(fs))
    expected_max_idx = expected_min_idx[:-1] + 14
    expected_waveform_count = len(expected_min_idx) - 1

    # --- Check channel-wise output structure ---
    assert set(wav_storage_user) == set(channels)
    assert set(wav_storage_details) == set(channels)

    for channel, scale in zip(channels, channel_scales):
        user_results = wav_storage_user[channel]
        details = wav_storage_details[channel]

        assert set(user_results) == {
            "bvp_max_value",
            "bvp_max_idx",
            "bvp_min_value",
            "bvp_min_idx",
        }
        assert set(details) == {
            "list_wav_raw_and_y_normal",
            "nparray_wav_xy_normal_all",
            "nparray_wav_xy_normal_zscore_all",
        }

        # --- Check detected diastolic minima and systolic maxima ---
        np.testing.assert_array_equal(
            user_results["bvp_min_idx"],
            expected_min_idx)
        np.testing.assert_array_equal(
            user_results["bvp_max_idx"],
            expected_max_idx)
        np.testing.assert_allclose(
            user_results["bvp_min_value"],
            scale * pulse[expected_min_idx],
            atol=1e-12)
        np.testing.assert_allclose(
            user_results["bvp_max_value"],
            scale * pulse[expected_max_idx],
            atol=1e-12)

        # --- Check extracted raw and detrended waveforms ---
        extracted_waveforms = (
            details["list_wav_raw_and_y_normal"]
        )

        assert len(extracted_waveforms) == expected_waveform_count

        for waveform_idx, waveform in enumerate(
                extracted_waveforms):

            start = expected_min_idx[waveform_idx]
            stop = expected_min_idx[waveform_idx + 1]

            np.testing.assert_allclose(
                waveform["wav_raw"],
                scale * pulse[start:stop],
                atol=1e-12)
            np.testing.assert_allclose(
                waveform["wav_y_normal"][[0, -1]],
                np.zeros(2),
                atol=1e-12)
            np.testing.assert_allclose(
                waveform["wav_time_s"][[0, -1]],
                np.array([0.0, 1.0]),
                atol=1e-12)

        # --- Check normalized waveform matrices ---
        xy_normal = details[
            "nparray_wav_xy_normal_all"
        ]
        zscore_normal = details[
            "nparray_wav_xy_normal_zscore_all"
        ]

        assert xy_normal.shape == (
            100,
            expected_waveform_count)
        assert zscore_normal.shape == (
            100,
            expected_waveform_count)
        assert np.all(np.isfinite(xy_normal))
        assert np.all(np.isfinite(zscore_normal))

        # All input cycles are identical and must remain identical.
        np.testing.assert_allclose(
            xy_normal,
            np.repeat(
                xy_normal[:, :1],
                expected_waveform_count,
                axis=1),
            atol=1e-12)
        np.testing.assert_allclose(
            zscore_normal,
            np.repeat(
                zscore_normal[:, :1],
                expected_waveform_count,
                axis=1),
            atol=1e-12)

        # The final normalization scales the mean maximum to one.
        np.testing.assert_allclose(
            np.max(np.mean(zscore_normal, axis=1)),
            1.0,
            atol=1e-12)

    # Amplitude normalization must remove the channel scaling difference.
    np.testing.assert_allclose(
        wav_storage_details["S1D1"][
            "nparray_wav_xy_normal_zscore_all"
        ],
        wav_storage_details["S2D2"][
            "nparray_wav_xy_normal_zscore_all"
        ],
        atol=1e-12)

def test_extract_waveforms_filterlogic():
    """Tests filtering of non-physiological BVP waveforms.

    The signal contains a shallow minimum following an unusually short
    waveform, a dicrotic notch falsely detected as a diastolic minimum,
    and a transient baseline artifact that suppresses a true minimum
    and therefore creates an abnormally long waveform.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    fs = 50.0
    k = np.arange(1, 9)[:, None]

    # Most waveforms are 50 samples long.
    #
    # Filter cases:
    #   100 -> 126: unusually short waveform
    #   256:        dicrotic notch detected as a shallow minimum
    #   391 -> 491: unusually long waveform caused by an artifact
    waveform_boundaries = np.array([
        0, 50, 100, 126, 171, 221, 291, 341, 391, 441,
        491, 541, 591, 641, 691,
    ])

    # --- Define the amplitude of each physiological minimum ---
    normal_minimum = np.sum(b)

    minimum_values = np.full(
        waveform_boundaries.size,
        normal_minimum,
    )

    # The shallow minimum at 126 belongs to the unusually
    # short waveform between samples 100 and 126.
    minimum_values[
        np.where(waveform_boundaries == 126)[0][0]
    ] = -0.30

    # --- Generate the physiological waveform segments ---
    bvp_signal = np.empty(
        waveform_boundaries[-1] + 1
    )

    for waveform_idx, (start, stop) in enumerate(zip(
            waveform_boundaries[:-1],
            waveform_boundaries[1:])):

        waveform_length = stop - start

        phase = np.linspace(
            0.0,
            1.0,
            waveform_length + 1,
        )

        angles = 2 * np.pi * k * phase

        physiological_waveform = np.sum(
            a[:, None] * np.sin(angles)
            + b[:, None] * (np.cos(angles) - 1),
            axis=0,
        )

        minimum_trend = np.linspace(
            minimum_values[waveform_idx],
            minimum_values[waveform_idx + 1],
            waveform_length + 1,
        )

        waveform = (
            physiological_waveform
            + minimum_trend
        )

        # The final sample is also the first sample of the
        # following waveform and is therefore excluded here.
        bvp_signal[start:stop] = waveform[:-1]

    bvp_signal[waveform_boundaries[-1]] = (
        minimum_values[-1]
    )

    sample_idx = np.arange(bvp_signal.size)

    # --- Add a pronounced dicrotic notch ---
    # The notch is located inside the waveform from 221 to 291.
    # Its value is less than 10 % of the normal minimum depth.
    notch_idx = 256
    notch_value = -0.05
    notch_width = 1.5

    notch_depth = (
        bvp_signal[notch_idx]
        - notch_value
    )

    bvp_signal -= notch_depth * np.exp(
        -0.5
        * ((sample_idx - notch_idx) / notch_width) ** 2
    )

    # --- Add a transient baseline artifact ---
    # Without the artifact, sample 441 would be the physiological
    # minimum separating the waveforms 391-441 and 441-491.
    # The artifact raises this minimum above zero, causing peakseek
    # to combine both pulses into one long waveform.
    artifact_center_idx = 441
    artifact_center_value = 0.05
    artifact_width = 20.0

    artifact_amplitude = (
        artifact_center_value
        - bvp_signal[artifact_center_idx]
    )

    bvp_signal += artifact_amplitude * np.exp(
        -0.5
        * (
            (sample_idx - artifact_center_idx)
            / artifact_width
        ) ** 2
    )

    # --- Build the BVP time series ---
    time = np.arange(bvp_signal.size) / fs

    bvp_ts = build_timeseries(
        bvp_signal[None, :],
        ["channel", "time"],
        time,
        ["S1D1"],
        "uM",
        "s",
    )

    # --- Verify the unfiltered minima in the test signal ---
    detected_minima_idx, detected_minima_value = peakseek(
        -bvp_signal,
        minpeakdist=int(fs / 2),
        minpeakh=0,
    )

    expected_detected_minima_idx = np.array([
        50, 100, 126, 171, 221, 256, 291,
        341, 391, 491, 541, 591, 641,
    ])

    np.testing.assert_array_equal(
        detected_minima_idx,
        expected_detected_minima_idx,
    )

    np.testing.assert_allclose(
        -detected_minima_value,
        bvp_signal[expected_detected_minima_idx],
        atol=1e-12,
    )

    # The dicrotic notch must initially be detected as a minimum.
    assert notch_idx in detected_minima_idx

    # The artifact must suppress the physiological minimum at 441.
    assert artifact_center_idx not in detected_minima_idx

    np.testing.assert_allclose(
        bvp_signal[notch_idx],
        notch_value,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        bvp_signal[artifact_center_idx],
        artifact_center_value,
        atol=1e-12,
    )

    # --- Extract and filter the individual waveforms ---
    wav_storage_user, wav_storage_details = (
        extract_waveforms(bvp_ts)
    )

    user_results = wav_storage_user["S1D1"]
    details = wav_storage_details["S1D1"]

    # Index 126 is removed because it belongs to an unusually
    # short waveform. Index 256 is removed because the dicrotic
    # notch is too shallow to represent a diastolic minimum.
    expected_minima_idx = np.array([
        50, 100, 171, 221, 291, 341,
        391, 491, 541, 591, 641,
    ])

    np.testing.assert_array_equal(
        user_results["bvp_min_idx"],
        expected_minima_idx,
    )

    np.testing.assert_allclose(
        user_results["bvp_min_value"],
        bvp_signal[expected_minima_idx],
        atol=1e-12,
    )

    # The artifact-suppressed minimum at 441 produces the
    # 100-sample waveform between 391 and 491. This exceeds
    # 1.5 times the local median duration and is rejected.
    accepted_bounds = [
        (50, 100),
        (100, 171),
        (171, 221),
        (221, 291),
        (291, 341),
        (341, 391),
        (491, 541),
        (541, 591),
        (591, 641),
    ]

    expected_max_idx = np.array([
        start + np.argmax(bvp_signal[start:stop])
        for start, stop in accepted_bounds
    ])

    expected_max_value = bvp_signal[
        expected_max_idx
    ]

    # --- Check maxima of the retained waveforms ---
    np.testing.assert_array_equal(
        user_results["bvp_max_idx"],
        expected_max_idx,
    )

    np.testing.assert_allclose(
        user_results["bvp_max_value"],
        expected_max_value,
        atol=1e-12,
    )

    # --- Check the retained raw waveform segments ---
    extracted_waveforms = details[
        "list_wav_raw_and_y_normal"
    ]

    assert len(extracted_waveforms) == len(
        accepted_bounds
    )

    for extracted_waveform, (start, stop) in zip(
            extracted_waveforms,
            accepted_bounds):

        np.testing.assert_allclose(
            extracted_waveform["wav_raw"],
            bvp_signal[start:stop],
            atol=1e-12,
        )

    # --- Check that only retained waveforms are normalized ---
    xy_normal = details[
        "nparray_wav_xy_normal_all"
    ]

    zscore_normal = details[
        "nparray_wav_xy_normal_zscore_all"
    ]

    assert xy_normal.shape == (
        100,
        len(accepted_bounds),
    )

    assert zscore_normal.shape == (
        100,
        len(accepted_bounds),
    )

    assert np.all(np.isfinite(xy_normal))
    assert np.all(np.isfinite(zscore_normal))

def test_remove_artifact_waveforms():
    """Tests channel-wise detection and removal of artifact waveforms.

    Each channel contains 40 physiological waveforms. One waveform per
    channel is deliberately distorted. The test verifies the deviation
    metric, percentile threshold, detected artifact, cleaned waveform
    matrices, and preservation of the original matrices.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Generate a physiological reference waveform ---
    fs = 50.0
    n_samples = 100
    n_waveforms = 40

    phase = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angles = 2 * np.pi * k * phase

    reference_waveform = np.sum(
        a[:, None] * np.sin(angles)
        + b[:, None] * (np.cos(angles) - 1),
        axis=0,
    )
    reference_waveform = (
        reference_waveform + np.sum(b)
    )

    # --- Build a two-channel BVP time series ---
    channels = ["S1D1", "S2D2"]
    time = np.arange(n_samples) / fs

    bvp_ts = build_timeseries(
        np.vstack([
            reference_waveform,
            0.8 * reference_waveform,
        ]),
        ["channel", "time"],
        time,
        channels,
        "uM",
        "s",
    )

    # The artifact appears at a different position and with a
    # different direction in each channel.
    artifact_specs = {
        "S1D1": (9, 1.0),
        "S2D2": (27, -1.0),
    }

    wav_storage_user = {}
    wav_storage_details = {}
    expected_results = {}

    for channel in channels:
        artifact_idx, artifact_direction = (
            artifact_specs[channel]
        )

        # --- Create 40 initially identical waveforms ---
        xy_normal = np.repeat(
            reference_waveform[:, None],
            n_waveforms,
            axis=1,
        )

        # Add a localized distortion to one waveform.
        artifact_distortion = (
            artifact_direction
            * 0.8
            * np.exp(
                -0.5
                * ((phase - 0.65) / 0.06) ** 2
            )
        )

        xy_normal[:, artifact_idx] = (
            reference_waveform
            + artifact_distortion
        )

        # --- Z-score each waveform independently ---
        zscore_normal = (
            xy_normal
            - np.mean(xy_normal, axis=0)
        ) / np.std(
            xy_normal,
            axis=0,
        )

        wav_storage_user[channel] = {}

        wav_storage_details[channel] = {
            "nparray_wav_xy_normal_all":
                xy_normal.copy(),
            "nparray_wav_xy_normal_zscore_all":
                zscore_normal.copy(),
        }

        # --- Calculate the expected artifact scores ---
        expected_mean = np.mean(
            zscore_normal,
            axis=1,
        )

        expected_deviation = np.sum(
            np.abs(
                zscore_normal
                - expected_mean[:, None]
            ),
            axis=0,
        )

        expected_p975 = np.percentile(
            expected_deviation,
            97.5,
        )

        expected_artifact_idx = np.where(
            expected_deviation > expected_p975
        )[0]

        # Verify that the artificial distortion is the only outlier.
        np.testing.assert_array_equal(
            expected_artifact_idx,
            np.array([artifact_idx]),
        )

        expected_results[channel] = {
            "xy_normal": xy_normal,
            "zscore_normal": zscore_normal,
            "deviation": expected_deviation,
            "p975": expected_p975,
            "artifact_idx": artifact_idx,
        }

    # --- Detect and remove artifact waveforms ---
    result_user, result_details = (
        remove_artifact_waveforms(
            bvp_ts,
            wav_storage_user,
            wav_storage_details,
        )
    )

    # The input dictionaries are updated in place.
    assert result_user is wav_storage_user
    assert result_details is wav_storage_details

    # --- Check the channel-wise results ---
    for channel in channels:
        expected = expected_results[channel]
        artifact_idx = expected["artifact_idx"]

        user_results = result_user[channel]
        details = result_details[channel]

        expected_clean_xy = np.delete(
            expected["xy_normal"],
            artifact_idx,
            axis=1,
        )

        expected_clean_zscore = np.delete(
            expected["zscore_normal"],
            artifact_idx,
            axis=1,
        )

        # Check deviation scores and percentile threshold.
        np.testing.assert_allclose(
            details["bvp_wav_dev"],
            expected["deviation"],
            atol=1e-12,
        )

        np.testing.assert_allclose(
            details["P_975"],
            expected["p975"],
            atol=1e-12,
        )

        # Check removal from both normalized matrices.
        np.testing.assert_allclose(
            user_results[
                "nparray_wav_xy_normal_all_woa"
            ],
            expected_clean_xy,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            user_results[
                "nparray_wav_xy_normal_zscore_all_woa"
            ],
            expected_clean_zscore,
            atol=1e-12,
        )

        assert user_results[
            "nparray_wav_xy_normal_all_woa"
        ].shape == (
            n_samples,
            n_waveforms - 1,
        )

        assert user_results[
            "nparray_wav_xy_normal_zscore_all_woa"
        ].shape == (
            n_samples,
            n_waveforms - 1,
        )

        # The original matrices must remain available unchanged.
        np.testing.assert_allclose(
            details[
                "nparray_wav_xy_normal_all"
            ],
            expected["xy_normal"],
            atol=1e-12,
        )

        np.testing.assert_allclose(
            details[
                "nparray_wav_xy_normal_zscore_all"
            ],
            expected["zscore_normal"],
            atol=1e-12,
        )

def test_classify_waveforms_max():
    """Tests waveform classification by maximum amplitude.

    Physiological waveforms with known maximum values are provided in two
    channels and deliberately arranged in different orders. The calculated
    maxima, percentile thresholds, and classified waveform matrices are
    checked channel-wise.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Generate a physiological reference waveform ---
    fs = 50.0
    n_samples = 100
    phase = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angles = 2 * np.pi * k * phase

    reference_waveform = np.sum(
        a[:, None] * np.sin(angles)
        + b[:, None] * (np.cos(angles) - 1),
        axis=0,
    )
    reference_waveform = (
        reference_waveform + np.sum(b)
    )

    # --- Normalize the reference waveform to a maximum of one ---
    reference_normalized = (
        reference_waveform
        - np.mean(reference_waveform)
    ) / np.std(reference_waveform)

    reference_normalized = (
        reference_normalized
        / np.max(reference_normalized)
    )

    # --- Build a two-channel BVP time series ---
    channels = ["S1D1", "S2D2"]
    time = np.arange(n_samples) / fs

    bvp_ts = build_timeseries(
        np.vstack([
            reference_waveform,
            0.8 * reference_waveform,
        ]),
        ["channel", "time"],
        time,
        channels,
        "uM",
        "s",
    )

    # Each channel contains the same maximum values in a
    # different waveform order.
    expected_maxima = {
        "S1D1": np.arange(1.0, 9.0),
        "S2D2": np.array([
            4.0, 8.0, 1.0, 6.0,
            3.0, 7.0, 2.0, 5.0,
        ]),
    }

    bvp_cont = BVP_Container()
    bvp_cont["bvp_ts"] = bvp_ts
    bvp_cont.wav_storage_user = {}
    bvp_cont.wav_storage_details = {}

    waveform_matrices = {}

    for channel in channels:
        maxima = expected_maxima[channel]

        waveforms = (
            reference_normalized[:, None]
            * maxima[None, :]
        )

        waveform_matrices[channel] = waveforms.copy()

        bvp_cont.wav_storage_user[channel] = {
            "nparray_wav_xy_normal_zscore_all_woa":
                waveforms,
        }
        bvp_cont.wav_storage_details[channel] = {}

    # --- Classify waveforms by their maxima ---
    result_user, result_details = classify_waveforms(
        bvp_cont,
        "max",
    )

    assert result_user is bvp_cont.wav_storage_user
    assert result_details is bvp_cont.wav_storage_details
    assert set(result_user) == set(channels)
    assert set(result_details) == set(channels)

    # --- Check channel-wise classification results ---
    for channel in channels:
        maxima = expected_maxima[channel]
        waveforms = waveform_matrices[channel]

        user_results = result_user[channel]
        details = result_details[channel]

        expected_p25 = 2.75
        expected_p75 = 6.25

        idx_type1 = maxima < expected_p25
        idx_type2 = maxima > expected_p75
        idx_type3 = (
            (maxima > expected_p25)
            & (maxima < expected_p75)
        )

        assert set(user_results) == {
            "nparray_wav_xy_normal_zscore_all_woa",
            "nparray_wav_max_type1",
            "nparray_wav_max_type2",
            "nparray_wav_max_type3",
        }
        assert set(details) == {
            "max_bvp_wav",
            "max_P_25",
            "max_P_75",
        }

        # --- Check classification metric and thresholds ---
        np.testing.assert_allclose(
            details["max_bvp_wav"],
            maxima,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            details["max_P_25"],
            expected_p25,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            details["max_P_75"],
            expected_p75,
            atol=1e-12,
        )

        # --- Check classified waveform matrices ---
        np.testing.assert_allclose(
            user_results["nparray_wav_max_type1"],
            waveforms[:, idx_type1],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            user_results["nparray_wav_max_type2"],
            waveforms[:, idx_type2],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            user_results["nparray_wav_max_type3"],
            waveforms[:, idx_type3],
            atol=1e-12,
        )

        assert user_results[
            "nparray_wav_max_type1"
        ].shape == (n_samples, 2)

        assert user_results[
            "nparray_wav_max_type2"
        ].shape == (n_samples, 2)

        assert user_results[
            "nparray_wav_max_type3"
        ].shape == (n_samples, 4)

def test_classify_waveforms_delta():
    """Tests waveform classification by delta.

    Physiological waveforms with known deltas are provided
    in two channels. Each channel additionally contains one waveform without
    a dicrotic notch, which must be removed before classification.
    """

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Generate a physiological waveform with a dicrotic notch ---
    fs = 50.0
    n_samples = 100
    phase = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angles = 2 * np.pi * k * phase

    reference_waveform = np.sum(
        a[:, None] * np.sin(angles)
        + b[:, None] * (np.cos(angles) - 1),
        axis=0,
    )
    reference_waveform = (
        reference_waveform + np.sum(b)
    )

    # The physiological reference waveform contains one local
    # minimum inside the accepted sample range from 20 to 60.
    notch_idx, notch_value = peakseek(
        -reference_waveform
    )
    notch_value = -notch_value

    np.testing.assert_array_equal(
        notch_idx,
        np.array([38]),
    )

    reference_delta = (
        np.max(reference_waveform)
        - notch_value[0]
    )

    # --- Build a two-channel BVP time series ---
    channels = ["S1D1", "S2D2"]
    time = np.arange(n_samples) / fs

    bvp_ts = build_timeseries(
        np.vstack([
            reference_waveform,
            0.8 * reference_waveform,
        ]),
        ["channel", "time"],
        time,
        channels,
        "uM",
        "s",
    )

    # Both channels contain the same delta values in a
    # different waveform order.
    expected_deltas = {
        "S1D1": np.arange(1.0, 9.0),
        "S2D2": np.array([
            4.0, 8.0, 1.0, 6.0,
            3.0, 7.0, 2.0, 5.0,
        ]),
    }

    bvp_cont = BVP_Container()
    bvp_cont["bvp_ts"] = bvp_ts
    bvp_cont.wav_storage_user = {}
    bvp_cont.wav_storage_details = {}

    valid_waveform_matrices = {}

    for channel in channels:
        delta_values = expected_deltas[channel]

        valid_waveforms = (
            reference_waveform[:, None]
            * (
                delta_values[None, :]
                / reference_delta
            )
        )

        # A parabolic waveform has no local minimum and therefore
        # produces a delta of zero. It must be removed.
        parabola_x = np.linspace(
            -1.0,
            1.0,
            n_samples,
        )

        invalid_waveform = (
            reference_waveform[0]
            * parabola_x ** 2
        )

        waveforms = np.column_stack([
            valid_waveforms,
            invalid_waveform,
        ])

        valid_waveform_matrices[channel] = (
            valid_waveforms.copy()
        )

        bvp_cont.wav_storage_user[channel] = {
            "nparray_wav_xy_normal_all_woa":
                waveforms,
        }
        bvp_cont.wav_storage_details[channel] = {}

    # --- Classify waveforms by their delta ---
    result_user, result_details = classify_waveforms(
        bvp_cont,
        "delta",
    )

    assert result_user is bvp_cont.wav_storage_user
    assert result_details is bvp_cont.wav_storage_details
    assert set(result_user) == set(channels)
    assert set(result_details) == set(channels)

    # --- Check channel-wise classification results ---
    for channel in channels:
        delta_values = expected_deltas[channel]
        valid_waveforms = valid_waveform_matrices[
            channel
        ]

        user_results = result_user[channel]
        details = result_details[channel]

        expected_p25 = 2.75
        expected_p75 = 6.25

        idx_type1 = delta_values < expected_p25
        idx_type2 = delta_values > expected_p75
        idx_type3 = (
            (delta_values > expected_p25)
            & (delta_values < expected_p75)
        )

        assert set(user_results) == {
            "nparray_wav_xy_normal_all_woa",
            "nparray_wav_delta_type1",
            "nparray_wav_delta_type2",
            "nparray_wav_delta_type3",
        }
        assert set(details) == {
            "delta_bvp_wav",
            "delta_P_25",
            "delta_P_75",
            "text_num_del_wavs",
        }

        # --- Check removed waveform and classification metric ---
        assert details[
            "text_num_del_wavs"
        ] == f"{channel}:  1  of  9"

        np.testing.assert_allclose(
            details["delta_bvp_wav"],
            delta_values,
            atol=1e-12,
        )

        # --- Check percentile thresholds ---
        np.testing.assert_allclose(
            details["delta_P_25"],
            expected_p25,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            details["delta_P_75"],
            expected_p75,
            atol=1e-12,
        )

        # --- Check classified waveform matrices ---
        np.testing.assert_allclose(
            user_results["nparray_wav_delta_type1"],
            valid_waveforms[:, idx_type1],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            user_results["nparray_wav_delta_type2"],
            valid_waveforms[:, idx_type2],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            user_results["nparray_wav_delta_type3"],
            valid_waveforms[:, idx_type3],
            atol=1e-12,
        )

        assert user_results[
            "nparray_wav_delta_type1"
        ].shape == (n_samples, 2)

        assert user_results[
            "nparray_wav_delta_type2"
        ].shape == (n_samples, 2)

        assert user_results[
            "nparray_wav_delta_type3"
        ].shape == (n_samples, 4)

