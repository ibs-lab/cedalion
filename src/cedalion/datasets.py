"""Cedalin datasets and utility functions."""

import pooch

import cedalion.io

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://eike.middell.net/share/cedalion",  # FIXME
    env="CEDALION_DATA_DIR",
    registry={
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944"  # noqa: E501
    },
)


def get_snirf_test_data():
    fname = DATASETS.fetch("mne_nirsport2_raw.snirf")
    return cedalion.io.read_snirf(fname)
