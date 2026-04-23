import pytest
import cedalion.utils as utils


def test_zero_padded_numbers():
    assert utils.zero_padded_numbers([1, 2, 3]) == ["1", "2", "3"]
    assert utils.zero_padded_numbers([1, 9, 30]) == ["01", "09", "30"]
    assert utils.zero_padded_numbers([1, 19, 300]) == ["001", "019", "300"]

    assert utils.zero_padded_numbers([1, 19, 93], "S") == ["S01", "S19", "S93"]

    assert utils.zero_padded_numbers([1, 19, -930]) == ["001", "019", "-930"]
    assert utils.zero_padded_numbers([]) == []
    assert utils.zero_padded_numbers(range(1,1)) == []

