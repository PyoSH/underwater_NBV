import numpy as np
import pytest

from stage2_sitl_dvl_injector import confidence_from_fom, flu_to_frd


def test_flu_to_frd_axis_contract():
    np.testing.assert_allclose(
        flu_to_frd(np.array([1.0, 2.0, 3.0])), [1.0, -2.0, -3.0]
    )


@pytest.mark.parametrize(
    ("fom", "expected"),
    [(0.0, 100.0), (0.003, 99.25), (0.2, 50.0), (0.4, 0.0), (1.0, 0.0)],
)
def test_water_linked_fom_mapping(fom, expected):
    assert confidence_from_fom(fom) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_bad_fom_rejected(bad):
    with pytest.raises(ValueError):
        confidence_from_fom(bad)


def test_bad_body_vector_rejected():
    with pytest.raises(ValueError):
        flu_to_frd(np.array([1.0, 2.0]))
