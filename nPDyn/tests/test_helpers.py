import numpy as np

from nPDyn import Sample

from nPDyn.helpers import (
    groupBy_qValues,
    dataset_from_array,
    get_stokes_einstein_curves,
)


def test_group_q_vals(bats_data):
    new_data = groupBy_qValues(
        bats_data.diffraction[0], bats_data.qdiff, bats_data.q
    )
    assert new_data.shape == (19,)


def test_dataset_from_array():
    data = dataset_from_array(
        np.arange(6 * 6 * 6).reshape(6, 6, 6),
        errors=np.sqrt(np.arange(6 * 6 * 6).reshape(6, 6, 6)),
        energies=np.array([0, 1, 2, 3, 4, 5]),
        qVals=np.linspace(0, 2, 6),
    )
    assert data.shape == (6, 6, 6)
    assert isinstance(data, Sample)


def test_stokes_einstein():
    curves, temps, radii = get_stokes_einstein_curves([3.2e-10], [280])
    assert radii[0] == 2.834907803372412e-9
