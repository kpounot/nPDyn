import os

filePath = os.path.dirname(os.path.abspath(__file__))

import unittest

import pytest

import nPDyn

try:
    from mantid.simpleapi import IndirectILLReductionQENS

    _FOUND_MANTID = True
except ImportError:
    _FOUND_MANTID = False


@pytest.mark.skipif(_FOUND_MANTID is False, reason="Requires Mantid")
@pytest.mark.parametrize(
    "attr",
    [
        "intensities",
        "errors",
        "energies",
        "qVals",
        "qIdx",
        "times",
        "temps",
        "observable",
        "observable_name",
        "name",
    ],
)
def test_workspace(attr):
    vana = IndirectILLReductionQENS(
        filePath + "/../sample_data/vanadium/270940.nxs",
        SumRuns=True,
        SpectrumAxis="Q",
    )
    data = nPDyn.Dataset(resFiles=[vana])
    assert len(data.resData[0].data.__getattribute__(attr)) > 0


@pytest.mark.skipif(_FOUND_MANTID is False, reason="Requires Mantid")
@pytest.mark.parametrize(
    "replace,expect",
    [
        pytest.param("intensities", [0, 1, 2], marks=pytest.mark.xfail),
        ("qVals", [0, 1, 2]),
        pytest.param("error", [0, 1, 2], marks=pytest.mark.xfail),
    ],
)
def test_replaceWorkspace(replace, expect):
    vana = IndirectILLReductionQENS(
        filePath + "/../sample_data/vanadium/270940.nxs",
        SumRuns=True,
        SpectrumAxis="Q",
    )
    data = nPDyn.Dataset(resFiles=[vana])
    newData = data.resData[0].data._replace(**{replace: expect})
    assert newData.__getattribute__(replace) == expect
