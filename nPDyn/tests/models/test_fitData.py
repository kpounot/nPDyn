import pytest

import numpy as np

from nPDyn.models.builtins import modelPVoigt, modelD2OBackground
from nPDyn.models.presets import calibratedD2O


def test_fitRes_PVoigt(fullQENSDataset):
    q = fullQENSDataset.resData[0].data.qVals[:, np.newaxis]
    fullQENSDataset.fitRes(model=modelPVoigt(q))
    res = fullQENSDataset.resData[0]
    assert np.sum((res.data.intensities - res.fit_best()) ** 2) < 0.004


def test_fitD2O_Builtin(fullQENSDataset):
    q = fullQENSDataset.D2OData.data.qVals[:, np.newaxis]
    fullQENSDataset.D2OData.fit(modelD2OBackground(q))
    d2O = fullQENSDataset.D2OData
    assert np.sum((d2O.data.intensities - d2O.fit_best()) ** 2) < 0.005
