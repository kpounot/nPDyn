import os

path = os.path.dirname(os.path.abspath(__file__))

import unittest
import pytest

import nPDyn


def test_import_ec():
    dataPath = path + "/sample_data/empty_cell_QENS_280K.nxs"
    data = nPDyn.Dataset(ECFile=dataPath)
    assert data.ECData.data.qVals is not None


def test_import_fec():
    dataPath = path + "/sample_data/lys_part_01_FWS.nxs"
    data = nPDyn.Dataset(fECFile=dataPath)
    assert data.fECData.data.qVals is not None


def test_import_resFiles():
    dataPath = path + "/sample_data/D_syn_fibers_QENS_10K.inx"
    dataList = [dataPath, dataPath]
    data = nPDyn.Dataset(resFiles=dataList)
    assert data.resData[1].data.qVals is not None


def test_import_D2O():
    dataPath = path + "/sample_data/D_syn_fibers_QENS_300K.inx"
    data = nPDyn.Dataset(D2OFile=dataPath)
    assert data.D2OData.data.qVals is not None


def test_import_fD2O():
    dataPath = path + "/sample_data/lys_part_01_FWS.nxs"
    data = nPDyn.Dataset(fD2OFile=dataPath)
    assert data.fD2OData.data.qVals is not None


def test_import_QENS():
    dataPath = path + "/sample_data/D_syn_fibers_QENS_300K.inx"
    dataList = [dataPath, dataPath]
    data = nPDyn.Dataset(QENSFiles=dataList)
    assert data.dataList[1].data.qVals is not None


def test_import_FWS():
    dataPath = path + "/sample_data/lys_part_01_FWS.nxs"
    dataList = [dataPath, dataPath]
    data = nPDyn.Dataset(FWSFiles=dataList)
    assert data.dataList[1].data.qVals is not None
