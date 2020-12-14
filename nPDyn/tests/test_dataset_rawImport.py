import os

path = os.path.dirname(os.path.abspath(__file__))

import pytest

import numpy as np

import nPDyn


def test_import_QENS():
    dataPath = path + "/sample_data/vanadium/"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "QENS")
    assert data.dataList[0].data.qVals is not None


def test_import_FWS():
    dataPath = path + "/sample_data/lys_part_01/"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "FWS")
    assert data.dataList[0].data.qVals is not None


def test_import_res():
    dataPath = path + "/sample_data/vanadium/"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "res")
    assert data.resData[0].data.qVals is not None


def test_import_ec():
    dataPath = path + "/sample_data/vanadium"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "ec")
    assert data.ECData.data.qVals is not None


def test_import_fec():
    dataPath = path + "/sample_data/lys_part_01"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "fec")
    assert data.fECData.data.qVals is not None


def test_import_D2O():
    dataPath = path + "/sample_data/vanadium"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "D2O")
    assert data.D2OData.data.qVals is not None


def test_import_fD2O():
    dataPath = path + "/sample_data/lys_part_01"
    data = nPDyn.Dataset()
    data.importRawData(dataPath, "IN16B", "fD2O")
    assert data.fD2OData.data.qVals is not None
