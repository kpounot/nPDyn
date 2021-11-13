import os

path = os.path.dirname(os.path.abspath(__file__))

from nPDyn.dataParsers.groupScans import printGroup, groupScansIN16B


def test_group_in16b():
    out = printGroup(groupScansIN16B(path + "/../sample_data/bats_data/"))
    assert out.split("\n")[2] == "\tscans: 316112:316122"
