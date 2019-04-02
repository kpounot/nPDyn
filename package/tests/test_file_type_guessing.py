import unittest

from ..nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters
from ..nPDyn.dataParsers.inxConvert import convert


qData = namedtuple('qData', 'qVals X intensities errors temp norm qIdx')


class TestFileTypeGuess(unittest.TestCase):

    def test_inx_file_noFWS(self):

        testFile = "sample_data/D_syn_fibers_QENS_10K.inx"

        result = guessFileFormat(testFile)

        self.assertIsInstance(result, qData, ".inx file type couldn't be guessed correctly")
