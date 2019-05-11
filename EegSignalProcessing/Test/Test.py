import unittest
import EegSignalProcessing as eeg

class Test_File(unittest.TestCase):

    def test_ctor_ThrowsWhenNoFile(self):
        with self.assertRaises(ValueError):
            eeg.File("fileThatDoesNotExist.txt")

    def test_ctor_SetsVariables(self):
        with self.assertRaises(ValueError):
            eeg.File("fileThatDoesNotExist.txt")
            
class Test_EegFile(unittest.TestCase):
    
    def test_ctor_ThrowsWhenNoFile(self):
        with self.assertRaises(ValueError):
            eeg.EegFile("fileThatDoesNotExist.txt")
   
    def test_AsDataFrame(self):
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.AsDataFrame()
        self.assertEqual(actual.shape, (6553, 128))

    def test_GetAverageBandpower(self):
        samplingRate = 100
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.GetAverageBandpower()
        expected = {'Alpha': 0.0013945768812877765, 'Beta': 0.0016353515167911857, 'Delta': 0.0015713140875959664, 'Gamma': 0.0016561031328069058, 'Theta': 0.001499703555615015}
        self.assertDictEqual(expected, actual);

    def test_GetAverageBandpower(self):
        samplingRate = 100
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.GetAverageBandpower()
        expected = {'Alpha': 0.0013945768812877765, 'Beta': 0.0016353515167911857, 'Delta': 0.0015713140875959664, 'Gamma': 0.0016561031328069058, 'Theta': 0.001499703555615015}
        self.assertDictEqual(expected, actual);

if __name__ == '__main__':
    unittest.main()

