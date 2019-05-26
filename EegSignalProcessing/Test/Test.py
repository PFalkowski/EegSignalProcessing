import unittest
import EegSignalProcessing as eeg
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


if __name__ == '__main__':
    unittest.main()

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

    #def test_ctor_SetsVariables(self):
    #    eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
    #    actual = eegFile.subject
    #    expected = "TestSub01"
    #    self.assertEqual(expected, actual)
    #    actual = eegFile.session
    #    expected = "TestSession"
    #    self.assertEqual(expected, actual)
    #    actual = eegFile.condition
    #    expected = "testCondition"
    #    self.assertEqual(expected, actual)
        #actual = eegFile.binaryCondition
        #expected = "Unconscious"
        #self.assertEqual(expected, actual)
   
    def test_AsDataFrame(self):
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.AsDataFrame(False)
        self.assertEqual(actual.shape, (6553, 128))        
        
    def test_Subject(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.Subject()
        expected = "TestSub01"
        self.assertEqual(expected, actual)

    def test_Session(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.Session()
        expected = "TestSession"
        self.assertEqual(expected, actual)
        
    def test_Condition(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.Condition()
        expected = "testCondition"
        self.assertEqual(expected, actual)
        
    #def test_BinaryCondition_Conscious(self):
    #    eegFile = eeg.EegFile("Test/TestSub01_TestSession_AwakeEyesOpened.vhdr")
    #    actual = eegFile.BinaryCondition()
    #    expected = "Conscious"
    #    self.assertEqual(expected, actual)

    #def test_BinaryCondition_Unconscious(self):
    #    eegFile = eeg.EegFile("Test/TestSub01_TestSession_Sleeping.vhdr")
    #    actual = eegFile.BinaryCondition()
    #    expected = "Unconscious"
    #    self.assertEqual(expected, actual)

    def test_GetAverageBandpower(self):
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.GetAverageBandpower(False)
        #expected = {'Alpha': 0.0013945768812877765, 'Beta': 0.0016353515167911857, 'Delta': 0.0015713140875959664, 'Gamma': 0.0016561031328069058, 'Theta': 0.001499703555615015}
        expected = {'Alpha': 0.046372396504643934, 'Beta': 0.021799368301619663, 'Delta': 0.3797795190319582, 'Gamma': 0.015256991787747547, 'Theta': 0.0961496475016523}
        self.assertDictEqual(expected, actual);
        
    def test_GetAverageChannelBandpower(self):
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        actual = eegFile.GetAverageChannelBandpower("ECoG_ch001")
        expected = {'Alpha': 0.001581301582526992, 'Beta': 0.001105882882813178, 'Delta': 0.02409971332527757, 'Gamma': 0.0008023666358686522, 'Theta': 0.002751648980509086}
        self.assertDictEqual(expected, actual);

    #def test_Fft(self):
    #    chName = "ECoG_ch003"
    #    eegFile = eeg.EegFile("Test/100HzTest.vhdr")
    #    eegFile.plotSpectrum()
    #    data = eegFile.GetChannel(chName)
    #    #data = eegFile.AsDataFrame()
    #    plt.plot(data)
    #    plt.show()
    #    bands = eegFile.GetAverageBandpower()
    #    eeg.EegFile.PlotBands(bands)
    #    ## Perform FFT WITH SCIPY
    #    signalFFT = np.fft.rfft(data)

    #    ## Get Power Spectral Density
    #    signalPSD = np.abs(signalFFT) ** 2

    #    ## Get frequencies corresponding to signal PSD
    #    fftFreq = np.fft.rfftfreq(len(data), 1.0/eegFile.samplingRate)

    #    plt.figurefigsize=(8,4)
    #    plt.plot(fftFreq, 10*np.log10(signalPSD))
    #    #plt.xlim(0, 100);
    #    plt.xlabel('Frequency Hz')
    #    plt.ylabel('Power Spectral Density (dB)')
    #    plt.show()
    #    print('duh')

