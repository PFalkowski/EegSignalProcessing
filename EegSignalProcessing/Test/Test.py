import unittest
import EegSignalProcessing as eeg
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import mne
import pandas as pd


if __name__ == '__main__':
    unittest.main()

class Test_File(unittest.TestCase):

    def test_ctor_ThrowsWhenNoFile(self):
        with self.assertRaises(ValueError):
            eeg.File("fileThatDoesNotExist.txt")

    def test_ctor_SetsVariables(self):
        tested = eeg.File("Test/fileThatExists.txt")
        self.assertTrue(tested.fullFilePath.endswith("Test/fileThatExists.txt"))
        self.assertEqual("fileThatExists", tested.nameWithoutExtension)
        self.assertFalse(tested.pathWithoutFileName.endswith("Test/fileThatExists.txt"))

    def test_ComputeSha256(self):
        file = eeg.File("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = file.ComputeFileSha256()
        expected = "aed7686f60db75fec3016e136f7bdb73a0c8dc6ca57bb55051502647528b0974"
        self.assertEqual(expected, actual)
            
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
   
    def test_AsDataFrame_withoutLabels(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.AsDataFrame(False)
        self.assertEqual((6553, 128), actual.shape)        

    def test_AsDataFrame_withLabels(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.AsDataFrame(True)
        self.assertEqual((6553, 133), actual.shape)        
        
    def test_Subject(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.subject
        expected = "TestSub01"
        self.assertEqual(expected, actual)

    def test_Session(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.session
        expected = "TestSession"
        self.assertEqual(expected, actual)
        
    def test_Condition(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.condition
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

class Test_EegSample(unittest.TestCase):
    
    def GetMockDataFrame(self, withLabels = True):
        rawData = mne.io.read_raw_brainvision("Test/TestSub01_TestSession_testCondition.vhdr", preload=True, stim_channel=False, verbose = True)
        brain_vision = rawData.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=rawData.ch_names)
        if (withLabels):             
            df["Subject"] = "TestSub01"
            df["Session"] = "TestSession"
            df["Condition"] = "testCondition"
            df["BinaryCondition"] = "testCondition"
            df["TernaryCondition"] = "testCondition"
        return df
    
    def test_DataFrameHasLabels_True(self):
        df = self.GetMockDataFrame(True)
        actual = eeg.EegSample.DataFrameHasLabels(df)
        self.assertTrue(actual)

    def test_DataFrameHasLabels_False(self):
        df = self.GetMockDataFrame(False)
        actual = eeg.EegSample.DataFrameHasLabels(df)
        self.assertFalse(actual)
        
    def test_DataFrameHasLabels_CustomLabels_True(self):
        df = self.GetMockDataFrame(True)
        actual = eeg.EegSample.DataFrameHasLabels(df, ["Subject", "Session"])
        self.assertTrue(actual)
        
    def test_DataFrameHasLabels_CustomLabels_False(self):
        df = self.GetMockDataFrame(False)
        actual = eeg.EegSample.DataFrameHasLabels(df, ["Subject", "Session"])
        self.assertFalse(actual)

    def test_GetDfWithDroppedLabels(self):
        df = self.GetMockDataFrame(True)
        dfWithDroppedLabels = eeg.EegSample.GetDfWithDroppedLabels(df, ["Subject", "Session"])
        self.assertEqual((6553, 131), dfWithDroppedLabels.shape)
        
    def test_GetDfWithDroppedLabels_WhenNoLabelsPassed(self):
        df = self.GetMockDataFrame(True)
        dfWithDroppedLabels = eeg.EegSample.GetDfWithDroppedLabels(df, [])
        self.assertEqual((6553, 133), dfWithDroppedLabels.shape)

    def test_GetDfWithDroppedLabels_WhenNoMatchingLabelsPassed(self):
        df = self.GetMockDataFrame(True)
        self.assertRaises(KeyError, eeg.EegSample.GetDfWithDroppedLabels, df, ["TheseAreNotTheLabelsYouAreLookingFor"])
        
    def test_GetChannel(self):
        df = self.GetMockDataFrame(False)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetChannel("ECoG_ch001")
        self.assertEqual((6553,), actual.shape)
        
    def test_GetRandomSubset_WithLabels(self):
        df = self.GetMockDataFrame(True)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetRandomSubset(0.1, True)
        expected = (int(6553 * 0.1), 133)
        self.assertEqual(expected, actual.shape)

    def test_GetRandomSubset_WithLabels_ThrowsWhenNoLabels(self):
        df = self.GetMockDataFrame(False)
        tested = eeg.EegSample(df, 100)
        self.assertRaises(ValueError, tested.GetRandomSubset, 0.1, True)
        
    def test_GetRandomSubset_RatioIsOne(self):
        df = self.GetMockDataFrame(True)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetRandomSubset(1, True)
        expected = (6553, 133)
        self.assertEqual(expected, actual.shape)
        
    def test_GetRandomSubset_RatioIsZero(self):
        df = self.GetMockDataFrame(True)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetRandomSubset(0, True)
        expected = (0, 133)
        self.assertEqual(expected, actual.shape)
        
    def test_GetRandomSubset_NoLabels(self):
        df = self.GetMockDataFrame(True)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetRandomSubset(0.5, False)
        expected = (int(6553 * 0.5), 128)
        self.assertEqual(expected, actual.shape)

    def test_GetRandomSubset_NoLabels2(self):
        df = self.GetMockDataFrame(False)
        tested = eeg.EegSample(df, 100)
        actual = tested.GetRandomSubset(0.5, False)
        expected = (int(6553 * 0.5), 128)
        self.assertEqual(expected, actual.shape)        
        
    def test_InitializeFromEegFile(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        tested = eeg.EegSample.InitializeFromEegFile(eegFile)
        actual = tested.dataFrame.shape
        expected = (6553, 133)
        self.assertEqual(expected, actual)

    def test_Ctor_RaisesErrorWhenNotPdDf(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        self.assertRaises(TypeError, eeg.EegSample, eegFile)
        
    def test_GetDataFrame_GetsLabels(self):
        tested = eeg.EegSample(self.GetMockDataFrame(), 100)
        actual = tested.GetDataFrame(True).shape
        expected = (6553, 133)
        self.assertEqual(expected, actual)

    def test_GetDataFrame_NoLabels(self):
        tested = eeg.EegSample(self.GetMockDataFrame(), 100)
        actual = tested.GetDataFrame(False).shape
        expected = (6553, 128)
        self.assertEqual(expected, actual)

    def test_GetAverageBandpower(self):
        eegSample = eeg.EegSample(self.GetMockDataFrame(), 100)
        actual = eegSample.GetAverageBandpower()
        expected = {'Alpha': 0.046372396504643934, 'Beta': 0.021799368301619663, 'Delta': 0.3797795190319582, 'Gamma': 0.015256991787747547, 'Theta': 0.0961496475016523}
        self.assertDictEqual(expected, actual);
        
    def test_GetAverageChannelBandpower(self):
        eegSample = eeg.EegSample(self.GetMockDataFrame(), 100)
        actual = eegSample.GetAverageChannelBandpower("ECoG_ch001")
        expected = {'Alpha': 0.001581301582526992, 'Beta': 0.001105882882813178, 'Delta': 0.02409971332527757, 'Gamma': 0.0008023666358686522, 'Theta': 0.002751648980509086}
        self.assertDictEqual(expected, actual);


