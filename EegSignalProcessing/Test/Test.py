import unittest
import EegSignalProcessing as eeg
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import mne
import pandas as pd
import os


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
        
    def test_Validate_valid(self):
        file = eeg.File("Test/TestSub01_TestSession_testCondition.vhdr")
        shaDigest = "aed7686f60db75fec3016e136f7bdb73a0c8dc6ca57bb55051502647528b0974"
        actual = file.Validate(shaDigest)
        self.assertTrue(actual)
        
    def test_Validate_invalid(self):
        file = eeg.File("Test/TestSub01_TestSession_testCondition.vhdr")
        shaDigest = "aed7686f60db75fec3016e136f7bdb73a0c8dc6ca57bb55051502647528b0973"
        actual = file.Validate(shaDigest)
        self.assertFalse(actual)

    def test_Validate_invalid(self):
        tested = eeg.File("Test/fileThatExists.txt")
        actual = len(tested.GetAllLines())
        expected = 3
        self.assertEqual(expected, actual)
           

class Test_ChecksumFile(unittest.TestCase):
    
    def test_ctor_ThrowsWhenNoFile(self):
        with self.assertRaises(ValueError):
            eeg.ChecksumFile("fileThatDoesNotExist.txt")

    def test_GetChecksumDictionary(self):
        file = eeg.ChecksumFile("Test\Sub0x - checksums.txt")
        actual = file.GetChecksumDictionary()
        expected = {'S01': 'F6t', 'S02': 'A4t'}
        self.assertEqual(expected, actual)       

            
class Test_EegFile(unittest.TestCase):
    
    def test_ctor_ThrowsWhenNoFile(self):
        with self.assertRaises(ValueError):
            eeg.EegFile("fileThatDoesNotExist.txt")

    def test_ctor_SetsVariables(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.subject
        expected = "TestSub01"
        self.assertEqual(expected, actual)
        actual = eegFile.session
        expected = "TestSession"
        self.assertEqual(expected, actual)
        actual = eegFile.condition
        expected = "testCondition"
        self.assertEqual(expected, actual)
        
    def test_AsDataFrame_withoutLabels(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.AsDataFrame(False)
        self.assertEqual((6553, 128), actual.shape)        

    def test_AsDataFrame_withLabels(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        actual = eegFile.AsDataFrame(True)
        self.assertEqual((6553, 133), actual.shape)  

    def test_SaveToCsv_withLabelsNoExtensionRelativePath(self):
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")
        outputPath = "Test/test_SaveToCsv_withLabels.csv"
        eegFile.SaveToCsv(outputPath)
        self.assertTrue(os.path.isfile(outputPath))
        os.remove(outputPath)

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
    
    def GetMockDataFrame(self, withLabels=True):
        rawData = mne.io.read_raw_brainvision("Test/TestSub01_TestSession_testCondition.vhdr", preload=True, stim_channel=False, verbose = True)
        brain_vision = rawData.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=rawData.ch_names)
        if (withLabels):             
            df["Subject"] = "testSubject"
            df["Session"] = "testSession"
            df["Condition"] = "testAwakeCondition"
            df["BinaryCondition"] = "Conscious"
            df["TernaryCondition"] = "Conscious"
        return df      
    
    def GetMockEegSample(self, withLabels=True):
        df = self.GetMockDataFrame(withLabels)
        subject = "testSubject"
        session = "testSession"
        condition = "testAwakeCondition"
        return eeg.EegSample(df, 100, subject, session, condition)
    
    def test_Ctor(self):
        df = self.GetMockDataFrame(True)
        subject = "testSubject"
        session = "testSession"
        condition = "testConditionAwake"
        tested = eeg.EegSample(df, 78, subject, session, condition)
        self.assertEqual(78, tested.samplingRate)
        self.assertEqual(df.shape, tested.dataFrame.shape)
        self.assertEqual(subject, tested.subject)
        self.assertEqual(session, tested.session)
        self.assertEqual(condition, tested.condition)
        self.assertEqual("Conscious", tested.binaryCondition)
        self.assertEqual("Conscious", tested.ternaryCondition)
    
    def test_InitializeFromEegFile(self):	
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")	
        tested = eeg.EegSample.InitializeFromEegFile(eegFile)	
        actual = tested.dataFrame.shape	
        expected = (6553, 133)	
        self.assertEqual(expected, actual)	        
           
    def test_BinaryCondition_Conscious(self):
        actual = eeg.EegSample.BinaryCondition("AwakeEyesOpened")
        expected = "Conscious"
        self.assertEqual(expected, actual)

    def test_BinaryCondition_Unconscious(self):
        actual = eeg.EegSample.BinaryCondition("Sleeping")
        expected = "Unconscious"
        self.assertEqual(expected, actual)
        
    def test_TernaryCondition_Conscious(self):
        actual = eeg.EegSample.TernaryCondition("AwakeEyesOpened")
        expected = "Conscious"
        self.assertEqual(expected, actual)

    def test_TernaryCondition_InBetween(self):
        actual = eeg.EegSample.TernaryCondition("RecoveryEyesClosed")
        expected = "InBetween"
        self.assertEqual(expected, actual)

    def test_Ctor_RaisesErrorWhenNotPdDf(self):	
        eegFile = eeg.EegFile("Test/TestSub01_TestSession_testCondition.vhdr")	
        subject = "testSubject"
        session = "testSession"
        condition = "testCondition"
        self.assertRaises(TypeError, eeg.EegSample, eegFile, 78, subject, session, condition)

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

    def test_GetDfWithoutLabels(self):
        df = self.GetMockDataFrame(True)
        dfWithDroppedLabels = eeg.EegSample.GetDfWithoutLabels(df, ["Subject", "Session"])
        self.assertEqual((6553, 131), dfWithDroppedLabels.shape)
        
    def test_GetDfWithoutLabels_WhenNoLabelsPassed(self):
        df = self.GetMockDataFrame(True)
        dfWithDroppedLabels = eeg.EegSample.GetDfWithoutLabels(df, [])
        self.assertEqual((6553, 133), dfWithDroppedLabels.shape)

    def test_GetDfWithoutLabels_WhenNoMatchingLabelsPassed(self):
        df = self.GetMockDataFrame(True)
        self.assertRaises(KeyError, eeg.EegSample.GetDfWithoutLabels, df, ["TheseAreNotTheLabelsYouAreLookingFor"])
        
    def test_GetChannel(self):
        tested = self.GetMockEegSample()
        actual = tested.GetChannel("ECoG_ch001")
        self.assertEqual((6553,), actual.shape)
        
    def test_GetRandomSubset_WithLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetRandomSubset(0.1, True)
        expected = (int(6553 * 0.1), 133)
        self.assertEqual(expected, actual.shape)

    def test_GetRandomSubset_WithLabels_ThrowsWhenNoLabels(self):
        tested = self.GetMockEegSample(False)
        self.assertRaises(ValueError, tested.GetRandomSubset, 0.1, True)
        
    def test_GetRandomSubset_RatioIsOne(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetRandomSubset(1, True)
        expected = (6553, 133)
        self.assertEqual(expected, actual.shape)
        
    def test_GetRandomSubset_RatioIsZero(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetRandomSubset(0, True)
        expected = (0, 133)
        self.assertEqual(expected, actual.shape)
        
    def test_GetRandomSubset_NoLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetRandomSubset(0.5, False)
        expected = (int(6553 * 0.5), 128)
        self.assertEqual(expected, actual.shape)

    def test_GetRandomSubset_NoLabels2(self):
        tested = self.GetMockEegSample(False)
        actual = tested.GetRandomSubset(0.5, False)
        expected = (int(6553 * 0.5), 128)
        self.assertEqual(expected, actual.shape)        
        
    def test_GetDataFrame_GetsLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetDataFrame(True).shape
        expected = (6553, 133)
        self.assertEqual(expected, actual)

    def test_GetDataFrame_NoLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetDataFrame(False).shape
        expected = (6553, 128)
        self.assertEqual(expected, actual)

    def test_GetAverageBandpower(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetAverageBandpower()
        expected = {'Alpha': 0.046372396504643934, 'Beta': 0.021799368301619663, 'Delta': 0.3797795190319582, 'Gamma': 0.015256991787747547, 'Theta': 0.0961496475016523}
        self.assertDictEqual(expected, actual)
        
    def test_GetAverageBandpowerAsDataFrame_NoLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetAverageBandpowerAsDataFrame(False)
        expected = pd.DataFrame({'Alpha': [0.046372396504643934], 'Beta': [0.021799368301619663], 'Delta': [0.3797795190319582], 'Gamma': [0.015256991787747547], 'Theta': [0.0961496475016523]})
        self.assertTrue(expected.sort_index(axis=1).equals(actual.sort_index(axis=1)))

    def test_GetAverageBandpowerAsDataFrame_WithLabels(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetAverageBandpowerAsDataFrame(True)
        expected = pd.DataFrame({
             'Alpha': [0.046372396504643934], 
             'Beta': [0.021799368301619663], 
             'Delta': [0.3797795190319582], 
             'Gamma': [0.015256991787747547], 
             'Theta': [0.0961496475016523],            
             'Subject': ['testSubject'],
             'Session': ['testSession'],
             'Condition': ['testAwakeCondition'],
             'BinaryCondition': ['Conscious'],
             'TernaryCondition': ['Conscious']
             })
        expected = expected.sort_index(axis=1)
        actual = actual.sort_index(axis=1)
        self.assertTrue(expected.equals(actual))
        
    def test_GetAverageChannelBandpower(self):
        tested = self.GetMockEegSample(True)
        actual = tested.GetAverageChannelBandpower("ECoG_ch001")
        expected = {'Alpha': 0.001581301582526992, 'Beta': 0.001105882882813178, 'Delta': 0.02409971332527757, 'Gamma': 0.0008023666358686522, 'Theta': 0.002751648980509086}
        self.assertDictEqual(expected, actual)
        
    def test_SplitEvenly(self):
        tested = self.GetMockEegSample(True)
        tested = tested.SplitEvenly(10)
        self.assertEqual(10, len(tested))
        
    def test_SplitEvenly_OneSlice(self):
        tested = self.GetMockEegSample(True)
        tested = tested.SplitEvenly(1)
        self.assertEqual(1, len(tested))
        
    def test_SplitEvenly_ZeroSlices(self):
        tested = self.GetMockEegSample(True)
        self.assertRaises(ValueError, tested.SplitEvenly, 0)

    def test_SplitEvenly_MoreSlicesThanRows(self):
        tested = self.GetMockEegSample(True)
        rowsNo = len(tested.dataFrame)
        self.assertRaises(ValueError, tested.SplitEvenly, rowsNo + 1)

        
class Test_Directory(unittest.TestCase):

    def test_Ctor(self):
        path = "Test"
        tested = eeg.Directory(path)
        self.assertEqual(tested.fullPath, path)
        
    def test_ctor_ThrowsWhenDirDoesntExist(self):
        with self.assertRaises(ValueError):
            eeg.Directory("D:\DirectoryThatDoesNotExist")
            
    def test_EnumerateFiles_WithDot(self):
        path = "Test\DirectoryForDirectoryClassTests"
        extension = ".csv"
        tested = eeg.Directory(path)
        actual = tested.EnumerateFiles(extension)
        expected = [os.path.join(path, f"fileB{extension}")]
        self.assertEqual(expected, actual)
        
    def test_EnumerateFiles_WithoutDot(self):
        path = "Test\DirectoryForDirectoryClassTests"
        extensionNoDot = "csv"
        tested = eeg.Directory(path)
        actual = tested.EnumerateFiles(extensionNoDot)
        expected = [os.path.join(path, f"fileB.{extensionNoDot}")]
        self.assertEqual(expected, actual)
        
    def test_GetMatchingFilesRecursive(self):
        path = "Test\DirectoryForDirectoryClassTests"
        extensionNoDot = "csv"
        tested = eeg.Directory(path)
        actual = tested.EnumerateFilesRecursive("*file*")
        expected = [os.path.join(path, "fileA.txt"), os.path.join(path, "fileB.csv"), os.path.join(path, "fileC.zip"), os.path.join(path, "ForRecursiveTests", "fileD.txt")]
        self.assertEqual(expected, actual)
        
    def test_EnumerateFilesRecursive(self):
        path = "Test\DirectoryForDirectoryClassTests"
        extensionNoDot = "csv"
        tested = eeg.Directory(path)
        actual = tested.EnumerateFilesRecursive("*file*")
        expected = [os.path.join(path, "fileA.txt"), os.path.join(path, "fileB.csv"), os.path.join(path, "fileC.zip"), os.path.join(path, "ForRecursiveTests", "fileD.txt")]
        self.assertEqual(expected, actual)

    def test_SplitAll(self):
        path = "D:\\Eeg\\Test\\asdasd"
        actual = eeg.Directory.SplitAll(path)
        expected = ["D:", "Eeg", "Test", "asdasd"]
        self.assertEqual(expected, actual)
      
class Test_ZipDirectory(unittest.TestCase):
  
    def test_Ctor(self):
        path = "Test"
        tested = eeg.ZipDirectory(path)
        self.assertEqual(tested.fullPath, path)
        
    def test_ctor_ThrowsWhenDirDoesntExist(self):
        with self.assertRaises(ValueError):
            eeg.ZipDirectory("D:\\DirectoryThatDoesNotExist")

class Test_EegDataApi(unittest.TestCase):
    
    def test_Ctor(self):
        path = "Test"
        tested = eeg.EegDataApi(path)
        self.assertEqual(tested.directoryHandle.fullPath, path)

    def test_Ctor_ThrowsWhenDirDoesntExist(self):
        with self.assertRaises(ValueError):
            eeg.EegDataApi("D:\\DirectoryThatDoesNotExist")
            
    def test_GetAverageBandpowers_NoFiltering(self):
        path = "Test"
        tested = eeg.EegDataApi(path)
        actual = tested.GetAverageBandpowers(None)
        expected = pd.DataFrame.from_csv("Test\\test_GetAverageBandpowers_input.csv")
        self.assertTrue(expected.sort_index(axis=1).equals(actual.sort_index(axis=1)))
        
    def test_GetAverageBandpowers_Filtered(self):
        path = "Test"
        tested = eeg.EegDataApi(path)
        actual = tested.GetAverageBandpowers(["Awake", "Sleep"])
        expected = pd.DataFrame.from_csv("Test\\test_GetAverageBandpowers_input.csv")
        #expected = expected[(expected.Condition not in ["RecoveryEyesClosed", "testCondition"])]
        expected = expected[(expected.Condition != "RecoveryEyesClosed") &  (expected.Condition !=  "testCondition")]
        self.assertTrue(expected.sort_index(axis=1).equals(actual.sort_index(axis=1)))

    def test_GetAverageBandpowers_Filtered_CaseInsensitive(self):
        path = "Test"
        tested = eeg.EegDataApi(path)
        actual = tested.GetAverageBandpowers(["awake", "sleep"])
        expected = pd.DataFrame.from_csv("Test\\test_GetAverageBandpowers_input.csv")
        #expected = expected[(expected.Condition not in ["RecoveryEyesClosed", "testCondition"])]
        expected = expected[(expected.Condition != "RecoveryEyesClosed") &  (expected.Condition !=  "testCondition")]
        self.assertTrue(expected.sort_index(axis=1).equals(actual.sort_index(axis=1)))
