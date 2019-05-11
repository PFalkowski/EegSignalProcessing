import mne
import matplotlib
import zipfile
import hashlib
import zipfile
import PyQt5
import os
from glob import glob
from os import listdir
from os.path import isfile, join
import PyQt5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne
import pandas as pd
import re
import numpy as np

class File:

    def __init__(self, fullFilePath):
        if not os.path.isfile(fullFilePath) or not os.path.exists(fullFilePath):
            raise ValueError(f'File {fullFilePath} does not exist.')
        self.fullFilePath = fullFilePath
        self.nameWithoutExtension = File.GetFileNameWithoutExtension(fullFilePath)
        self.pathWithoutFileName = File.GetPathWithoutFileName(fullFilePath)
        
    @staticmethod
    def GetFileNameWithoutExtension(fullFilePath):
        return os.path.splitext(os.path.basename(fullFilePath))[0]

    @staticmethod
    def GetPathWithNewExtension(fullFilePath, newExtension):
        return os.path.splitext(fullFilePath)[0] + newExtension

    @staticmethod
    def GetPathWithoutFileName(fullFilePath):
        return os.path.dirname(fullFilePath)
    
    def ComputeFileSha256(self):
        hash = hashlib.sha256()
        with open(self.fullFilePath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        return hash.hexdigest()

    def Validate(self, checksum):
        actual = str.casefold(self.ComputeFileSha256())
        expected = str.casefold(checksum)
        isValid = actual == expected
        return isValid

    def GetAllLines(self):
        content = []
        with open(self.fullFilePath) as f:
            content = f.readlines()
        return [x.strip() for x in content] 

class ChecksumFile(File):
    
   def __init__(self, fullFilePath):
       File.__init__(self, fullFilePath)

   def GetChecksumDictionary(self):
        lines = self.GetAllLines()
        dict = {}
        for line in lines:
            split = line.split()
            dict[split[0]] = split[2]
        return dict 

class EegFile(File):
    
    def __init__(self, fullFilePath, samplingRate):
        File.__init__(self, fullFilePath)    
        self.samplingRate = samplingRate
        
    def RawData(self):
        rawData = mne.io.read_raw_brainvision(self.fullFilePath, preload=True, stim_channel=False)
        return rawData

    def AsDataFrame(self):
        rawData = self.RawData()
        brain_vision = rawData.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=rawData.ch_names)
        return df

    def SaveToCsv(self):
        self.AsDataFrame().to_csv(os.path.join(self.pathWithoutFileName, f"{self.nameWithoutExtension}.csv"))

    def Plot(self):
        self.RawData().plot()
        plt.show()
        
    def GetAverageBandpower(self):    

        #https://dsp.stackexchange.com/a/45662/43080
        #https://raphaelvallat.com/bandpower.html

        data = self.AsDataFrame()
        fft_vals = np.absolute(np.fft.rfft(data))
        fft_freq = np.fft.rfftfreq(len(data), 1.0/self.samplingRate)

        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}

        result = dict()
        for band in eeg_bands:  
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            result[band] = np.mean(fft_vals[freq_ix])

        return result

class Directory:

    def __init__(self, fullPath):
        if not os.path.isdir(fullPath) or not os.path.exists(fullPath):
            raise ValueError(f'Directory {fullPath} does not exist.')
        self.fullPath = fullPath

    def EnumerateFiles(self, extension):
        return [join(self.fullPath, f) for f in listdir(self.fullPath) if f.endswith(extension) and isfile(os.path.join(self.fullPath, f))]

    def GetMatchingFilesRecursive(self, pattern):
        return [y for x in os.walk(self.fullPath) for y in glob(os.path.join(x[0], pattern))]

    def EnumerateFilesRecursive(self, extension):
        return self.GetMatchingFilesRecursive(f'*{extension}')
    
    @staticmethod
    def SplitAll(path):
        path = os.path.normpath(path)
        return path.split(os.sep)

class ZipData:
    
    extension = '.zip'

    def __init__(self, directoryHandle):
        self.directoryHandle = directoryHandle
        self.filePathsList = self.directoryHandle.EnumerateFiles(self.extension)

    def GetFilesSha256(self):
        hashDictionary = {}
        for fullFilePath in self.filePathsList:
            fileHandle = File(fullFilePath)
            hashDictionary[fileHandle.fullFilePath] = fileHandle.ComputeFileSha256()
        return hashDictionary

    def ExtractZipFile(self, fullFilePath): 
        with zipfile.ZipFile(fullFilePath, 'r') as zipObj:
            zipObj.extractall(self.directoryHandle.fullPath)

    def ExtractAllFiles(self):
        count = 0
        for fullFilePath in self.filePathsList:
            self.ExtractZipFile(fullFilePath)
            ++count
        return count

class Validator:

    def __init__(self, zipData, checksumFileHandle):
        self.zipData = zipData
        self.checksumFile = checksumFileHandle
        
    def Validate(self):
        result = {}
        expected = self.checksumFile.GetChecksumDictionary()
        files = self.zipData.filePathsList
        if (len(expected) != len(files)):
           raise ValueError("Invalid validation file")
       
        for filePath in files:
            file = File(filePath)
            result[filePath] = file.Validate(expected[file.nameWithoutExtension])  
            
        return result
        
class EegData:
    
    extension = '.vhdr'

    def __init__(self, directoryHandle):
        self.directoryHandle = directoryHandle
        self.filePathsList = self.directoryHandle.EnumerateFilesRecursive(self.extension)
        self.dataDictionary = {}

    def LoadDataFromAllFiles(self):        
        for filePath in self.filePathsList:
            self.dataDictionary[filePath] = self.RawData(filePath)
        return self.dataDictionary
    
#Use this class directly, not the classes above.
class EegDataApi:
    
    checksumFilePattern = '*checksum*.txt'
    validator = None

    def __init__(self, workingDirectoryPath):
        self.directoryHandle = Directory(workingDirectory)
        self.zipHandle = ZipData(self.directoryHandle)
        self.eegHandle = EegData(self.directoryHandle)

    def LoadValidationFile(self, checksumFileFullPath):    
        self.checksumFileHandle = ChecksumFile(checksumFileFullPath)
        self.validator = Validator(self.zipHandle, self.checksumFileHandle)

    def UnzipAll(self):
        print("Unzipping...")
        count = self.zipHandle.ExtractAllFiles()
        print(f"/rUnzipped sucessfully {count} archives")

    def LoadValidationFileByConvention(self):    
        checksumFileFullPath = self.directoryHandle.GetMatchingFilesRecursive(self.checksumFilePattern)[0]
        self.LoadValidationFile(checksumFileFullPath)
       
    def Validate(self):
        if self.validator is None:
           self.LoadValidationFileByConvention()
        validationResult = self.validator.Validate()
        for key, value in validationResult.items():
            print('%s - %s.'%(key,'valid' if value else 'invalid'))

    def PlotFile(self, fileName):
        fileName = File.GetPathWithNewExtension(fileName, ".vhdr")
        filePath = self.directoryHandle.GetMatchingFilesRecursive(f"*{fileName}*")[0]
        fileHandle = EegFile(filePath)
        fileHandle.Plot()

    def SaveToCSV(self, vhdrFileFullPath, newFileFullPath):
        filePath = self.directoryHandle.GetMatchingFilesRecursive(vhdrFileFullPath)[0]
        fileHandle = EegFile(vhdrFileFullPath)
        df = fileHandle.AsDataFrame()
        df.to_csv(newFileFullPath)

    def GetAllVhdrFiles(self):
        return self.directoryHandle.GetMatchingFilesRecursive(f"*.vhdr")
    
    def __GetCsvConversionDict(self):
        allVhdrFiles = api.GetAllVhdrFiles()
        result = {}
        for f in allVhdrFiles:
            newFilePath = f.replace(self.directoryHandle.fullPath, self.directoryHandle.fullPath + "Csv")
            newFilePath = File.GetPathWithNewExtension(newFilePath, ".csv")
            result[f] = newFilePath
        return result

    def ConvertAllToCsv(self):
        filesDictionary = self.__GetCsvConversionDict()
        for key, value in filesDictionary.items():
            os.makedirs(File.GetPathWithoutFileName(value), exist_ok=True)
            self.SaveToCSV(key, value)



#usage
workingDirectory = 'D:\EEG' #<- put your zip archives along with checksum file here
api = EegDataApi(workingDirectory)
#api.UnzipAll()
#api.Validate()
#api.PlotFile("Sub01_Session0101")
#api.ConvertAllToCsv()