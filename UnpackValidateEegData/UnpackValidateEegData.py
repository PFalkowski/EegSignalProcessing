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
       super(ChecksumFile, self).__init__(fullFilePath)

   def GetChecksumDictionary(self):
        lines = self.GetAllLines()
        dict = {}
        for line in lines:
            split = line.split()
            dict[split[0]] = split[2]
        return dict 

class EegFile(File):
    
    def __init__(self, fullFilePath):
        super(ChecksumFile, self).__init__(fullFilePath)       

    def AsDataFrame(self):
        raw_data = mne.io.read_raw_brainvision(self.fullFilePath, preload=False, stim_channel=False)
        brain_vision = raw_data.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=raw_data.ch_names)
        return df

    def SaveToCsv(self):
        self.AsDataFrame().to_csv(os.path.join(self.pathWithoutFileName, f"{self.nameWithoutExtension}.csv"))

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

    def GetRawDataFromFile(self, filePath):
        raw_data = mne.io.read_raw_brainvision(filePath, preload=True, stim_channel=False)
        return raw_data
    
    def LoadDataFromAllFiles(self):        
        for filePath in self.filePathsList:
            self.dataDictionary[filePath] = self.GetRawDataFromFile(filePath)
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
        filePath = self.directoryHandle.GetMatchingFilesRecursive(f"*{fileName}*.vhdr")[0]
        rawData = self.eegHandle.GetRawDataFromFile(filePath)
        rawData.plot()
        plt.show()


#usage
workingDirectory = 'D:\EEG'
api = EegDataApi(workingDirectory)
api.UnzipAll()
api.Validate()
api.PlotFile("Sub01_Session0101")