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

class File:

    def __init__(self, fullFilePath):
        if not os.path.isfile(fullFilePath) or not os.path.exists(fullFilePath):
            raise ValueError(f'File {fullFilePath} does not exist.')
        self.fullFilePath = fullFilePath
        self.nameWithoutExtension = File.GetFileNameWithoutExtension(fullFilePath)

    @staticmethod
    def GetFileNameWithoutExtension(fullFilePath):
        return os.path.splitext(os.path.basename(fullFilePath))[0]

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

class Directory:

    def __init__(self, workingDir):
        if not os.path.isdir(workingDir) or not os.path.exists(workingDir):
            raise ValueError(f'Directory {workingDir} does not exist.')
        self.workingDir = workingDir

    def EnumerateFiles(self, extension):
        return [join(self.workingDir, f) for f in listdir(self.workingDir) if f.endswith(extension) and isfile(join(self.workingDir, f))]

    def GetMatchingFilesRecursive(self, pattern):
        return [y for x in os.walk(self.workingDir) for y in glob(os.path.join(x[0], pattern))]

    def EnumerateFilesRecursive(self, extension):
        return self.GetMatchingFilesRecursive(f'*{extension}')

class ZipData:
    
    extension = '.zip'

    def __init__(self, workingDir):
        self.workingDir = workingDir
        self.directoryHandle = Directory(workingDir)
        self.filePathsList = self.directoryHandle.EnumerateFiles(self.extension)

    def GetFilesSha256(self):
        hashDictionary = {}
        for fullFilePath in self.filePathsList:
            fileHandle = File(fullFilePath)
            hashDictionary[fileHandle.fullFilePath] = fileHandle.ComputeFileSha256()
        return hashDictionary


    def ExtractZipFile(self, fullFilePath): 
        with zipfile.ZipFile(fullFilePath, 'r') as zipObj:
            zipObj.extractall(self.workingDir)

    def ExtractAllFiles(self):
        for fullFilePath in self.filePathsList:
            self.ExtractZipFile(fullFilePath)

class Validator:


    def __init__(self, zipData, checksumFileFullPath):
        self.zipData = zipData
        self.checksumFile = ChecksumFile(checksumFileFullPath)
        
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

    def __init__(self, workingDir):
        self.workingDir = workingDir
        self.directoryHandle = Directory(workingDir)
        self.filePathsList = self.directoryHandle.EnumerateFilesRecursive(self.extension)

    def GetRawDataFromFile(self, filePath):
        raw_data = mne.io.read_raw_brainvision(filePath, preload=True, stim_channel=False)
        return raw_data
    
    def GetRawDataFromAllFiles(self):
        dictionary = {}
        for filePath in self.filePathsList:
            dictionary[filePath] = self.GetRawDataFromFile(filePath)
        return dictionary
            
    def GetSummary(self):
        dictionary = {}
        for filePath in self.filePathsList:
            dictionary[filePath] = self.GetRawDataFromFile(filePath).info
        return dictionary

#usage
workingDirectory = 'E:\EEG Data'
checksumFilePattern = '*checksum*.txt'
dir = Directory(workingDirectory)
checksumFileName = dir.GetMatchingFilesRecursive(checksumFilePattern)[0]
zipHandle = ZipData(workingDirectory)
validator = Validator(zipHandle, checksumFileName)
validationResult = validator.Validate()
zipHandle.ExtractAllFiles()
eegHandle = EegData(workingDirectory)

#print example chart 
rawData = eegHandle.GetRawDataFromFile(eegHandle.filePathsList[0])
rawData.plot()
plt.show()