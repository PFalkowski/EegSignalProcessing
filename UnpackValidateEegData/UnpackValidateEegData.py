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

class Directory:

    def __init__ (self, workingDir):
        self.workingDir = workingDir

    def EnumerateFiles(self, extension):
        if not os.path.isdir(self.workingDir) or not os.path.exists(self.workingDir):
            raise ValueError(f'Directory {dir} does not exist.')
        files = [join(self.workingDir, f) for f in listdir(self.workingDir) if f.endswith(extension) and isfile(join(self.workingDir, f))]
        return files

    def EnumerateFilesRecursive(self, extension):
        files = [y for x in os.walk(self.workingDir) for y in glob(os.path.join(x[0], f'*{extension}'))]
        return files

class ZipData:
    
    extension = '.zip'
    def __init__ (self, workingDir):
        self.workingDir = workingDir
        self.directoryHandle = Directory(workingDir)
        self.filePathsList = self.directoryHandle.EnumerateFiles(self.extension)

    def __ComputeFileSha256(self, fileName):
        hash = hashlib.sha256()
        with open(fileName, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        return hash.hexdigest()

    def ValidateFiles(self):
        hashDictionary = {}
        for file in self.filePathsList:
            hashDictionary[file] = self.__ComputeFileSha256(file)
        return hashDictionary

    def ExtractZipFile(self, fileName): 
        with zipfile.ZipFile(fileName, 'r') as zipObj:
            zipObj.extractall(self.workingDir)

    def ExtractAllFiles(self):
        for fileName in self.filePathsList:
            self.ExtractZipFile(fileName)

class EegData:
    
    extension = '.vhdr'
    def __init__  (self, workingDir):
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
zipHandle = ZipData(workingDirectory)
#filesHashes = zipHandle.ValidateFiles()
#zipHandle.ExtractAllFiles()

eegHandle = EegData(workingDirectory)
#eegHandle.GetSummary()


rawData = eegHandle.GetRawDataFromFile(eegHandle.filePathsList[0])
rawData.plot()
plt.show()