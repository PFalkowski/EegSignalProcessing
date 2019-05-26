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
import datetime
import re
import time
from tqdm import tqdm

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
    
    def FileNameWithoutExtension(self):
        return File.GetFileNameWithoutExtension(self.fullFilePath)

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
    

    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    def __init__(self, fullFilePath):
        File.__init__(self, fullFilePath)    
        self.samplingRate = self.RawData().info["sfreq"]
        #self.subject = self.Subject()
        #self.session = self.Session()
        #self.condition = self.Condition()
        #self.binaryCondition = self.BinaryCondition()
    
    def Subject(self):
        return self.FileNameWithoutExtension().split("_")[0]

    def Session(self):
        return self.FileNameWithoutExtension().split("_")[1]
    
    def Condition(self):
        return self.FileNameWithoutExtension().split("_")[2]

    def BinaryCondition(self):
        if (re.search("Anesthetized", self.Condition(), re.IGNORECASE) or re.search("Sleeping", self.Condition(), re.IGNORECASE)):
            return "Unconscious"
        elif(re.search("Awake", self.Condition(), re.IGNORECASE)):
            return "Conscious"
    
    def TernaryCondition(self):
        if (re.search("Anesthetized", self.Condition(), re.IGNORECASE) or re.search("Sleeping", self.Condition(), re.IGNORECASE)):
            return "Unconscious"
        elif(re.search("Awake", self.Condition(), re.IGNORECASE)):
            return "Conscious"
        else:
            return "InBetween"

    def RawData(self):
        rawData = mne.io.read_raw_brainvision(self.fullFilePath, preload=False, stim_channel=False, verbose = False)
        return rawData
    
    def AsDataFrame(self, withLabels = True):
        rawData = self.RawData()
        brain_vision = rawData.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=rawData.ch_names)
        if (withLabels):             
            df["Condition"] = self.Condition()
            df["BinaryCondition"] = self.BinaryCondition()
            df["TernaryCondition"] = self.TernaryCondition()
        return df

    def SaveToCsv(self, fullNewFilePath, withLabels = True):
        if (fullNewFilePath is None):
           fullNewFilePath = os.path.join(self.pathWithoutFileName, f"{self.nameWithoutExtension}{'_labelled' if withLabels else ''} .csv")
        self.AsDataFrame(withLabels).to_csv(fullNewFilePath)

    def Plot(self):
        self.RawData().plot()
        plt.show()

    def GetChannel(self, channelName):        
        df = self.AsDataFrame(False)
        return df.loc[:,channelName]
    
    def GetRandomSubset(self, ratio, withLabels = True):
        df = self.AsDataFrame(withLabels)
        count = int(df.shape[0] * ratio)
        return df.sample(n=count)

    ## Spectral analysis region
        #https://dsp.stackexchange.com/a/45662/43080
        #https://raphaelvallat.com/bandpower.html
        #https://stackoverflow.com/q/25735153/3922292
        #https://stackoverflow.com/a/52388007/3922292
        
        
    def Fft(self):    
        df = self.AsDataFrame()
        return np.abs(np.fft.rfft2(df))
    
    def PowerSpectralDensity(self):    
        return self.Fft() ** 2
    
    @staticmethod
    def PlotBands(eeg_bands):
        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = eeg_bands.keys()
        df['val'] = [eeg_bands[band] for band in eeg_bands]
        ax = df.plot.bar(x='band', y='val', legend=False)
        ax.set_xlabel("EEG Band")
        ax.set_ylabel("Mean Band Amplitude")
        plt.show()
        
    def GetAverageChannelBandpower(self, channelName):    
        data = self.GetChannel(channelName)
        fft_vals = np.absolute(np.fft.rfft(data))
        fft_freq = np.fft.rfftfreq(len(data), 1.0/self.samplingRate)

        result = dict()
        for band in self.eeg_bands:  
            freq_ix = np.where((fft_freq >= self.eeg_bands[band][0]) & 
                               (fft_freq < self.eeg_bands[band][1]))[0]
            result[band] = np.mean(fft_vals[freq_ix])

        return result


    def GetAverageBandpower(self, withLabels = True):    
        data = self.AsDataFrame(withLabels)
        fft_vals = np.absolute(np.fft.rfft2(data))
        fft_freq = np.fft.rfftfreq(len(data), 1.0/self.samplingRate)

        result = dict()
        for band in self.eeg_bands:  
            freq_ix = np.where((fft_freq >= self.eeg_bands[band][0]) & 
                               (fft_freq < self.eeg_bands[band][1]))[0]
            result[band] = np.mean(fft_vals[freq_ix])

        return result
    
    def GetAverageBandpowerAsDataFrame(self, withLabels = True):
        bandpowers = self.GetAverageBandpower(withLabels)
        s = pd.Series(bandpowers, name="testName")
        s.index.name = 'BandPower'
        s.reset_index()
        return s#, orient='index', columns = ["", "", ""]
    
    def makeSpectrum(self, E, dx, dy, upsample=10):
        zeropadded = np.array(E.shape) * upsample
        F = np.fft.fftshift(np.fft.fft2(E, zeropadded)) / E.size
        xf = np.fft.fftshift(np.fft.fftfreq(zeropadded[1], d=dx))
        yf = np.fft.fftshift(np.fft.fftfreq(zeropadded[0], d=dy))
        return (F, xf, yf)


    def extents(self, f):
        # Convert a vector into the 2-element extents vector imshow needs
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]


    def plotSpectrum(self):
        
        data = self.AsDataFrame()
        # Generate spectrum and plot
        spectrum, xf, yf = self.makeSpectrum(data, 0.001, 0.001)
        # Plot a spectrum array and vectors of x and y frequency spacings
        plt.figure()
        plt.imshow(abs(spectrum),
                   aspect="equal",
                   interpolation="none",
                   origin="lower",
                   extent=self.extents(xf) + self.extents(yf))
        plt.colorbar()
        plt.xlabel('f_x (Hz)')
        plt.ylabel('f_y (Hz)')
        plt.title('|Spectrum|')
        plt.show()




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
        for fullFilePath in tqdm(self.filePathsList):
            self.ExtractZipFile(fullFilePath)

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
        self.zipHandle.ExtractAllFiles()
        print("Done Unzipping")

    def LoadValidationFileByConvention(self):    
        matchingFiles = self.directoryHandle.GetMatchingFilesRecursive(self.checksumFilePattern)
        if (len(matchingFiles) > 0):
            checksumFileFullPath = matchingFiles[0]
            self.LoadValidationFile(checksumFileFullPath)
       
    def Validate(self):
        if self.validator is None:
           self.LoadValidationFileByConvention()        
        if self.validator is not None:
            validationResult = self.validator.Validate()
            for key, value in tqdm(validationResult.items()):
                print('%s - %s.'%(key,'valid' if value else 'invalid'))
        else:  
            print('Validation file conforming to pattern *checksum*.txt  not found. No validation will be done.')

    def PlotFile(self, fileName):
        fileName = File.GetPathWithNewExtension(fileName, ".vhdr")
        filePath = self.directoryHandle.GetMatchingFilesRecursive(f"*{fileName}*")[0]
        fileHandle = EegFile(filePath)
        fileHandle.Plot()
                
    def SaveToCsv(self, vhdrFileFullPath, newFileFullPath, withLabels = True):
        filePath = self.directoryHandle.GetMatchingFilesRecursive(vhdrFileFullPath)[0]
        fileHandle = EegFile(vhdrFileFullPath)
        fileHandle.SaveToCsv(newFileFullPath, withLabels)

    def GetAllVhdrFiles(self):
        return self.directoryHandle.GetMatchingFilesRecursive(f"*.vhdr")
    
    def __GetCsvConversionDict(self, newDirectorySuffix = "Csv"):
        allVhdrFiles = self.GetAllVhdrFiles()
        result = {}
        for f in allVhdrFiles:
            newFilePath = f.replace(self.directoryHandle.fullPath, self.directoryHandle.fullPath + newDirectorySuffix)
            newFilePath = File.GetPathWithNewExtension(newFilePath, ".csv")
            result[f] = newFilePath
        return result
    
    def SaveAllToCsv(self, withLabels = True):
        filesDictionary = self.__GetCsvConversionDict("CsvLabelled" if withLabels else "Csv")
        for key, value in tqdm(filesDictionary.items()):
            os.makedirs(File.GetPathWithoutFileName(value), exist_ok=True)
            self.SaveToCsv(key, value, withLabels)
                
    def GetStratifiedSubset(self, ratio, filterConditions = None):
        allVhdrFiles = self.GetAllVhdrFiles()
        result = pd.DataFrame()
        for f in allVhdrFiles:
            eegFile = EegFile(f)
            if filterConditions is None or any(c in eegFile.Condition() for c in filterConditions):
                result = result.append(eegFile.GetRandomSubset(ratio, True))
        return result

    def SaveStratifiedSubsetToOneCsvFile(self, ratio, filterConditions = None):
        subset = self.GetStratifiedSubset(ratio, filterConditions)
        now = datetime.datetime.now()
        fullPathOfNewFile = os.path.join(self.directoryHandle.fullPath, f"StratifiedPartition_{ratio}_{now.day}-{now.month}-{now.hour}-{now.minute}-{now.second}.csv")
        subset.to_csv(fullPathOfNewFile)
        
    def GetAverageBandpowersLabelled(self, filterConditions = None):
        allVhdrFiles = self.GetAllVhdrFiles()
        result = pd.DataFrame()
        for f in tqdm(allVhdrFiles):
            eegFile = EegFile(f)
            bandpowers = eegFile.GetAverageBandpowerAsDataFrame()
            bandpowers["Condition"] = eegFile.Condition()
            bandpowers["BinaryCondition"] = eegFile.BinaryCondition()
            bandpowers["TernaryCondition"] = eegFile.TernaryCondition()
            result = result.append(bandpowers)
        return result

    def SaveAverageBandpowersLabelled(self):
        bandpaowersDataset = self.GetAverageBandpowersLabelled()        
        now = datetime.datetime.now()
        fullPathOfNewFile = os.path.join(self.directoryHandle.fullPath, f"AverageBandpowersLabelled_{now.day}-{now.month}-{now.hour}-{now.minute}-{now.second}.csv")
        bandpaowersDataset.to_csv(fullPathOfNewFile)

#usage
workingDirectory = 'D:\EEG Test' #<- put your zip archives along with checksum file here
api = EegDataApi(workingDirectory)
#api.UnzipAll()
api.Validate()
#api.PlotFile("Sub01_Session0101")
#api.SaveStratifiedSubsetToOneCsvFile(0.1, ['Sleeping', 'Awake', 'Anesthetized'])
api.SaveAllToCsv()
#api.SaveAverageBandpowersLabelled() #<- this one takes at least 10 hours for whole Neurotycho 100Hz dataset