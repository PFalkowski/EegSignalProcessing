import mne
import matplotlib
import zipfile
import hashlib
import zipfile
import PyQt5
import os
import PyQt5
import matplotlib
import matplotlib.pyplot as plt
import mne
import pandas as pd
import re
import numpy as np
import datetime
import re
import time
from glob import glob
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
matplotlib.use('Qt5Agg')

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

    def __init__(self, fullFilePath):
        File.__init__(self, fullFilePath)    
        self.samplingRate = self.RawData().info["sfreq"]
        splittedFileName = self.nameWithoutExtension.split("_")
        self.subject = splittedFileName[0]
        self.session = splittedFileName[1]
        self.condition = splittedFileName[2]
    

    def RawData(self):
        rawData = mne.io.read_raw_brainvision(self.fullFilePath, preload=True, stim_channel=False, verbose = False)
        return rawData
    
    def AsDataFrame(self, withLabels = True):
        rawData = self.RawData()
        brain_vision = rawData.get_data().T
        df = pd.DataFrame(data=brain_vision, columns=rawData.ch_names)
        if (withLabels):             
            df["Subject"] = self.subject
            df["Session"] = self.session
            df["Condition"] = self.condition
            df["BinaryCondition"] = EegSample.BinaryCondition(self.condition)
            df["TernaryCondition"] = EegSample.TernaryCondition(self.condition)
        return df

    def SaveToCsv(self, fullNewFilePath, withLabels = True):
        if (fullNewFilePath is None):
           fullNewFilePath = os.path.join(self.pathWithoutFileName, f"{self.nameWithoutExtension}{'_labelled' if withLabels else ''} .csv")
        self.AsDataFrame(withLabels).to_csv(fullNewFilePath)

    def Plot(self):
        self.RawData().plot()
        plt.show()



class Directory:

    def __init__(self, fullPath):
        if not os.path.isdir(fullPath) or not os.path.exists(fullPath):
            raise ValueError(f'Directory {fullPath} does not exist.')
        self.fullPath = fullPath

    def EnumerateFiles(self, extension):
        return [join(self.fullPath, f) for f in listdir(self.fullPath) if f.endswith(extension) and isfile(os.path.join(self.fullPath, f))]

    def EnumerateFilesRecursive(self, pattern):
        files = [y for x in os.walk(self.fullPath) for y in glob(os.path.join(x[0], pattern))]
        return files
    
    @staticmethod
    def SplitAll(path):
        path = os.path.normpath(path)
        return path.split(os.sep)

class ZipDirectory(Directory):
    
    extension = '.zip'

    def __init__(self, fullPath):
        Directory.__init__(self, fullPath)
        self.filePathsList = self.EnumerateFiles(self.extension)

    def GetFilesSha256(self):
        hashDictionary = {}
        for fullFilePath in self.filePathsList:
            fileHandle = File(fullFilePath)
            hashDictionary[fileHandle.fullFilePath] = fileHandle.ComputeFileSha256()
        return hashDictionary

    def ExtractZipFile(self, fullFilePath): 
        with zipfile.ZipFile(fullFilePath, 'r') as zipObj:
            zipObj.extractall(self.fullPath)

    def ExtractAllFiles(self):
        for fullFilePath in tqdm(self.filePathsList):
            self.ExtractZipFile(fullFilePath)        

class Validator:

    def __init__(self, zipFolder, checksumFileHandle):
        self.ZipDirectory = zipFolder
        self.checksumFile = checksumFileHandle
        
    def Validate(self):
        result = {}
        expected = self.checksumFile.GetChecksumDictionary()
        files = self.zipFolder.filePathsList
        if (len(expected) != len(files)):
           raise ValueError("Invalid validation file")
       
        for filePath in tqdm(files):
            file = File(filePath)
            result[filePath] = file.Validate(expected[file.nameWithoutExtension])  
            
        return result

class EegSample:
    
    defaultEegBands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    
    @staticmethod
    def GenerateEegBands(step):
        i = 0
        d = {}
        while i < 45:
            d[f"{i}-{i+step}"] = (i, i+step)
            i=i+step
        return d 

    label_names = ["Subject", "Session", "Condition", "BinaryCondition", "TernaryCondition"]
    
    def __init__(self, dataFrame, samplingRate, subject, session, condition):
        self.samplingRate  = samplingRate
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        else:
            raise TypeError("only Pandas DataFrame can be input as ctor arg.  Use classmethod InitializeFromEegFile to initialie from EegFile")        
        self.subject = subject
        self.session = session
        self.condition = condition
        self.binaryCondition = EegSample.BinaryCondition(condition)
        self.ternaryCondition = EegSample.TernaryCondition(condition)

    @classmethod
    def InitializeFromEegFile(cls, eegFile):
        return cls(eegFile.AsDataFrame(True), eegFile.samplingRate, eegFile.subject, eegFile.session, eegFile.condition)

    def GetDataFrame(self, withLabels = True):
        df = self.dataFrame
        labelsExist = EegSample.DataFrameHasLabels(df)
        if (withLabels and labelsExist) or (not withLabels and not labelsExist):
            return df
        elif not withLabels and labelsExist:
            return EegSample.GetDfWithoutLabels(df)
        elif withLabels and not labelsExist:
            raise ValueError('Labels do not exist. Therefore, cannot return data frame with labels. Create EegSample using DataFrame with labels.')
    
    @staticmethod
    def DataFrameHasLabels(df, columnNames = label_names):
        return set(columnNames).issubset(df.columns)
    
    @staticmethod
    def GetDfWithoutLabels(df, columnNames = label_names):
        return df.drop(columnNames, axis = 1)
    
    @staticmethod
    def BinaryCondition(condition):
        if (re.search("Anesthetized", condition, re.IGNORECASE) or re.search("Sleeping", condition, re.IGNORECASE)):
            return "Unconscious"
        elif(re.search("Awake", condition, re.IGNORECASE)):
            return "Conscious"
    
    @staticmethod
    def TernaryCondition(condition):
        if (re.search("Anesthetized", condition, re.IGNORECASE) or re.search("Sleeping", condition, re.IGNORECASE)):
            return "Unconscious"
        elif(re.search("Awake", condition, re.IGNORECASE)):
            return "Conscious"
        else:
            return "InBetween"

    def GetChannel(self, channelName):        
        df = self.GetDataFrame(False)
        return df.loc[:,channelName]
    
    def GetRandomSubset(self, ratio, withLabels = True):
        df = self.GetDataFrame(withLabels)
        count = int(df.shape[0] * ratio)
        return df.sample(n=count)
    
    def __splitToSmallerDataFrames(self, slicesNo):
        if (slicesNo > len(self.dataFrame)):
            raise ValueError(f"Can't split into more slices than the length of the collection. Choose value lower than {len(self.dataFrame)}. Currently have {slicesNo}")
        return np.array_split(self.dataFrame, slicesNo)

    def SplitEvenly(self, slicesNo):
        slices = self.__splitToSmallerDataFrames(slicesNo)
        return [EegSample(e, self.samplingRate, self.subject, self.session, self.condition) for e in slices]

    ## Spectral analysis region
        #https://dsp.stackexchange.com/a/45662/43080
        #https://raphaelvallat.com/bandpower.html
        #https://stackoverflow.com/q/25735153/3922292
        #https://stackoverflow.com/a/52388007/3922292    

    def GetAverageBandpower(self, eegBands = None):    
        data = self.GetDataFrame(False)
        fft_vals = np.absolute(np.fft.rfft2(data))
        fft_freq = np.fft.rfftfreq(len(data), 1.0/self.samplingRate)

        result = dict()
        eegBands = self.defaultEegBands if eegBands is None else eegBands
        for band in eegBands:  
            freq_ix = np.where((fft_freq >= eegBands[band][0]) & 
                               (fft_freq < eegBands[band][1]))[0]
            result[band] = np.mean(fft_vals[freq_ix])

        return result
    
    def GetAverageBandpowerAsDataFrame(self, withLabels = False, eegBands = None):
        bandpowers = self.GetAverageBandpower(eegBands)
        df = pd.DataFrame(bandpowers, index=[0])
        if withLabels:
            df["Subject"] = self.subject
            df["Session"] = self.session
            df["Condition"] = self.condition
            df["BinaryCondition"] = EegSample.BinaryCondition(self.condition)
            df["TernaryCondition"] = EegSample.TernaryCondition(self.condition)
        return df
    
    def Fft(self):    
        df = self.GetDataFrame()
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
        for band in self.defaultEegBands:  
            freq_ix = np.where((fft_freq >= self.defaultEegBands[band][0]) & 
                               (fft_freq < self.defaultEegBands[band][1]))[0]
            result[band] = np.mean(fft_vals[freq_ix])

        return result

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
        
        data = self.GetDataFrame(False)
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
    
#Use this class directly, not the classes above.
class EegDataApi:
    
    checksumFilePattern = '*checksum*.txt'
    validator = None

    def __init__(self, workingDirectoryPath):
        self.directoryHandle = Directory(workingDirectoryPath)
        self.zipHandle = ZipDirectory(workingDirectoryPath)

    def LoadValidationFile(self, checksumFileFullPath):    
        self.checksumFileHandle = ChecksumFile(checksumFileFullPath)
        self.validator = Validator(self.zipHandle, self.checksumFileHandle)

    def UnzipAll(self):
        print("Unzipping...")
        self.zipHandle.ExtractAllFiles()
        print("Done Unzipping")

    def LoadValidationFileByConvention(self):    
        matchingFiles = self.directoryHandle.EnumerateFilesRecursive(self.checksumFilePattern)
        if (len(matchingFiles) > 0):
            checksumFileFullPath = matchingFiles[0]
            self.LoadValidationFile(checksumFileFullPath)
       
    def Validate(self):
        if self.validator is None:
           self.LoadValidationFileByConvention()        
        if self.validator is not None:
            validationResult = self.validator.Validate()
            for key, value in validationResult.items():
                print('%s - %s.'%(key,'valid' if value else 'invalid'))
        else:  
            print('Validation file conforming to pattern *checksum*.txt  not found. No validation will be done.')

    def PlotFile(self, fileName):
        fileName = File.GetPathWithNewExtension(fileName, ".vhdr")
        filePath = self.directoryHandle.EnumerateFilesRecursive(f"*{fileName}*")[0]
        fileHandle = EegFile(filePath)
        fileHandle.Plot()
                
    def SaveToCsv(self, vhdrFileFullPath, newFileFullPath, withLabels = True):
        filePath = self.directoryHandle.EnumerateFilesRecursive(vhdrFileFullPath)[0]
        fileHandle = EegFile(vhdrFileFullPath)
        fileHandle.SaveToCsv(newFileFullPath, withLabels)

    def GetAllVhdrFiles(self):
        return self.directoryHandle.EnumerateFilesRecursive("*.vhdr")
    
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
                
    def GetStratifiedSubset(self, ratio, conditionsFilter = None):
        allVhdrFiles = self.GetAllVhdrFiles()
        result = pd.DataFrame()
        for f in allVhdrFiles:
            eegFile = EegFile(f)
            sample = EegSample.InitializeFromEegFile(eegFile)
            if conditionsFilter is None or any(re.findall("|".join(conditionsFilter), sample.condition, re.IGNORECASE)):
                result = result.append(sample.GetRandomSubset(ratio, True))
        return result

    def SaveStratifiedSubsetToOneCsvFile(self, ratio, conditionsFilter = None):
        subset = self.GetStratifiedSubset(ratio, conditionsFilter)
        now = datetime.datetime.now()
        fullPathOfNewFile = os.path.join(self.directoryHandle.fullPath, f"StratifiedPartition_{ratio}_{now.day}-{now.month}-{now.hour}-{now.minute}-{now.second}.csv")
        subset.to_csv(fullPathOfNewFile)

    def GetAverageBandpowers(self, conditionsFilter = None, slicesPerSession = 1, customEegBands = None):
        allVhdrFiles = self.GetAllVhdrFiles()
        result = pd.DataFrame()
        for f in tqdm(allVhdrFiles):
            eegFile = EegFile(f)
            sample = EegSample.InitializeFromEegFile(eegFile)
            if conditionsFilter is None or any(re.findall("|".join(conditionsFilter), sample.condition, re.IGNORECASE)):
                slices = sample.SplitEvenly(slicesPerSession)
                for s in slices:
                    bandpowers = s.GetAverageBandpowerAsDataFrame(True, customEegBands)
                    result = result.append(bandpowers)
        return result    
        

    def ConstructBandpowersOutputFileName(self, rootPath, fileNameBase, conditionsFilter = None, slicesPerSession = 1, customEegBands = None):
        now = datetime.datetime.now()
        outputFilename = f"{fileNameBase}_{now.day}-{now.month}-{now.hour}-{now.minute}_{slicesPerSession}-{'str(len(EegSample.defaultEegBands))' if customEegBands is None else str(len(customEegBands))}{'' if conditionsFilter is None else ('_' + '+'.join(conditionsFilter))}.csv"
        fullPathOfNewFile = os.path.join(rootPath, outputFilename)

    def SaveAverageBandpowersToCsv(self, conditionsFilter = None, slicesPerSession = 1, customEegBands = None):
        fullPathOfNewFile = self.ConstructBandpowersOutputFileName(self.directoryHandle.fullPath, "Bandpowers", conditionsFilter, slicesPerSession, customEegBands)
        bandpowersDataset = self.GetAverageBandpowers(conditionsFilter, slicesPerSession, customEegBands)    
        bandpowersDataset.to_csv(fullPathOfNewFile)
        print(f"Output saved to {fullPathOfNewFile}")

if __name__ == '__main__':
    workingDirectory = 'D:\EEG' #<- put your zip archives along with checksum file here
    api = EegDataApi(workingDirectory)
    #api.UnzipAll()
    #api.Validate()
    #api.PlotFile("Sub01_Session0101_Anesthetized")
    #api.SaveStratifiedSubsetToOneCsvFile(0.1, ['Sleeping', 'Awake', 'Anesthetized'])
    #api.SaveAllToCsv()
    customBands = EegSample.GenerateEegBands(1)
    api.SaveAverageBandpowersToCsv(conditionsFilter = ["awake", "anesthetized"], slicesPerSession = 100, customEegBands = customBands) #<- this one may take a long time. The more slices, the less time it'll take, as FFT is O(n log n) in complexity