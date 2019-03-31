import mne
import matplotlib
import zipfile
import os
import hashlib
import zipfile
import PyQt5
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

inputDir = 'E:\EEG Data'
outputDir = 'E:\EEG Data'
zipExtension = '.zip'

def get_files_from_directory(dir, extension):
    if not os.path.isdir(dir) or not os.path.exists(dir):
        raise ValueError(f'Directory {dir} does not exist.')
    files = [join(dir, f) for f in listdir(dir) if f.endswith(extension) and isfile(join(dir, f))]
    return files

def ComputeFileSha256(fileName):
    hash = hashlib.sha256()
    with open(fileName, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash.hexdigest()

def ValidateFiles(fileNamesList):
    hashDictionary = {}
    for fileName in fileNamesList:
        hashDictionary[fileName] = ComputeFileSha256(fileName)
    return hashDictionary

def ExtractZipArchive(fileName, directory_to_extract_to): 
    with zipfile.ZipFile(fileName, 'r') as zipObj:
        zipObj.extractall(directory_to_extract_to)

def ExtractAllFiles(fileNamesList, directory_to_extract_to):
    for fileName in fileNamesList:
        ExtractZipArchive(fileName, directory_to_extract_to)

def GetRawDataFrom(filePath):
    raw_data = mne.io.read_raw_brainvision(filePath, preload=True, stim_channel=False)
    numpy_array = raw_data._data
    channel_list = raw_data.ch_names
    return raw_data

    

#files = get_files_from_directory(inputDir, zipExtension)
#filesHashes = ValidateFiles(files)
#ExtractAllFiles(files, outputDir)


path = "E:\EEG Data\Sub01\Session010\Sub01_Session0101_AnestheticInjection.vhdr"
rawData = GetRawDataFrom(path)
print(rawData.info)
rawData.plot()
plt.show()