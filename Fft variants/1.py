#https://stackoverflow.com/q/25735153/3922292

    def test_Fft(self):
        eegFile = eeg.EegFile("Test/100HzTest.vhdr")
        data = eegFile.GetChannel("ECoG_ch001")
        plt.plot(data)
        plt.show()
        ## Perform FFT WITH SCIPY
        signalFFT = np.fft.rfft(data)

        ## Get Power Spectral Density
        signalPSD = signalFFT ** 2

        ## Get frequencies corresponding to signal PSD
        fftFreq = np.fft.rfftfreq(len(data), 1.0/eegFile.samplingRate)

        ## Get positive half of frequencies
        #i = fftFreq > 0

        ##
        plt.figurefigsize=(8,4)
        plt.plot(fftFreq, 10*np.log10(signalPSD))
        #plt.xlim(0, 100);
        plt.xlabel('Frequency Hz')
        plt.ylabel('Power Spectral Density (dB)')
        plt.show()
        print('duh')