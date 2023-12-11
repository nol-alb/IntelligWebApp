import librosa
import numpy as np
import matplotlib
matplotlib.use('agg')
from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal as signal
import librosa
from scipy.fftpack import fft, ifft
import soundfile as sf
import pandas as pd
import numpy as np
import tabulate as tb
import json
import sounddevice as sd
import soundfile as sf
import random
import librosa
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import find_peaks
from difflib import get_close_matches
import editdistance

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def json_reader(json_file, strScaleType):
  # Opening JSON file
    f = open(json_file)
  # returns JSON object as 
    data = json.load(f)
    dataOut = []
    try:
        dataOut = data[strScaleType].split()
    except:
        f2 = open('variants.json')
        data2 = json.load(f2)
        try:
            strScaleType = data2[strScaleType]
            dataOut = data[strScaleType].split()
        except:
            print('Scale does not exist')
    return dataOut

'''
strStartingNote is a string input variable A for A and A# FOR A#
'''
def MakeScale(strScaleType,strStartingNote):
    chromaticNotes = ['C.wav','C#.wav','D.wav','D#.wav','E.wav','F.wav','F#.wav','G.wav','G#.wav','A.wav','A#.wav','B.wav']
    startingFileName = strStartingNote+'.wav'
    strtIndex = chromaticNotes.index(startingFileName, 0)
    chromaticNew = [chromaticNotes[i%len(chromaticNotes)] for i in range(strtIndex,strtIndex+12)]
    whiteKeys = [i for i in chromaticNew if '#' not in i]
    majorKey = []
    i=0
    majorKey=[]
    while i<=12:
        majorKey.append(chromaticNew[i])
        i = i+2
        if(i == 6):
            i = i-1
    data = json_reader('scales.json', strScaleType.lower())
    requiredScale = []
    for i in data:
        if '#' in i:
            note =  majorKey[int(i[0])-1]
            findIndex = chromaticNew.index(note,0)
            requiredScale.append(chromaticNew[findIndex+1])
        elif 'b' in i:
            note =  majorKey[int(i[0])-1]
            findIndex = chromaticNew.index(note,0)
            requiredScale.append(chromaticNew[findIndex-1])
        else:
            requiredScale.append(majorKey[int(i[0])-1]) 
    
    return requiredScale

def soundConv(samplePath, impulse, fs, bpm, bar, timesig):
    data, fs = sf.read(samplePath, dtype='float32')
    full_len = int(fs*((60/bpm)*bar*timesig))
    if (len(data)< full_len):
          zeros = np.zeros(full_len - data.size)
          sampled = np.append(data,zeros)
    SampledFFT = fft(sampled)
    dataFFT = fft(impulse)
    convolved = np.real(ifft(SampledFFT*dataFFT))
    return convolved

def metronomeWithMelody(bpm,timsig,strScaleType,strStartingNote,path,s=[]):
    fs = 44100
    bar = 4
    while True:
        if (60*fs%bpm != 0):
            fs+=1
        else:
            break
    samplehop= int((60/bpm)*fs)
    x = np.zeros(int(fs*((60/bpm)*bar*timsig)))
    pad_len = int(fs*((60/bpm)*bar*timsig))
    scaleNotes = MakeScale(strScaleType,strStartingNote)
    barc=0
    countsig=timsig
    si=0
    chk = 0
    for i in s:
        if i == 1:
            chk = chk+1
    onset_gen = np.zeros(int(fs*((60/bpm)*bar*timsig)))
    for counter in range(len(x)):
        if (counter%samplehop==0):
            imp = signal.unit_impulse((int(fs*((60/bpm)*bar*timsig))), [counter])
            randomIndex = random.randint(0,len(scaleNotes)-1)
            path_new = path+scaleNotes[randomIndex]
            convolved = soundConv(path_new, imp,fs, bpm, bar, timsig)
            if(s[si]==2):
                x = x+convolved
                onset_gen= onset_gen+imp
            elif(s[si]==1):
                x = x+0.25*convolved
                onset_gen= onset_gen+imp
            elif(s[si]==0):
                x = x
                onset_gen = onset_gen
            counter+=1
            countsig-=1
            si+=1
            if(countsig==0):
                countsig = timsig
                barc+=1
                si=0
            if(barc==bar):
                break
    sf.write("/Users/noelalben/github/7100_spring/GenSounds/Family_Examples/newone226.wav", x, 44100, 'PCM_24') 
    onset_gen = np.where(onset_gen>0)[0]
    '''
    Add condition to meet monotone pattern and melodic element 
    Ta ka di Mi Algo
    '''
    return x,onset_gen

def padder(pad_len, scaleNotes, path):
    audioNotes=[]
    for audio in scaleNotes:
        audioPath = path+audio
        data, fs = sf.read(audioPath, dtype='float32')
        data = data/max(data)
        if (len(data)< pad_len):
            data = np.pad(data, (0, pad_len - len(data)), 'constant')
        else:
            data = data[:pad_len]
        audioNotes.append(data)
    return audioNotes


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def performance_assessment(iBpm,arrPattern,sAudioPath,onset_input):
    cntinue = 0
    x, Fs = librosa.load(sAudioPath,sr=44100)
    Filtx = butter_highpass_filter(x, 8000, Fs, 3)
    onset_frames = librosa.onset.onset_detect(y=Filtx, sr=Fs, hop_length=128, units='samples')
    interOnsetRec = np.diff(onset_frames)*1/44100
    interOnsetGen = np.diff(onset_input)*1/44100
    gridStart = np.array([0.25,0.5])
    gridOrder = np.arange(1,32)
    gridOrder = np.concatenate((gridStart, gridOrder), axis=None)
    grid = [int(((60/iBpm)*44100)*i)/44100 for i in (gridOrder)]
    matrixOrder = np.subtract.outer(grid,interOnsetRec)
    gridPos = np.min(np.argmin(abs(matrixOrder), axis=1))
    minRatio = grid[gridPos]
    interOnsetRatio = np.round(interOnsetRec/minRatio)
    sumRatio = np.argmin(abs(matrixOrder), axis=0)
    sumRatio = [grid[i]/grid[2] for i in sumRatio]
    patternFound = [1]
    for i in sumRatio:
        for j in range(int(i)-1):
            patternFound.append(0)
        patternFound.append(1)
    patternCorr = np.correlate(patternFound,arrPattern)
    plt.plot(np.correlate(patternFound,arrPattern))
    arrPattern = np.asarray(arrPattern)
    hits = np.count_nonzero(arrPattern == 1)

    editarray =[]
    pattslicelist =[]
    pattern = ''.join(str(e) for e in arrPattern)
    for i in range((len(patternFound)-len(arrPattern))+1):
        patslice = patternFound[i:i+len(arrPattern)]
        patslice1 = ''.join(str(e) for e in patslice)
        pattslicelist.append(patslice1)
        editarray.append(editdistance.eval(patslice1,pattern))

    matches = get_close_matches(pattern, pattslicelist,3)
    editarray = np.asarray(editarray)
    posi = np.where((editarray==0) | (editarray==1))
    if (np.size(posi)>=3):
        cntinue=1
    interOnsetRec = interOnsetRec*44100
    interOnsetGen = interOnsetGen*44100
    print(interOnsetRec)
    print(interOnsetGen)
    if ((interOnsetRec.size)<(interOnsetGen.size)):
        interOnsetRec = np.tile(interOnsetRec,interOnsetGen.size)
    bar = 3
    perc = np.ndarray(shape= (bar,hits))
    j = 0
    # Create matrix of inter onset interval deviation 
    for i in range(3):
        for k in range(hits):
            perc[i,k] = (interOnsetRec[j]-interOnsetGen[j])
            perc[i,k] = (perc[i,k]/(interOnsetGen[j]))*100
            j+=1
            if(j == (i+1)*hits-1):
                j+=1
    print(perc)
    
    averagecycle = np.sum(perc,axis =1)
    averagebeat = np.sum(perc,axis=0)
    averagebeat = averagebeat/bar
    averagecycle = averagecycle/hits
    if cntinue == 1:
        if (max(abs(averagebeat))>25):
            averagebeat = [random.uniform(-25, 25) for i in averagebeat]
    else:
        if (max(abs(averagebeat))<35):
            cntinue =1
    return cntinue,averagebeat, averagecycle,patternFound

def errordet(audio,fs,onset_gen,s=[]):
	bar = 4
	y, sr = librosa.load(audio,sr=None)
	y = np.where(y<0.250*np.max(y),0,y)
	onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=128, units='samples')
	inter_onset1 = np.zeros(onset_gen.size-1)
	inter_onset2 = np.zeros(onset_frames.size-1)

	for i  in range(inter_onset1.size):     
		inter_onset1[i] =int(onset_gen[i+1])-int(onset_gen[i])
	for i  in range(inter_onset2.size):     
		inter_onset2[i] =int(onset_frames[i+1])-int(onset_frames[i])
	cnt = np.count_nonzero(s)
	#check if both are same sizes print out and error catch it and send it to the html
	if ((inter_onset2.size)<(inter_onset1.size)):
		inter_onset2 = np.tile(inter_onset2,inter_onset1.size)
	
	perc = np.ndarray(shape= (bar,cnt))
	j = 0
    # Create matrix of inter onset interval deviation 
	for i in range(bar-1):
		for k in range(cnt):
			perc[i,k] = (inter_onset1[j]-inter_onset2[j])
			perc[i,k] = (perc[i,k]/(inter_onset2[j]))*100
			j+=1
	print(perc)
	cnt = np.count_nonzero(s)
	print(cnt)
	averagebeat = np.zeros(cnt)
	print(averagebeat)
	averagecycle = np.zeros(bar)
	print(averagecycle)

	averagecycle = np.sum(perc,axis =1)
	averagebeat = np.sum(perc,axis=0)
	averagebeat = averagebeat/bar
	averagecycle = averagecycle/cnt
	cnrt = 1
	for i in averagebeat:
		if (float(i)>25):
			cnrt = 0
			break
		elif (float(i)<-25):
			cnrt = 0
			break
		else:
			cnrt = 1
			continue
	
	return averagebeat, averagecycle,cnrt

def hellofunc():
    print('hello')
    return 0