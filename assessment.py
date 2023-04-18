import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from scipy.signal import welch
from scipy.signal import welch, medfilt
from scipy.optimize import minimize_scalar
import math
from scipy.ndimage import gaussian_filter1d
from textdistance import levenshtein, damerau_levenshtein, jaro, jaro_winkler
from scipy.signal import find_peaks, peak_widths

"""
The Needleman-Wunsch Algorithm
==============================
This is a dynamic programming algorithm for finding the optimal alignment of
two strings.
Example
-------
    >>> x = "GATTACA"
    >>> y = "GCATGCU"
    >>> print(nw(x, y))
    G-ATTACA
    GCA-TGCU
LICENSE
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
"""

import numpy as np

def nw(x, y, match = 1, mismatch = 1, gap = 1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    srx = ''.join(rx)[::-1]
    sry = ''.join(ry)[::-1]
    return '\n'.join([srx, sry]), F,srx,sry


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def onset_postproc(onset_frames,x,Fs,n_bins,hop_length,med_len):
    Cqt = np.abs(librosa.cqt(x, sr=Fs ,n_bins=n_bins, hop_length=hop_length))
    energy = np.sqrt(np.sum(np.square(Cqt[int(np.shape(Cqt)[0]/2):]), axis=0))
    energy_smooth = gaussian_filter1d(energy, 8)
    local_average = np.convolve(energy_smooth, np.ones(4) / 4, mode='same')
    threshold_local = filters.median_filter(local_average, med_len)
    
    peaks, _ = find_peaks(energy_smooth,threshold_local,distance=5,prominence = 0.1)
    
    widths = peak_widths(local_average, peaks, rel_height=0.5)
    #Extract left side of the width
    new_peaks = (widths[0]/2).astype(int)
    #Backtrack and convert from frames to samples
    backtrack_peaks = (peaks-new_peaks)*hop_length
    
    #Remove spurious onsets
    onset_frames_out = onset_frames
    #onset_frames_out = [onset_frames[np.argmin(np.abs(onset_frames - val))] for val in backtrack_peaks]
    return np.unique(onset_frames_out),backtrack_peaks


def noise_floor(x,Fs):
    #Welch psd estimation
    f, Pxx = welch(x, Fs, nperseg=256)
    # Calculate the noise floor (in dB)
    filtered_Pxx = medfilt(Pxx, kernel_size=5)
    noise_floor = 10 * np.log10(np.mean(filtered_Pxx))

    return noise_floor
    
def ExtractOnsets(sRefAudioPath,sPerfAudioPath):
    xPerf, Fs = librosa.load(sPerfAudioPath,sr=44100)
    FiltxPerf = butter_highpass_filter(xPerf, 8000, Fs, 3)
    bNoisyPerf = True if noise_floor(xPerf,Fs)>(-65) else False
    Onset_frames_Perf = librosa.onset.onset_detect(y=FiltxPerf, sr=Fs, hop_length=128, units='samples')
    xRef, Fs = librosa.load(sRefAudioPath,sr=44100)    
    FiltxRef = butter_highpass_filter(xRef, 8000, Fs, 3)
    Onset_frames_Ref = librosa.onset.onset_detect(y=FiltxRef, sr=Fs, hop_length=128, units='samples')
    bNoisyRef = True if noise_floor(xRef,Fs)>(-65) else False
    if(bNoisyRef == True):
        Onset_frames_Ref,back_track_ref = onset_postproc(Onset_frames_Ref,xRef,Fs,100,128,30)
    if(bNoisyPerf == True):
        Onset_frames_Perf,back_track_perf = onset_postproc(Onset_frames_Perf,xPerf,Fs,100,128,30)
    #Onset_frames_Ref,back_track_ref = onset_postproc(Onset_frames_Ref,xRef,Fs,100,128,30)
    #Onset_frames_Perf,back_track_perf = onset_postproc(Onset_frames_Perf,xPerf,Fs,100,128,30)
    return Onset_frames_Ref, Onset_frames_Perf

def ExtractOnsets_wRef(sAudioPath):
    xPerf, Fs = librosa.load(sAudioPath,sr=44100)
    FiltxPerf = butter_highpass_filter(xPerf, 8000, Fs, 3)
    bNoisyPerf = True if noise_floor(xPerf,Fs)>(-65) else False
    Onset_frames_Perf = librosa.onset.onset_detect(y=FiltxPerf, sr=Fs, hop_length=128, units='samples')
    if(np.size(Onset_frames_Perf)>np.size(Onset_frames_Perf) or bNoisyPerf):
        Onset_frames_Perf = onset_postproc(Onset_frames_Perf,xPerf,Fs,100,128,30)
    return Onset_frames_Perf
    
def optimal_grid(interOnset):   
    def cost_function(X):
        gridOrder = np.arange(1, 32)
        grid = [int(((60 / X) * 44100) * i) / 44100 for i in gridOrder]

        return np.min(abs(np.subtract.outer(grid, interOnset)), axis=0).sum()
    res = minimize_scalar(cost_function, bounds=(0, 450), method='bounded')

    optimal_bpm = res.x
    gridStart = np.array([0.25, 0.5])
    gridOrder = np.arange(1, 32)
    gridOrder = np.concatenate((gridStart, gridOrder), axis=None)
    grid = [int(((60 / optimal_bpm) * 44100) * i) / 44100 for i in gridOrder]
    return grid,optimal_bpm

def find_pattern(grid,interOnset):
    matrixOrder = np.subtract.outer(grid, interOnset)
    sumRatio = np.argmin(abs(matrixOrder), axis=0)
    sumRatio = [grid[i]/grid[2] for i in sumRatio]
    pattern = [1]
    for i in sumRatio:
        for j in range(int(i)-1):
            pattern.append(0)
        pattern.append(1)
    return pattern,sumRatio
    
def next_power_of_two(n):
    return 2**math.ceil(math.log2(n)) 

def use_backtrack(back_track_perf, back_track_ref):
    interOnsetPerf = np.diff(np.unique(back_track_perf))*1/44100
    interOnsetRef = np.diff(np.unique(back_track_ref))*1/44100
    gridPerf,perBpm = optimal_grid(interOnsetPerf)
    patternPerf,sumRatioPerf = find_pattern(gridPerf,interOnsetPerf)
    gridRef,refBpm = optimal_grid(interOnsetRef)
    patternRef,sumRatioRef = find_pattern(gridRef,interOnsetRef)
    spatternPerf = ''.join(str(o) for o in patternRef)
    spatternRef  = ''.join(str(o) for o in patternPerf)
    align, cost,srx,sry = nw(spatternPerf, spatternRef)
    dlev_dist = damerau_levenshtein(srx,sry)
    return srx,sry,dlev_dist

def wrefAsessment(sPerfAudioPath, patternRef,savepath):
    Onset_frames_Perf = ExtractOnsets_wRef(sPerfAudioPath)
    interOnsetPerf = np.diff(np.unique(Onset_frames_Perf))*1/44100
    gridPerf,perBpm = optimal_grid(interOnsetPerf)
    patternPerf,sumRatioPerf = find_pattern(gridPerf,interOnsetPerf)
    cntinue,matches,editarray = edit_distance_check(patternPerf, patternRef)
    patternPerf = patternExpander(matches[0])
    print(patternPerf)
    print(patternRef)
    PatternErrorVisualizer(patternRef,patternPerf,savepath)
    
    return cntinue
    
    

def convert_to_binary(sRefAudioPath,sPerfAudioPath,bRepeat):
    Onset_frames_Ref, Onset_frames_Perf = ExtractOnsets(sRefAudioPath,sPerfAudioPath)
    interOnsetPerf = np.diff(np.unique(Onset_frames_Perf))*1/44100
    interOnsetRef = np.diff(np.unique(Onset_frames_Ref))*1/44100
    gridPerf,perBpm = optimal_grid(interOnsetPerf)
    patternPerf,sumRatioPerf = find_pattern(gridPerf,interOnsetPerf)
    gridRef,refBpm = optimal_grid(interOnsetRef)
    patternRef,sumRatioRef = find_pattern(gridRef,interOnsetRef)
    # pad not necessary with needleman algorithm 
    patternLength = [len(patternRef) if len(patternRef)>=len(patternPerf) else len(patternPerf)]
    lengthFull = next_power_of_two(patternLength[0])
    padpatternPerf = np.pad(patternPerf, (0,lengthFull-len(patternPerf)), 'constant')
    padpatternRef = np.pad(patternRef, (0,lengthFull-len(patternRef)), 'constant')
    return padpatternPerf, padpatternRef, Onset_frames_Ref, Onset_frames_Perf, patternPerf, patternRef

def isochrone(ratio,bpm,timsig,s=[]):
	fs = 44100
	bar = 1
	while True:
		if (60*fs%bpm != 0):
			fs+=1
		else:
			break

	samplehop= int(((60/bpm)*fs)*ratio)
	x = np.zeros(int((samplehop*bar*timsig)))
	barc=0
	countsig=timsig
	si=0
	for counter in range(len(x)):
		if (counter%samplehop==0):
            #Create Impulses at positon([Counter]) and of Length (SampleHop*TimeSig*Bars)
			imp = signal.unit_impulse((int((samplehop)*bar*timsig)), [counter])
            #Add impulses to a zero signal to generate pulse train
			if (s[si]==1):
				if(countsig==timsig):
					x = x+imp
	                
				else:
					x = x+imp
			elif (s[si]==0):
				x = x
	            
	                 
	            
	                
			counter+=1
			countsig-=1
			si+=1
			if(countsig==0):
				countsig = timsig
				barc+=1
				si=0
				if(barc==bar):
					break
    
	return x,np.where(x>0)[0]

def edit_distance_check(patternFound, arrPattern):
    cntinue = 0
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
    return cntinue,matches,editarray



def performance_assessment(patternPerf, patternRef, Onset_frames_Ref, Onset_frames_Perf,bRepeat):
    beat_difference = np.count_nonzero(patternRef)-np.count_nonzero(patternPerf)
    if(bRepeat == True):
        correct,matches,editarray = edit_distance_check(patternPerf, patternRef)
    else:
        spatternPerf = ''.join(str(o) for o in a)
        spatternRef  = ''.join(str(o) for o in b)
        jaro_winkler(spatternPerf,spatternRef)
        dlev_dist = damerau_levenshtein(spatternPerf,spatternRef)
        lev_dist  = levenshtein(spatternPerf,spatternRef)
        align, cost,srx,sry = nw(spatternPerf, spatternRef)
        
    return srx,sry

def patternExpander(strPattern):
    listPattern = [int(num) for num in list(strPattern)]
    return listPattern

def PatternErrorVisualizer(pattern,patternmain,savepath):
    plt.figure()
    isochrone = np.arange(0,(16*4)+8)
    patternPos = []
    for i,k in enumerate(pattern):
        if(k == 1):
            if((i*4)+4 == isochrone.size):
                patternPos.append((i*4 - 1)+4)
            else:
                patternPos.append((i*4)+4)
        
    patternPos2 = []
    
    for i,k in enumerate(patternmain):
        if(k == 1):
            if((i*4)+4 == isochrone.size):
                patternPos2.append((i*4 - 1)+4)
            else:
                patternPos2.append((i*4)+4)
    
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    labels=[]
    j=1
    for i in pattern:
        if(i==1):
            labels.append("Hit"+str(j))
            j = j+1
    print(len(labels))
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    scats = np.empty(len(isochrone))

    scats[:] = np.nan

    for i in range(len(patternPos)):
        scats[patternPos[i]] = 0.5
        cmap =colors[i%len(colors)]
        lo = plt.scatter(isochrone,scats, s=200, marker='x',color= colors[i%len(colors)],alpha = 1)
        print(cmap)
        scats[patternPos[i]] = np.nan
    for i in range(len(patternPos2)):
        scats[patternPos2[i]] = 0.5
        cmap =colors[i%len(colors)]
        print(cmap)
        li = plt.scatter(isochrone,scats, s=50, marker='o',color= colors[i%len(colors)], alpha = 1)
        scats[patternPos2[i]] = np.nan
    try:
        plt.legend((lo, li),
           ('Expected Hit', 'Your Performance'),
           scatterpoints=1,
           loc='upper right',
           facecolor="gray",
           fontsize=15)
    except UnboundLocalError:
        print('next')
    ax.set_xlabel(r'Time$\rightarrow$', color='black',  fontsize=18)
    ax.xaxis.set_label_position('top') 
    plt.vlines(isochrone[4::4][0:16],0,1,'grey','solid')
    plt.xticks(patternPos, labels, rotation=40)
    #plt.show()
    plt.savefig(savepath)
