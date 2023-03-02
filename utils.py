import random
import re
import numpy as np
import os

Stimuli_set = {"A": [[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]], 
"B": [[1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]], 
"C": [[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]], 
"D": [[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]], 
"E":[[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]], 
"F":[[1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0]],
"Gt":[[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0,0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]] }
 


 
# helper function to perform sort
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]
def pathOrganizer():
    Glist = ['G1','G2','G3','G4','G5','G6']
    folders = np.array(['A','B','C','D','E','F'])
    np.random.shuffle(folders)
    pathDictionary = {}
    filesDictionary = {}
    patternDictionary = {}
    randomDictionary= {}
    imgpathDictionary={}
    onsepathDictionary ={}
    for i in Glist:
        pathDictionary[i] = ''
        filesDictionary[i] = []
        patternDictionary[i] = []
        randomDictionary[i] = []
        imgpathDictionary[i] = []
        onsepathDictionary[i] = []
    for k,i in enumerate(folders):
        trialFiles = []
        trialList = []
        path = 'static/data/Stimuli/GenStimuli/'
        trials = path+i
        pathDictionary[Glist[k]] = ''
        for f in os.listdir(trials):
                if f.endswith(".mp3"):
                    trialFiles.append(os.path.join(trials, f))
        trialFiles.sort()
        filesDictionary[Glist[k]] = trialFiles

    for k in range(3,6):
        trialFiles = []
        trialList = []
        for i in range(5):
            l = i%3
            rnd = random.choice(filesDictionary[Glist[(k+l)%3+3]])
            filesDictionary[Glist[(k+l)%3+3]].remove(rnd)
            trialFiles.append(rnd)
        trialFiles.sort(key=num_sort)
        randomDictionary[Glist[k]] = trialFiles


    for k in range(3,6):
        filesDictionary[Glist[k]]=randomDictionary[Glist[k]] 

    for i in Glist:
        G = filesDictionary[i]
        for k in G:
            position = k.find('.mp3')
            pattern = int(k[position-1])
            file_number = position - 3
            patternDictionary[i].append(Stimuli_set[k[file_number]][pattern-1])


    for keys in filesDictionary.keys():
        for i in range(5):
            path = filesDictionary[keys][i]
            path2 = filesDictionary[keys][i]
            pathImg = list(path)
            pathOns = list(path2)
            position = path.find('.mp3')
            pathImg[position:] = list(".png")
            pathOns[position:] = list(".npy")
            onsepathDictionary[keys].append("".join(pathOns))
            imgpathDictionary[keys].append("".join(pathImg))

    rel_path = 'static/data/Stimuli/Gt'
    pathDictionary['Gt'] = rel_path
    filesDictionary['Gt'] = []
    patternDictionary['Gt'] = []
    imgpathDictionary['Gt'] = []
    onsepathDictionary['Gt']=[]
    patNo = 0
    for f in os.listdir(rel_path):
            if f.endswith(".mp3"):
                    filesDictionary['Gt'].append(os.path.join(rel_path, f))
                    patternDictionary['Gt'].append(Stimuli_set['Gt'][patNo])
                    patNo = patNo+1
            if f.endswith(".png"):
                    imgpathDictionary['Gt'].append(os.path.join(rel_path, f))
            if f.endswith(".npy"):
                    onsepathDictionary['Gt'].append(os.path.join(rel_path, f))
    filesDictionary['Gt'].sort(key=num_sort)
    imgpathDictionary['Gt'].sort(key=num_sort)
    onsepathDictionary['Gt'].sort(key=num_sort)
                    

    melodyBlocks = ['G2','G5']
    for i in melodyBlocks:
        files = filesDictionary[i]
        for j,A in enumerate(files):
            position = A.find('/GenStimuli')
            f = position+len('/GenStimuli')
            splice = A[f:]
            wMelodyPath = '/wmelody'+splice
            listA = list(A)
            listA[f:] = wMelodyPath
            path = ''.join(listA)
            filesDictionary[i][j] = path
    return pathDictionary,filesDictionary,patternDictionary,randomDictionary,imgpathDictionary, onsepathDictionary,folders