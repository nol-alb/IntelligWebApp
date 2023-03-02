import numpy as np
import matplotlib.pyplot as plt

def errorVisualization(pattern, averageError):
    isochrone = np.arange(0,(16*4)+8)
    patternPos =[]
    patternPos = []
    for i,k in enumerate(pattern):
        if(k == 1):
            if((i*4)+4 == isochrone.size):
                patternPos.append((i*4 - 1)+4)
            else:
                patternPos.append((i*4)+4)
    patternPlayPos = []
    j = 0
    if(np.isscalar(averageError)):
        return patternPlayPos
    else:
        for i,k in enumerate(pattern):
            if(k == 1):
                if (abs(averageError[j])<=10):
                    print('here10')
                    if((i*4)+4 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4)
                    else:
                        patternPlayPos.append((i*4)+4)
                elif(averageError[j]>=10 and averageError[j] < 40):
                    if((i*4)+4 + 1 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 + 1)
                    else:
                        patternPlayPos.append((i*4)+4 + 1)
                elif(averageError[j]>= 40 and averageError[j] <= 70):
                    if((i*4)+4 + 2 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 + 2)
                    else:
                        patternPlayPos.append((i*4)+4 + 2)
                elif(averageError[j] >= 70 and averageError[j] <= 100):
                    if((i*4)+4 + 3 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 + 3)
                    else:
                        patternPlayPos.append((i*4)+4 + 4)
                elif(averageError[j] <= -10 and averageError[j] >= -40):
                    if((i*4)+4 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 - 1)
                    else:
                        patternPlayPos.append((i*4)+4 -1)
                elif(averageError[j] <= -40 and averageError[j] >= -70):
                    if((i*4)+4 == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 -2)
                    else:
                        patternPlayPos.append((i*4)+4 - 2)
                elif(averageError[j] <= -70 and averageError[j] >= -100):
                    if((i*4)+4  == isochrone.size):
                        patternPlayPos.append((i*4 - 1)+4 - 3)
                    else:
                        patternPlayPos.append((i*4)+4 - 3)
                j = j+1

    return patternPlayPos
def errorPlot(isochrone, patternPos, patternPlayPos,pattern):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    k = 0
    for i,k in enumerate(patternPos):    
        plt.vlines(isochrone[patternPlayPos[i]],0,1,colors[k%len(colors)])
        plt.vlines(isochrone[k],0,1,colors[k%len(colors)],'dotted')
        k = k+1
    plt.xlabel("Onsets")
    locs, labels = plt.xticks()
    j=1
    labels=[]
    for i in pattern:
        if(i==1):
            labels.append("Hit"+str(j))
            j = j+1
    ax = plt.gca()
    plt.xticks(patternPos, labels, rotation=40)
    ax.get_yaxis().set_visible(False)
    return 0

def PatternVisualizer(pattern,path):
    plt.figure(figsize=(10.5,3.5))
    isochrone = np.arange(0,(16*4)+8)
    patternPos = []
    for i,k in enumerate(pattern):
        if(k == 1):
            if((i*4)+4 == isochrone.size):
                patternPos.append((i*4 - 1)+4)
            else:
                patternPos.append((i*4)+4)
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
        plt.scatter(isochrone,scats, s=200, marker='x',color= colors[i%len(colors)],alpha = 1)
        print(cmap)
        scats[patternPos[i]] = np.nan
    # for i in range(len(patternPos2)):
    #     scats[patternPos2[i]] = 0.5
    #     cmap =colors[i%len(colors)]
    #     print(cmap)
    #     plt.scatter(isochrone,scats, s=50, marker='o',color= colors[i%len(colors)], alpha = 1)
    #     scats[patternPos2[i]] = np.nan
    ax.xaxis.set_label_position('top') 
    plt.vlines(isochrone[4::4],0,1,'grey','solid')
    ax.set_xlabel(r'Time$\rightarrow$', color='black',  fontsize=18)
    plt.xticks(patternPos, labels, rotation=40)
    plt.show()
    plt.legend()
    plt.savefig(path)

def PatternErrorVisualizer(pattern,patternPos2,savepath):
    plt.figure()
    isochrone = np.arange(0,(16*4)+8)
    patternPos = []
    for i,k in enumerate(pattern):
        if(k == 1):
            if((i*4)+4 == isochrone.size):
                patternPos.append((i*4 - 1)+4)
            else:
                patternPos.append((i*4)+4)
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
    plt.savefig(savepath)