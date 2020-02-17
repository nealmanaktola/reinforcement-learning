# My approach:
# Get 10 random sequences.
# For each sequence calculate delta w
# Do this for 10 sequences and then add them.
# For weight update:
# 1) W_iter+1<- W_iter+delta W
# 2)calculate Predicted value using W_iter+1
# 3)Calculate new delta W using these predicted values (new delta W
# 4)W_iter+2<- W_iter+1 + new delta W
# Do 1 to 4 until tolerance ( difference in two consecutive weights) is less than 0.01 (can vary tolerance)
# Do steps 1-4 for 100 training sets (set of 10 sequence each)
# calculate RMSE as follows:
# RMSE for one training set: sqrt((predicted-actual prbab)^2/5 )(do element by element as it is an array)
# Do this for all 100 training sets.
# Take average and report answer.

# Am I missing anything? Please advise.


# each seq calculate deltaW
import numpy as np
import matplotlib.pyplot as plt
import random


index = {
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5
}

def createTestSequence():
    return [
        ['D', 'C', 'D', 'E', 'F', 'G'],
        ['D', 'E', 'F', 'G'],
        ['D', 'C', 'B', 'A'],
        ['D', 'C', 'D', 'E', 'F', 'G'],
        ['D', 'E', 'F', 'G'],
        ['D', 'C', 'B', 'A'],
        ['D', 'C', 'D', 'E', 'F', 'G'],
        ['D', 'E', 'F', 'G'],
        ['D', 'C', 'B', 'A'],
        ['D', 'C', 'D', 'E', 'F', 'G']              
    ]

def createSequences():
    sequences = []

    for _ in range(10):
        sequence = ['D']
        state = 'D'
        while (state != 'A' and state != 'G'):
            ch = random.choice([-1, 1])
            state = chr(ord(state) + ch)
            sequence.append(state)
        sequences.append(sequence)
    return sequences

def predict(w, c):
    if c == 'A':
        return 0.0
    if c == 'G':
        return 1.0
    idx = index[c] - 1

    return w[idx] 

def idx(s, t):
    return index[s[t]] - 1

def unitVec(i):
    vec = 5*[0.0]
    vec[i-1] = 1.0
    return vec

def update2(sequences, lbda, w, alpha):
    #print(sequences)
    for i,s in enumerate(sequences):
        for t in range(0, len(s) - 1):
            cur = predict(w, s[t])
            nxt = predict(w, s[t + 1])
            deltaW = np.zeros(5)
            deltaW[idx(s,t)] = alpha * (nxt - cur) * sum([lbda ** (t-k) for k in range(1,t+1)])
            w += deltaW

    return w

def update(sequences, lbda, w, alpha):
    #print(sequences)
    totalDeltaW = np.zeros((10, 5))
    for i,s in enumerate(sequences):
        for t in range(0, len(s) - 1):
            cur = predict(w, s[t])
            nxt = predict(w, s[t + 1])
            totalDeltaW[i][idx(s,t)] += alpha * (nxt - cur) * sum([lbda ** (t-k) for k in range(1,t+1)])

    #print("totalDeltaW", np.sum(totalDeltaW, axis =0))
    return np.sum(totalDeltaW, axis=0)

def experiment1():
    #keep calling update until we reach consensus
    eps = 0.0001
    lbdas = [0,0.1,0.3, 0.5,0.7,0.9,1]
    alphas = [0.0230, 0.0210, 0.0165, 0.0115, 0.0080, 0.0040, 0.0020]
    # lbdas = [0.0, 0.0, 0.0]
    # alphas = [0.0230, 0.0115, 0.0020]
    results = []
    for l,a in zip(lbdas,alphas):
        w = np.ones((100, 5)) * 0.5
        rmslist = np.zeros(100)
        for i, seqs in enumerate(trainingData):
            deltaW = update(seqs, l, w[i], a)
            itr = 1
            while np.linalg.norm(deltaW) > eps:
                #if np.linalg.norm(deltaW) > 1:
                    #print("greater than 1: ", w[i])
                #print(deltaW)
                deltaW = update(seqs, l, w[i], a)
                w[i] += deltaW

                #print("deltaW: ", deltaW)
                itr +=1
            #print(i, "\nnumber of iterations:", itr, "w value: ",w[i])

            #print("alpha", a, "training set ", i, "wset @ ", w[i])
            rmslist[i] = rmse(w[i], actual())
            #print("rmse", rmslist[i])
        print("lambda", "alpha", l, a, np.mean(rmslist))
        results.append(np.mean(rmslist))
        #print("wsets", w)
    return results


def createTrainingData():
    trainingData = []
    for i in range(100):
        seqs = createSequences()
        trainingData.append(seqs)
    return trainingData

def plotGraph(dataY, dataX=None, showMarker=False, plotTitle=None, xTitle=None, yTitle=None):
    if plotTitle is not None: fig = plt.figure()
    if dataX is not None:
        plt.plot(dataX, dataY, 'o-') if showMarker else plt.plot(dataY)
    else:
        plt.plot(dataY, 'o-') if showMarker else plt.plot(dataY)
        
    if plotTitle is not None:
        fig.suptitle(plotTitle, fontsize=20)
    if xTitle is not None: 
        plt.xlabel(xTitle, fontsize=18)
    if yTitle is not None: 
        plt.ylabel(yTitle, fontsize=16)
    
    plt.show()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def actual():
    return np.array([1.0/6.0, 1.0/3.0, 1.0/2.0, 2.0/3.0, 5.0/6.0])

def experiment2(alphas, lbdas):

    out = {}
    for l in lbdas:
        results = []
        for a in alphas:
            w = np.ones((100, 5)) * 0.5
            rmslist = np.zeros(100)
            for i, seqs in enumerate(trainingData):
                w[i] = update(seqs, l, w[i], a)
                rmslist[i] = rmse(w[i], actual())
            #print("rmse", rmslist[i])
            results.append(np.mean(rmslist))
        out[l] = results
    return out

random.seed(5)
trainingData = createTrainingData()
# results = experiment1()
# print(results)
# plotGraph(np.array(results), np.array([0,0.1,0.3, 0.5,0.7,0.9,1]))
lbdas = [0, 0.3, 0.8, 1.0]
alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]  
results = experiment2(alphas, lbdas)
for _,result in results.items():
    plt.plot(alphas, result)
axes = plt.gca()
axes.set_ylim([0, 0.8])
plt.legend(lbdas, loc='upper left')
plt.show()
print(results)        