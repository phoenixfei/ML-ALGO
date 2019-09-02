import numpy as np

def loadData(file='data/datingTestSet2.txt'):
    with open(file, 'r') as f:
        lines = f.readlines()
    trainset = np.zeros((len(lines),3))
    label = []
    index = 0
    for line in lines:
        listLine = line.strip().split('\t')
        trainset[index,:] = listLine[0:3]
        label.append(int(listLine[-1]))
        index += 1
    return trainset, label

def normalData(data):
    minval = data.min(0)
    maxval = data.max(0)
    ranges = maxval - minval

if __name__ == '__main__':
    trainset, label = loadData()
    print(trainset, label)