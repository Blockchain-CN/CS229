# encoding=utf8

"""
https://blog.csdn.net/tanhongguang1/article/details/45016421
"""

def loadDataSet():  # 读入数据
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1代表侮辱性 0代表正常
    return postingList, classVec


def createVocabList(dataSet):  # 创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWord2VecMN(vocabList, inputSet): #
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def divVect(numVec, denom):
    resultVec = []
    for num in numVec:
        resultVec.append(num/denom)
    return resultVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = [1]*numWords; p1Num = [1]*numWords   #计算频数初始化为1
    '''
    原本结果对每个次的频数统计应该是[2,4,2,5,6,2,0,0,1:0] [4,2,5,0,0,1,5,6,2:1]
    但是发现有两项没出现过，没出现不意味着出现就一定不是，防止出现概率奇异
    所以给每个分类中加一个全一向量  [1,1,1,1,1,1,1,1,1:0] [1,1,1,1,1,1,1,1,1:1]
    最终结果为                   [3,5,3,6,7,3,1,1,2:0] [5,3,6,1,1,2,6,7,3:1]
    '''
    p0Denom = 2.0;p1Denom = 2.0  # 即拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            for l in range(numWords):
                p1Num[l] += trainMatrix[i][l]
            p1Denom += 1
        else:
            for l in range(numWords):
                p0Num[l] += trainMatrix[i][l]
            p0Denom += 1
    p1Vect = divVect(p1Num, p1Denom)
    p0Vect = divVect(p0Num, p0Denom)
    return p0Vect, p1Vect, pAbusive         #P(Ni|Y0), P(Ni|Y1), P(Y1)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = 1 - pClass1
    p1 = pClass1
    for i in range(len(vec2Classify)):
        if vec2Classify[i] == 1:
            p0 *= p0Vec[i]
            p1 *= p1Vec[i]
    if p1 > p0:
        return "侮辱言论, p1=%f > p0=%f" %(p1,p0)
    else:
        return "正常言论, p0=%f > p1=%f" %(p0,p1)

listOPosts,listClasses = loadDataSet()#加载数据
myVocabList = createVocabList(listOPosts)#建立词汇表
print myVocabList
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bagOfWord2VecMN(myVocabList, postinDoc))
p0V,p1V,pAb = trainNB0(trainMat,listClasses)#训练
testEntry = ['stop']
thisDoc = bagOfWord2VecMN(myVocabList,testEntry)
print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)