import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import re
import nltk
import nltk.stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from numpy import *
import csv
import time
start =time.clock()
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^a-z']", " ", text)
        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"\'m", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " will ", text)
        text = re.sub(r"ain\'t", " are not ", text)
        text = re.sub(r"aren't", " are not ", text)
        text = re.sub(r"couldn\'t", " can not ", text)
        text = re.sub(r"didn't", " do not ", text)
        text = re.sub(r"doesn't", " do not ", text)
        text = re.sub(r"don't", " do not ", text)
        text = re.sub(r"hadn't", " have not ", text)
        text = re.sub(r"hasn't", " have not ", text)
        text = re.sub(r"\'ll", " will ", text)
        # 换掉一些停词，貌似更差了
        # text = re.sub(r"am", "", text)
        #进行词干提取
        new_text = ""
        s = nltk.stem.snowball.EnglishStemmer()  # 英文词干提取器
        for word in word_tokenize(text):
            new_text = new_text + " " + s.stem(word)
        # 放回去
        comment_list.append(new_text)
    return comment_list

def read_data(file):
    train_data = csv.reader(open(file, encoding="utf-8"))
    lines = 0
    for r in train_data:
        lines += 1
    train_data_label = np.zeros([lines - 1, ])
    train_data_content = []
    train_data = csv.reader(open(file, encoding="utf-8"))
    i = 0
    for data in train_data:
        if data[0] == "Label" or data[0] == "SmsId":
            continue
        if data[0] == "ham":
            train_data_label[i] = 0
        if data[0] == "spam":
            train_data_label[i] = 1
        train_data_content.append(data[1])
        i += 1
    print(train_data_label.shape, len(train_data_content))
    return train_data_label,train_data_content


# 载入数据
train_y,train_data_content = read_data("train.csv")
_,test_data_content = read_data("test.csv")
train_data_content = clean_text(train_data_content)
test_data_content = clean_text(test_data_content)

# 数据的TF-IDF信息计算
all_comment_list = list(train_data_content) + list(test_data_content)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_x = text_vector.transform(train_data_content)
test_x = text_vector.transform(test_data_content)
train_x = train_x.toarray()
test_x = test_x.toarray()
print(train_x.shape,test_x.shape,type(train_x))#将训练和测试样本转化为向量(5572, 5000) (1115, 5000)

#训练朴素贝叶斯模型（得到所需后验概率）
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive
#朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
p0V,p1V,pSpam = trainNB0(train_x,train_y)
answer = pd.read_csv(open("sampleSubmission.csv"))
for i in range(test_x.shape[0]):
    if classifyNB(test_x[i],p0V,p1V,pSpam )==1:
        answer.loc[i,"Label"] = "spam"
    else:
        answer.loc[i,"Label"] = "ham"
answer.to_csv("submission.csv",index=False)  # 不要保存引索列
end = time.clock()
print('Running time: %s Seconds'%(end-start))



