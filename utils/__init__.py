# from functools import reduce
# from  numpy  import *
#
# def process(item,temp_max,temp_min):
#     # print(type(temp_min))
#     temp = int((temp_max-temp_min)/3)
#     for i in range(len(item)):
#         if temp_min<= item[i] and item[i]>temp_min+temp:
#             item[i]=0
#         elif item[i] >= temp_min + temp and item[i] <temp_min + 2*temp:
#             item[i] = 1
#         else:
#             item[i]=2
#     return item
#
# train=[]
# number_sample=int(input())
# label=list(map(int,input().split()))
# for i in range(number_sample):
#     train.append(list(map(int,input().split())))
# test=[list(map(int,input().split()))]
# temp=[x[2] for x in train]
# changdi=[x[3] for x in train]
# persons=[x[4] for x in train]
# test_temp=[x[2] for x in test]
# test_changdi=[x[3] for x in test]
# test_persons=[x[4] for x in test]
# temp_max=max(temp)
# temp_min=min(temp)
# changdi_max=max(changdi)
# changdi_min=min(changdi)
# persons_max=max(persons)
# persons_min=min(persons)
# temp=process(temp,temp_max,temp_min)
# changdi=process(changdi,changdi_max,changdi_min)
# persons=process(persons,persons_max,persons_min)
# test_temp=process(test_temp,temp_max,temp_min)
# test_changdi=process(test_changdi,changdi_max,changdi_min)
# test_persons=process(test_persons,persons_max,persons_min)
#
# for i in range(number_sample):
#     train[i][2]=temp[i]
#     train[i][3] = changdi[i]
#     train[i][4] = persons[i]
# test[0][2]=test_temp[0]
# test[0][3]=test_changdi[0]
# test[0][4]=test_persons[0]
# print(test)
# print(train)
# class NaiveBayesClassifier(object):
#
#     def __init__(self,label,feature,number_sample):
#         self.dataMat = feature
#         self.labelMat = label
#         self.pLabel1 = 0
#         self.p0Vec = list()
#         self.p1Vec = list()
#         self.number_sample=number_sample
#     def train(self):
#         dataNum = self.number_sample
#         featureNum = len(self.dataMat[0])
#         self.pLabel1 = sum(self.labelMat) / float(dataNum)
#         p0Num = zeros(featureNum)
#         print(p0Num )
#         p1Num = zeros(featureNum)
#         p0Denom = 1.0
#         p1Denom = 1.0
#         for i in range(len(self.labelMat)):
#             if self.labelMat[i] == 1:
#                 p1Num += self.dataMat[i]
#                 p1Denom += sum(self.dataMat[i])
#             else:
#                 p0Num += self.dataMat[i]
#                 p0Denom += sum(self.dataMat[i])
#         self.p0Vec = p0Num / p0Denom
#         self.p1Vec = p1Num / p1Denom
#
#     def classify(self, data):
#         p1 = reduce(lambda x, y: x * y, data * self.p1Vec) * self.pLabel1
#         p0 = reduce(lambda x, y: x * y, data * self.p0Vec) * (1.0 - self.pLabel1)
#         return p1/p0
#
#     def test(self,data):
#         # self.loadDataSet('testNB.txt')
#         self.train()
#         print(self.classify(data))
#
#
# if __name__ == '__main__':
#     NB = NaiveBayesClassifier(label,train,number_sample)
#     NB.test(test)
# 9
# 0 0 0 0 1 1 1 1 1
# 0 0 30 450 7
# 1 1 5 500 3
# 1 0 10 150 1
# 0 1 40 300 6
# 1 0 20 100 10
# 0 1 25 180 12
# 0 0 32 50 11
# 1 0 23 120 9
# 0 0 27 200 8

# 9
# 0 0 0 0 1 1 1 1 1
# 0 0 30 450 7
# 1 1 5 500 3
# 1 0 10 150 1
# 0 1 40 300 6
# 1 0 20 100 10
# 0 1 25 180 12
# 0 0 32 50 11
# 1 0 23 120 9
# 0 0 27 200 8
# 0 0 40 180 8
from .utils import My_Dataset
from .utils import search_file
from .utils import COLOR_DICT
from .crf  import dense_crf
from .dice_loss import MulticlassDiceLoss
from .dice_loss import loss_weight
