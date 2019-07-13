import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import cv2
import mahotas as mt 

path='dataset'
path2='dataset/train_aug'
fixed_size = tuple((500, 500))

result_color=[]
labels_color=[]

result_texture=[]
labels_texture=[]

result_shape=[]
labels_shape=[]
global_features=[]
labels=[]

def extract_color(im2):
    histr=cv2.calcHist([im2],[0,1,2],None,[8,8,8],[0,255,0,255,0,255])
    return histr.flatten()

def extract_texture(im2):
    texture=mt.features.haralick(im2)
    ht_mean=texture.mean(axis=0)
    return ht_mean.flatten()

def extract_shape(im2):
    shape=cv2.HuMoments(cv2.moments(im2))
    return shape.flatten()
i=0
for fol in os.listdir(path): #feature extraction
    new_path=path+'/'+fol
    i=i+1
    cur_dir=fol
    for image in os.listdir(new_path):
        des=new_path+'/'+image
        im = cv2.imread(des)
        img = cv2.imread(des)
        if im is not None:
            im = cv2.resize(im, fixed_size)
            feature1=extract_color(im) #color
            #result_color.append(feature) 
            #labels_color.append(fol)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            feature2=extract_texture(gray) #texture
            #result_texture.append(feature2)
            #labels_texture.append(fol)
            feature3=extract_shape(gray) #shape
            #result_shape.append(feature3)
            global_feature = np.hstack([feature1, feature2, feature3])
            labels.append(cur_dir)
            global_features.append(global_feature)
            #global_features=np.reshape(global_features,(-1,532))
            #labels_shape.append(fol)
        else:
            print('Failed to open the file')
    print ("[STATUS] processed folder: {}.{}".format(i,cur_dir))

global_feature=np.reshape(global_feature,(-1,532))
Tr_global_feature=np.transpose(global_features)
Tr2_global_feature=np.transpose(Tr_global_feature)
#global_feature=np.transpose(global_features)
#new_result_color=np.array(result_color)
#new_labels_color=np.array(labels_color)
#global_feature = np.hstack([result_color, result_texture, result_shape])
test_size = 0.10
seed = 9

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(Tr2_global_feature)
#new_rescaled_features=np.array(rescaled_features)
#new_labels=np.array(labels)

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(rescaled_features),np.array(target),test_size=test_size,random_state=seed)
num_trees=10
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(random_state=9)))

results = []
names = []
scoring = "accuracy"

import warnings
warnings.filterwarnings('ignore')

print("ALGO    MEAN    STD")
# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
   
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison ')
ax = fig.add_subplot(111)
#pyplot.boxplot(results)
plt.boxplot(results)
ax.set_xticklabels(names)
#pyplot.show()
plt.show()