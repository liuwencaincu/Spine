import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#应用标题
st.title('Application of Machine Learning Methods to Analyse Spinal Infection in Adults: Pyogenic Versus Tuberculous')



# conf

C = st.sidebar.selectbox('C',('code0','code1'),index=1)
D = st.sidebar.selectbox("D",('code0','code1'),index=1)
#E = st.sidebar.selectbox("Race",('Black','Other','White'),index=0)
F = st.sidebar.selectbox("F",('1','2','3'),index=1)
G = st.sidebar.slider("G", 0, 4, 0)
H = st.sidebar.slider("H", 0, 2, 0)
I = st.sidebar.slider("I", 0, 1, 1)
J = st.sidebar.selectbox("J",('0','1'))
K = st.sidebar.selectbox("K",('0','1','2'))
L = st.sidebar.selectbox("L",('0','1'))
#M =
N = st.sidebar.selectbox("N",('1','2'))
O = st.sidebar.selectbox("O",('1','2'))
P = st.sidebar.selectbox("P",('0','1'))
Q = st.sidebar.selectbox("Q",('0','1'))
R = st.sidebar.selectbox("R",('0','1','2'))
S = st.sidebar.selectbox("S",('0','1'))
T = st.sidebar.selectbox("T",('1','2','3'))




# str_to_int

map = {'<50':1,'>=50':2,'male':1,'female':2,'Black':1,'Other':2,'White':3,'ATC':1,'FTC':2,'MTC':3,'PTC':4}
map1 ={'code0':0,'code1':1,'0':0,'1':1,'2':2,'3':3,'4':4}
C =map1[C]
D =map1[D]
F =map1[F]
#G =map1[G]
#H =map1[H]
#I =map1[I]
J =map1[J]
K =map1[K]
L =map1[L]
N =map1[N]
O =map1[O]
P =map1[P]
Q =map1[Q]
R =map1[R]
S =map1[S]
T =map1[T]


# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['Tuberculous'] = thyroid_train['Tuberculous'].apply(lambda x : +1 if x==1 else 0)
thyroid_test = pd.read_csv('test.csv', low_memory=False)
thyroid_test['Tuberculous'] = thyroid_test['Tuberculous'].apply(lambda x : +1 if x==1 else 0)
features = ['C','D','F','G','H','I','J','K','L','N','O','P','Q','R','S','T']
target = 'Tuberculous'

#train and predict
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
#RF.fit(thyroid_train[features],thyroid_train[target])
rbf = SVC(kernel='rbf',C=1.0,random_state=32,probability=True)
rbf.fit(thyroid_train[features].astype('int64'),thyroid_train[target])
#读之前存储的模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.8
#figure
is_t =  (1-rbf.predict_proba([[C,D,F,G,H,I,J,K,L,N,O,P,Q,R,S,T]])[0][0]) > sp
prob = (1-rbf.predict_proba([[C,D,F,G,H,I,J,K,L,N,O,P,Q,R,S,T]])[0][0])#*1000//1/1000
#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))
if is_t == True:
    result = 'Tuberculous'
else:
    result = 'Pyogenic'
st.markdown('## Predict:  '+str(result))
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行
st.text("")
st.text("")
st.title("")
st.title("")
st.title("")

st.info('Information of the model: accuracy: 0.9762 ;precision: 1.0000 ;recall: 0.9545; f1:0.9767;specificity :1.0000 ;kappa:0.9524 ')
st.info('Thank you for using! This is a purely informational message. If you click the button below, there will be celebration!')
if st.button('Click for celebration'):
    st.balloons()
    st.balloons()



