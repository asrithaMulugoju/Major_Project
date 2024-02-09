import datapreprocess as dp
import VLSI as vls
import pmeasures as ps
import pandas as pd
import VNMF as vn
import VLDA as vl
import VPLSI as vpl

# Cluster Size
n11=40

# Total Number of Tweets
Tn=80

#Number of Topics
ntopics=2

#Initialize Arrays

ACC = []
NMI = []
F1 = []
REC = []
PREC = []

# Reading Tweets Data
datafile= 'D:\data\TwoTopics2.csv'
tweets=pd.read_csv(datafile, encoding='latin1')

# Data Preprocesseing
bow, t1 = dp.preprocess(tweets)

#VNMF Model
res1, gnd = vn.VNMF(bow,n11,Tn,ntopics)

#Performance Calculation of VNMF
ACC, NMI, F1, REC, PREC, CMNMF = ps.PM(ACC, NMI, F1, REC, PREC, gnd, res1)

#VLDA Model
res2, gndLDA, corpus, dictionary = vl.VLDA(n11,tweets,Tn,ntopics)

#Performance Calculation of VLDA
ACC, NMI, F1, REC, PREC, CMLDA = ps.PM(ACC, NMI, F1, REC, PREC, gnd, res2)

#VLSI Model
res3, gndLSI = vls.VLSI(n11, corpus, dictionary, Tn, ntopics)

#Performance Calculation of VLSI
ACC, NMI, F1, REC, PREC, CMLSI = ps.PM(ACC, NMI, F1, REC, PREC, gnd, res3)

#VPLSI Model
res4, gndPLSI = vpl.VPLSI(n11, datafile,Tn, ntopics)

#Performance Calculation of VPLSI
ACC, NMI, F1, REC, PREC, CMPLSI = ps.PM(ACC, NMI, F1, REC, PREC, gnd, res4)

#Confusion Matrix of VNMF
print('Confusion Matrix of VNMF')
print(CMNMF)

#Confusion Matrix of VLDA
print('Confusion Matrix of VLDA')
print(CMLDA)

#Confusion Matrix of VLSI
print('Confusion Matrix of VLSI')
print(CMLSI)

#Confusion Matrix of VPLSI
print('Confusion Matrix of VPLSI')
print(CMPLSI)

# Saving Performance Values into Excel Sheet
df = pd.DataFrame({'NMF': [ACC[0],NMI[0],F1[0],REC[0],PREC[0]], 'LDA': [ACC[1],NMI[1],F1[1],REC[1],PREC[1]],'LSA':[ACC[2],NMI[2],F1[2],REC[2],PREC[2]],'PLSI': [ACC[3],NMI[3],F1[3],REC[3],PREC[3]]})
writer =pd.ExcelWriter('D:\TWO-TOPIC.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
#print(metrics.calinski_harabasz_score(WPLSI, res4))
#print(metrics.davies_bouldin_score(WPLSI, res4))
#print(metrics.silhouette_score(WPLSI, res4, metric='euclidean'))