from apyori import apriori
import pandas as pd
import pickle

data = pd.read_csv("data.csv",header=None)
data = data.iloc[1:,]
print(data)
n,m=data.shape

data1=[]
for i in range(n):
    data1.append([str(data.values[i,j]) for j in range(m)]) 
print("\n",data1,"\n")
assosciation_rule= apriori(data1, min_support=0.0002,min_confidence=0.08,min_lift=1.2, max_length=2)
assosciation_resault=list(assosciation_rule)
#filter for minimum length = 2
assosciation_resault = filter(lambda x: len(x.items) > 1 , assosciation_resault)
assosciation_resault=list(assosciation_resault)

####################################################   Sort   ###############################################
df = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence','Lift'))
Support =[]
Confidence = []
Lift = []
Items = []
Antecedent = []
Consequent=[]

for RelationRecord in assosciation_resault:
    for ordered_stat in RelationRecord.ordered_statistics:
        Support.append(RelationRecord.support)
        Items.append(RelationRecord.items)
        Antecedent.append(ordered_stat.items_base)
        Consequent.append(ordered_stat.items_add)
        Confidence.append(ordered_stat.confidence)
        Lift.append(ordered_stat.lift)

df['Items'] = list(map(set, Items))                                   
df['Antecedent'] = list(map(set, Antecedent))
df['Consequent'] = list(map(set, Consequent))
df['Support'] = Support
df['Confidence'] = Confidence
df['Lift']= Lift
print(df.head(),"\n",df.shape)


result=[]
n=0
for _, row in df.iterrows():
        ID='01'
        if n>=3:
            break
        elif ID in row['Antecedent'] :
            result.append(list(row['Consequent'])[0])
            n+=1  
print(result)  
####################################################   SAVE MODEL   ###############################################
# filename="aprioi.sav"
# pickle.dump(df, open(filename, 'wb'))
