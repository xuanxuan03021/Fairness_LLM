import pandas as pd

#load accuarcy and the fairness result

acc=pd.read_csv('Heart_disease/accuracy_1.0_results.csv')
fairness=pd.read_csv('Heart_disease/heart_disease_fairness_1.0_results.csv')
fair_difference = fairness.iloc[0,:] - fairness.iloc[1,:]
print("fairness evaluation",fair_difference)

print("accuarcy evaluation",acc)
print(acc.iloc[0,:].astype('float'))
print(acc.iloc[1,:].astype('float'))
acc_difference = acc.iloc[0,:].astype('float') - acc.iloc[1,:].astype('float')
print("accuarcy evaluation",acc_difference)
