import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from textToTensor import textToTensor

#EXCEL FILE FORMATIING
df = pd.read_excel('Customer_requierments.xlsx')
df.drop(columns=['Artifact Type', 'isHeading', 'parentBinding', 'module', 'id'], inplace=True)

df.replace(to_replace='Non-functional with no effect on Product', value = '3', inplace=True)
df.replace(to_replace='Functional', value = '1', inplace=True)
df.replace(to_replace='Non-functional with effect on Product', value = '2', inplace=True)

df=df.dropna(axis=0).reset_index(drop=True)
df["Primary Text"] = df["Primary Text"].str.strip()

#df.to_excel('Test_Req.xlsx')

#TURN DATA INTO TENSORS

X = textToTensor(df['Primary Text'])
y = textToTensor(df['H_FunctionalClassification'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test)






