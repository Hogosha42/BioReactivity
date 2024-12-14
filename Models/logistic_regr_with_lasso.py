from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the CSV file into a DataFrame
df = pd.read_csv('train.csv')
mols = [Chem.MolFromSmiles(mol) for mol in df['SMILES_canonical']]
X = [[d[1](m) for d in Descriptors._descList] for m in mols]
Y = df['target_feature']

#splitting data in training and testing data
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

'''Scaling and PCA are done after splitting to prevent data leakage'''

#scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled= scaler.transform(X_test)

#Perfrom pca and transform data
pca = PCA(0.80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(X_train_pca.shape, X_val_pca.shape, X_test_pca.shape)

#Logistic regression algorithm
reg = LogisticRegression()
reg.fit(X_train_pca, y_train)

#Predict y values from X_val_pca and produce balanced accuracy, precision and recall  
y_pred_reg = reg.predict(X_val_pca)
print('balanced accuracy:',balanced_accuracy_score(y_val, y_pred_reg),'\n', 'precision:',precision_score(y_val, y_pred_reg), '\n', 'recall:', recall_score(y_val, y_pred_reg))

#Prediction on test data
y_pred_logistic_test = reg.predict(X_test_pca)

print(y_pred_logistic_test)