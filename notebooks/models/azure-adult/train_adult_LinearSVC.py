import os
import argparse
import joblib
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from azureml.core import Workspace, Dataset, Experiment, Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
dataset_name = 'Dataset-adults'
print("Loading data from " + dataset_name)
#Adult Census Income Binary classfication dataset
df = Dataset.get_by_name(workspace=run.experiment.workspace, name=dataset_name).to_pandas_dataframe()


le = LabelEncoder()
new_df = df.loc[:, ~df.columns.isin(['workclass','occupation','native-country'])]
new_df.loc[:,'education'] = le.fit_transform(new_df.loc[:,'education'].values)
new_df.loc[:,'marital-status'] = le.fit_transform(new_df.loc[:,'marital-status'].values)
new_df.loc[:,'relationship'] = le.fit_transform(new_df.loc[:,'relationship'].values)
new_df.loc[:,'race'] = le.fit_transform(new_df.loc[:,'race'].values)
new_df.loc[:,'sex'] = le.fit_transform(new_df.loc[:,'sex'].values)
new_df.loc[:,'income'] = le.fit_transform(new_df.loc[:,'income'].values)

X = new_df.drop('income', axis=1).values
y = new_df['income'].values

#use stratify for un balanced number of examples for each class label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42,shuffle=True)

print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_test.shape: {y_test.shape}')
    
# Model Training: Support Vector Machines
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODEL = 'LinearSVC'
model = make_pipeline(StandardScaler(), 
                    LinearSVC(penalty='l1', C=0.001, dual=False, max_iter=1000, random_state=42))
model.fit(X_train, y_train)
print(f"model: {model}")

# Evaluate the model
from sklearn import metrics

def convert_prob_to_binary(y_pred):
    # convert probabilities to binary prediction using threshold=0.5
    for i in range(0, y_pred.shape[0]):
        if y_pred[i] >= .5: # setting threshold to .5
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    
    return y_pred

# Prediction
y_pred = model.predict(X_test)
print(f"prediction: {y_pred}")
print(f'Misclassified examples:{(y_test != y_pred).sum()}')

y_pred = convert_prob_to_binary(y_pred)
        
# Confusion Matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred)
print(f"confusion matrix:\n {conf_mat}")

# AUC
auc = metrics.roc_auc_score(y_test, y_pred)
print(f"auc: {auc}")

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") 
model.score(X_test, y_test)
print(f'Accuracy:{model.score(X_test, y_test)}')

# Precision
precision = metrics.precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision}")

# Recall
recall = metrics.recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall}")

# F1
f1 = metrics.f1_score(y_test, y_pred, average='macro')
print(f"F1: {f1}")

# classification_report
print('classification_report:')
print(metrics.classification_report(y_test, y_pred))

# Save model
print('Exported model: adult_LinearSVC.pkl')
f = open('adult_LinearSVC.pkl', 'wb')
pickle.dump(model, f)
f.close()

print("----- Results from saved Model -----")
# Load Model
print('Load model: adult_LinearSVC.pkl')
f2 = open('adult_LinearSVC.pkl', 'rb')
model = pickle.load(f2)

# Prediction
y_pred = model.predict(X_test)
print(f"prediction: {y_pred}")
print(f'Misclassified examples:{(y_test != y_pred).sum()}')

y_pred = convert_prob_to_binary(y_pred)
        
# Confusion Matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred)
print(f"confusion matrix:\n {conf_mat}")

# AUC
auc = metrics.roc_auc_score(y_test, y_pred)
print(f"auc: {auc}")

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") 
model.score(X_test, y_test)
print(f'Accuracy:{model.score(X_test, y_test)}')

# Precision
precision = metrics.precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision}")

# Recall
recall = metrics.recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall}")

# F1
f1 = metrics.f1_score(y_test, y_pred, average='macro')
print(f"F1: {f1}")

# classification_report
print('classification_report:')
print(metrics.classification_report(y_test, y_pred))


os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/adult_LinearSVC.pkl')
