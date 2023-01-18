import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import glob
import warnings
warnings.filterwarnings('ignore')

path = 'D:/DataScience/credit_card_fraud/dataset_transacoes_bancarias/'
pkl_train_files = glob.glob(path + 'Treino/*.pkl')
pkl_test_files = glob.glob(path + 'Teste/*.pkl')
df_list = (pd.read_pickle(file) for file in pkl_train_files)

df_train_full = pd.concat(df_list)

# Searching for missing values and imbalanced dataset
print('---'*45, '\nSEARCHING FOR GENERAL ASPECTS OF THE TRAIN DATASET')
print('Missing values:', df_train_full.isnull().sum().max())
print('No Frauds', round(df_train_full['TX_FRAUD'].value_counts()[0]/len(df_train_full) * 100, 2), '% of the dataset')
print('Frauds', round(df_train_full['TX_FRAUD'].value_counts()[1]/len(df_train_full) * 100, 2), '% of the dataset')
print('Numbers of Frauds:', df_train_full['TX_FRAUD'].value_counts()[1])
print('Dataframe shape:', df_train_full.shape)

# Create a list with the columns that will be used
input_features = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW', 'TX_FRAUD']

df = df_train_full[input_features].copy()

# Preprocessing the TX_AMOUNT to be in the same scale as others features (between -1 and 1)
scaler = RobustScaler()
df['AMOUNT_SCALED'] = scaler.fit_transform(df['TX_AMOUNT'].values.reshape(-1, 1))
df.drop('TX_AMOUNT', axis=1, inplace=True)

# Undersampling data
df_new = df.sample(frac=1)
df_fraud = df_new.loc[df_new['TX_FRAUD'] == 1]
df_nonfraud = df_new.loc[df_new['TX_FRAUD'] == 0][:(df_new['TX_FRAUD'].value_counts()[1])]
equal_amount_of_frauds_df = pd.concat([df_fraud, df_nonfraud])

# Shuffling the dataset subsampled
df_new = equal_amount_of_frauds_df.sample(frac=1, random_state=42)

# Separate target and feature of the subsampled data
X_sub = df_new.drop('TX_FRAUD', axis=1)
y_sub = df_new['TX_FRAUD']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, train_size=0.8)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Checking correlations in the full dataset and the subsampled
sub_df_corr = df_new.corr()
df_corr = df.corr()

# Plotting the correlation matrix
f, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 20))
sns.heatmap(sub_df_corr,
            cmap='coolwarm',
            ax=ax)
ax.set_title('Correlation Matrix \n Subsampled data', fontsize=12)

sns.heatmap(df_corr,
            cmap='coolwarm',
            ax=ax1)
ax1.set_title('Correlation Matrix \n Full data', fontsize=12)

plt.show()

# Looking through the matrix we see that TX_DURING_WEEK has a negative correlation with TX_FRAUD
# And TERMINAL_ID_7_DAY_RISK_WINDOW has a slightly positive correlation with TX_FRAUD

# Now split into feature and target the full dataset
X = df.drop('TX_FRAUD', axis=1)
y = df['TX_FRAUD']

# Train Test Split of the full dataset
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# Classifier with optimal parameter and RandomSearchCV
log_reg_params = {'penalty': [None, 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

# Implementing SMOTE Technique in the Logistical Regression method.
for train, test in sss.split(original_Xtrain, original_ytrain):
    # SMOTE happens during Cross Validation not before, to avoid data leakage
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                        rand_log_reg)
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_

log_reg_preds = best_est.predict(original_Xtest)

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)

# This will be the data we will fit to check oversample results
Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)

# Deep learning model to ovesample and subsample dfs
n_inputs = Xsm_train.shape[1]
n_sub_inputs = X_train.shape[1]

# Create model for both datas
oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

sub_model = Sequential([
    Dense(n_sub_inputs, input_shape=(n_sub_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model with Adam as optimizer and sparse_categorical_crossentropy for final outputs
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
sub_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=100, shuffle=True, verbose=2)
sub_model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2)

# Make predictions in the original x test with the model and turn into classes predictions with np
oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
oversample_fraud_predictions = np.argmax(oversample_predictions, axis=1)
sub_preds = sub_model.predict(original_Xtest, batch_size=200, verbose=0)
sub_fraud_preds = np.argmax(sub_preds, axis=1)

# Input the confusions matrix with the results of classes predicts
oversample_cm = confusion_matrix(original_ytest, oversample_fraud_predictions)
sub_cm = confusion_matrix(original_ytest, sub_fraud_preds)

# Input the confusions matrix with the results of the Logistical Regression predicts
log_reg_cm = confusion_matrix(original_ytest, log_reg_preds)

# Create an 100% accuracy confusion matrix to compare
correct_cm = confusion_matrix(original_ytest, original_ytest)

# Defining a function to plot a confusion matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot all the confusion matrix
labels = ['No Fraud', 'Fraud']
a = plt.figure(figsize=(8,8))

a.add_subplot(221)
plot_confusion_matrix(oversample_cm, labels, title='OverSample (SMOTE) Neural Model \n Confusion Matrix',
                      cmap=plt.cm.Oranges)

a.add_subplot(222)
plot_confusion_matrix(correct_cm, labels, title='Confusion Matrix \n w/ 100% Accuracy', cmap=plt.cm.Greens)

a.add_subplot(223)
plot_confusion_matrix(log_reg_cm, labels, title='Logistical Regression \n Confusion Matrix', cmap=plt.cm.Blues)

a.add_subplot(224)
plot_confusion_matrix(sub_cm, labels, title='Subsampled Neural Model \n Confusion Matrix', cmap=plt.cm.copper)

plt.show()

# Calculate F1 Score and ROC_AUC for preds
from sklearn.metrics import f1_score, roc_auc_score

score_f1 = f1_score(original_ytest, oversample_fraud_predictions)
score_roc = roc_auc_score(original_ytest, oversample_fraud_predictions)
log_reg_f1 = f1_score(original_ytest, log_reg_preds)
log_reg_roc = roc_auc_score(original_ytest, log_reg_preds)
sub_f1 = f1_score(original_ytest, sub_fraud_preds)
sub_roc = roc_auc_score(original_ytest, sub_fraud_preds)

print('---' * 45)
print('F1 Score in neural network SMOTE:', round(score_f1 * 100, 2))
print('ROC Score in neural network SMOTE:', round(score_roc * 100, 2))
print('---' * 45)
print('F1 Score in Logistical Regression SMOTE:', round(log_reg_f1 * 100, 2))
print('ROC Score in Logistical Regression SMOTE:', round(log_reg_roc * 100, 2))
print('---' * 45)
print('F1 Score in Neural Network Subsample:', round(sub_f1 * 100, 2))
print('ROC Score in Neural Network Subsample:', round(sub_roc * 100, 2))

# Testing the best model into a new data
df_test_full = pd.concat((pd.read_pickle(files) for files in pkl_test_files))
df_test = df_test_full[input_features].copy()
df_test['AMOUNT_SCALED'] = scaler.fit_transform(df_test['TX_AMOUNT'].values.reshape(-1, 1))
df_test.drop('TX_AMOUNT', axis=1, inplace=True)

# Split the test dataset
X_test_oficial = df_test.drop('TX_FRAUD', axis=1)
y_test_oficial = df_test['TX_FRAUD']

# Make predictions in the x test with the model and turn into classes predictions with np
oversample_predictions = oversample_model.predict(X_test_oficial, batch_size=200, verbose=0)
oversample_fraud_predictions = np.argmax(oversample_predictions, axis=1)

# Input the confusions matrix with the results of classes predicts
oversample_smote = confusion_matrix(y_test_oficial, oversample_fraud_predictions)

# Create an 100% accuracy confusion matrix to compare
correct_cm = confusion_matrix(y_test_oficial, y_test_oficial)

# Plot the confusion matrix of the test dataset
a = plt.figure(figsize=(8,8))
a.add_subplot(221)
plot_confusion_matrix(oversample_smote, labels, title='OverSample (SMOTE) Neural Model \n Confusion Matrix',
                      cmap=plt.cm.Oranges)

a.add_subplot(222)
plot_confusion_matrix(correct_cm, labels, title='Confusion Matrix \n w/ 100% Accuracy', cmap=plt.cm.Greens)

plt.show()

# Calculate F1 Score and ROC_AUC for preds
score_f1 = f1_score(y_test_oficial, oversample_fraud_predictions)
score_roc = roc_auc_score(y_test_oficial, oversample_fraud_predictions)


print('---' * 45)
print('F1 Score in neural network SMOTE:', round(score_f1 * 100, 2))
print('ROC Score in neural network SMOTE:', round(score_roc * 100, 2))
