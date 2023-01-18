# Credit Card Fraud Detector

* In this activity, i was using the dataset from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to evaluate the best credit card fraud detector model, 
but it hadn't a new dataset to check if the model was overfitted, 'cause it was modified using PCA transformation, due to confidenciality issues.

* It reccomended to visist this [handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html),
where we can find a simulator to transaction data. And i used the data available from [github](https://github.com/Fraud-Detection-Handbook/simulated-data-transformed)
to test and evaluate the best model made.

* Some really nice ideas was absorbed by this brilliant [notebook](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets), like the plot confusion matrix function, the SMOTE method to oversample the dataset, and doing this during cross-validation to avoid dataleakage.
