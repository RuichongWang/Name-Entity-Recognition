# Name-Entity-Recognition in Clinical Reords, with Pre-trained Bio-BERT 
This project applies Pycaret and pre-trained Bio-BERT to the health-record de-identification task, achived F1 score of 0.89.
## Repository contents
* FE.ipynb == Feature Engineering
* NER with Pycaret.ipynb == Pycaret Modeling
* BERT.ipynb == Bi-LSTM with Pre-trained bio-bert in the raw data
## DetailsCancel changes
### FE.ipynb
In this notebook I load Word2Vec and BERT for feature extraction. I also added some other liguistic features such as pos tagging.
### NER with Pycaret.ipynb
Pycaret code, it uses SMOTE to solve the imbalanced problem. compared 9 different machine learning models and LightGBM has the best performance.
### BERT.ipynb
Bi-LSTM with Pre-trained bio-bert on the raw data, the performance is not as good as LightGBM because it is difficult to solve calss imbalance in NER task for Neural network.
