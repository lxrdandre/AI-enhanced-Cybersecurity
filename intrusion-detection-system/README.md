Intrusion Detection System Pipelines

Two deep learning pipelines for intrusion detection on TON_IoT data. Each pipeline includes data loading, preprocessing, training, evaluation, and transfer learning.

BiLSTM
The BiLSTM pipeline builds a sequential model for traffic features. It applies categorical encoding, numeric scaling, log transforms for selected columns, and Chi-Squared feature selection. Training uses SMOTE to balance classes and class weights to reduce bias. The base training stage produces a model and preprocessing artifacts. The transfer learning stage loads a pretrained BiLSTM, reuses the fixed feature list, and fine-tunes on a custom dataset using all available classes.

SE-DWNet 
The SE-DWNet pipeline uses a 1D convolutional residual model with squeeze-and-excitation blocks designed for tabular feature sequences. It performs label encoding, Min-Max scaling, mutual information feature selection, and produces a compact feature set used by the model. The base training stage saves the model and full preprocessing pipeline for reproducible inference. The transfer learning stage loads the pretrained model and pipeline, transforms a custom dataset with identical preprocessing, and fine-tunes on all available classes.