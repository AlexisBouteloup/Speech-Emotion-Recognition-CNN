# Speech-Emotion-Recognition-CNN
Objective of the project
The goal of the project is to use Convolutional neural Networks on Mel spectrogram in order to perform Speech Emotion Recognition (SER) on the Acted Emotional Speech Dynamic Database (AESDD) containing acted audio recordings of emotions in Greek. 

Materials:
I worked on python 3.9 in Anaconda Jupyter Notebook. The following libraries were used in addition to the classical python libraries: Librosa, os, tensorflow, keras, scipy, soundfile, tqdm (to monitor progression the code completion).

The database is available on Dagshub: 
Acted Emotional Speech Dynamic Database on DagsHub: https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database

Conclusion: 
3 model architecture were tested, with different parameters (kernel_size and dropout_rate).
Data augmentation was attempting, but was declared useless.
CNN architecture with 3 convoluational layers followed each by a max pooling layers, and 2 dense layers achieved good performance.
Almost 2 thirds of the emotion recordings were predicted accurately.
