# Speech-Emotion-Recognition-CNN

<img src="https://media.giphy.com/media/NqhohLDKCaixsl2Ygb/giphy.gif" width="200">

## Objective of the project:
The goal of the project is to use Convolutional neural Networks on Mel spectrogram in order to perform Speech Emotion Recognition (SER) on the Acted Emotional Speech Dynamic Database (AESDD) containing acted audio recordings of emotions in Greek. 

<img src="https://media.giphy.com/media/186DbBrxSQZOmY9NTp/giphy.gif" width="200">

## Materials:
I worked on python 3.9 in Anaconda Jupyter Notebook. The following libraries were used in addition to the classical python libraries: Librosa, os, tensorflow, keras, scipy, soundfile, tqdm (to monitor progression of the code completion).

The database is available on Dagshub: 
Acted Emotional Speech Dynamic Database on DagsHub: https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database

## Methods:
Mel spectograms of segments of the audio recordings where fed into 3 CNN models with more or less deep architectures to perform the emotion classification. 
Different Hyperparameters such as Kernel size and Dropout rate were tested.
Cross-validation was implemented and confusion matrices were printed to assess the models.  
Use of Pitch Shifting as a data augmentation procedure.

![image](https://github.com/JadeArpaliangeas/Speech-Emotion-Recognition-CNN/assets/149436763/f37a04d3-81ba-4503-ad73-5e6a080b55e6)


## Conclusion: 
3 model architecture were tested, with different parameters (kernel_size and dropout_rate).
Data augmentation was attempted, but was declared useless.
CNN architecture with 3 convoluational layers followed each by a max pooling layers, and 2 dense layers achieved good performance.
Almost 2 thirds of the emotion recordings were predicted accurately.
