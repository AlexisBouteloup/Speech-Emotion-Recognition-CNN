# Speech Emotion Recognition (SER) using CNN on AESSD Database

## Objective of the project
The goal of the project is to use Convolutional Neural Networks on Mel spectrograms in order to perform Speech Emotion Recognition (SER) on the Acted Emotional Speech Dynamic Database (AESDD) containing acted audio recordings of emotions in Greek. 

## Materials
I worked on python 3.9 in Anaconda Jupyter Notebook. The following libraries were used in addition to the classical python libraries: Librosa, os, tensorflow, keras, scipy, soundfile, tqdm (to monitor progression the code completion). 

## Data exploration
The database used for this project is the Acted Emotional Speech Dynamic Database (AESDD) that displays 605 recordings of 6 actors speaking 20 different lines/utterances for each of 5 distinct emotions in Greek language – anger, disgust, fear, happiness, sadness. The size of the database is 605. I listened to some of the audios on my computer to familiarize myself with the data and attempted to load the recordings. The audio for the 5th line of actor 3 depicting sadness was unloadable and unreadable, raising the following error:
LibsndfileError: Error opening 'C:/Users/jadea/Acted Emotional Speech Dynamic Database\\Acted Emotional Speech Dynamic Database\\sadness\\s05 (3).wav': Error in WAV file. No 'data' chunk marker.
I removed this recording from the database. The numbers of recordings for each emotion are almost equal, so we can state that the dataset is well balanced. The sampling rate is 44100HZ for all recordings. Concerning the lengths of the audios, underneath is the histogram of the distribution of the durations of audios:







We observe that most of the recordings have lengths comprised between 2 and 8 seconds. The mean duration is 4.10 seconds and the median is 3.78s. I extracted the path of the audio lasting more than 12 seconds: ‘…\sadness\s19 (6).wav’. I listened to it and realised that it was mostly silence, so I decided to remove it. 
The final number of recordings was then 603.
Here is the plot of the signal of the 1st recording of the anger emotion (a01 (1)):
 

## Data preprocessing

### Segments generation
The fact that the durations of the samples are not constant is an issue we must tackle. There are several possibilities: time stretching (in order to match the length of the audios with a target length), or cutting segments of equal length within the recordings. I chose the latter method. As a result, I could increase the number of observation by 84%, by choosing a target duration of 3 seconds (assuming that the emotion is displayed all along the initial recording). 
The number of sub-samples/segments generated from each recording in equal to the upper integer part of the length of recording (in seconds) divided by 3. (see code, function: audio_preprocessing). For instance, if len(audio) is 4, it is decomposed into 2 sub-recordings of duration 3seconds. As a result, some of the sub-samples are overlapping. I chose this method because it is a way to perform data augmentation since the beginning, as we have to acknowledge that 600 observations are not much for training a neural network. This procedure is related to time shifting, that can be performed with the np.roll() method on an audio. Indeed, the end of same of sub-sample will often be the beginning of another one.    

### Train/Test sets
At the beginning of the project, I first pre-processed each recording and generated the sub-samples to create a new database of 1108 observations, and then proceeded to train/test splitting. However, this was problematic as sub-samples from the same initial utterance could end up in train and test sets at the same time. As the sub-samples have audio segments in common, this creates data leakage. The issue was resolved by splitting between train and test set since the importation, and before pre-processing. Thus, all sub-samples stemming from the same recording will either all end up in train, or in test set. The train set represents 80% of the initial recordings.  

### Noise reduction/Normalization
The data was normalized using librosa.util.normalize(audio), meaning that the amplitude values of the signal are scaled in such a way that the maximum and minimum values are between -1 and 1.
For denoising the data, I apply a Wiener filter from the scipy library. 

## Conversion to spectrograms

I used the librosa library in Python to convert my audio signals to Mel spectrograms. First, a Fast Fourier Transform (FFT) is apply with window length of 2048 with a hop length of 1024. Thus, each time point is present in 2 windows. Then the spectrogram is created, here is the spectrogram for one of the full recordings for anger: 

  







As human perceive pitch differences more accurately at lower frequencies than at higher frequencies, the perception of a signal’s frequence variation is not linear for human ear. The Mel spectrogram transforms the classical spectrogram into a representation of the audio information that matches with the hearing of a typical human.
At the right is the Mel spectrogram describing the same recording, and below is the correspondence between frequencies and Mel scale.








It Is the Mel spectrogram images that are used to feed the Convolutional Neural Network.

## Models (CNN): (keras, tensorflow)
The activation functions are Relu, and softmax for the output layer. The optimizer is Adam.

### Model1:
The first model used is the one described underneath: (batch_size = 64 and epochs = 20, dropout_rate = 0.5).
 
### Results Model1:

It yields the following confusion matrices (normalized of true labels at the left = recall, normalized on predicted labels at the right = precision):  
The corresponding labels are (in order: anger – disgust – fear – happiness- sadness)

  
We see that sadness (4) is very well recognized by the model (recall of 82% and 72% precision), anger is also quite well identified (58% recall and precision). However, fear is almost never recognized, and is very often confounded for happiness (51% of the time).

-	Remove Wiener filter
I remarked that the Wiener filter from Scipy creates some issue in my dataset, as it seems not to work properly. I finally chose to remove it, and the confusion matrices get better (same order):

  
We remark that now, all emotions are recognized half of the time, which is way better. It is interesting to note that sadness has a precision of 97% - meaning that when the CNN predicts ‘sadness’, it is very trustworthy. The cross-validation average score over 5 training is 50.45%.



-	Dropout rate
I could note track overfitting using val_accuracy in keras during the training as there was data leakage between the validation set extracted by keras from the train set (problem described earlier). The only reliable measure of accuracy was computed on the real test set after the completion of training. I then wondered if increasing dropout_rate to 0.7 would help. As we can see, especially thank to the recall confusion matrix at the left, it doesn’t help. I finally tried dropout_rate=0.3 but it wasn’t better either. So we keep this rate to 0.5.
  
-	Kernel size
The initial kernel size was 3*3, I changed it for 5*5. The cross-validation accuracy over 5 folds reaches 48.63%, which is lower than 50.45%.
Model2: 














### Results Model2:
Finally, I tried to add a 3rd convolutional layer with 64 filters and a 3rd Max pooling layer. I also added a dense layer. The cross-validation accuracy soared to 58.05%. The standard deviation of accuracy is 0.027. Here are the corresponding confusion matrices (recall at the left in red shades and precision at the right in blue shades):
   

### Model3:
 
### Model3 Results:
I added a 4th CONV and MaxPool layers, and the cross-validation accuracy increases further: 64.12%, sd= 0.057. The last iteration of the cross-validation process showed these confusion matrices:
    
We notice that the labels 2 and 3 (fear and happiness) are the least well recognized. This result was also observed during subsequent trainings of the same model with various train/test splittings.

## Data augmentation
We cannot apply the classical methods for data augmentation for datasets of images. Indeed, for instance, a horizontal flip or a rotation would substantially alter the spectrogram and the sound that it represents. Therefore, I decided to augment the audio signal before computing the Mel spectrogram. 
As explained earlier, the pre-processing step already embeds a step that can be considered part of the data augmentation.
Besides, and in order to improve the previous results, I decided to increase 3x the size of my dataset by applying a data augmentation method. I chose the pitch shifting method that I implemented thanks to the librosa library. For each segment, I added to the dataset 2 modified versions: 1 with a pitch shifting of +1/3 octave and the other with -1/3 octave. The idea was to create the other gender equivalent of each segment. I found out that the males’ voice I usually 1 octave lower than females’ voice. However, when listening to recordings tuned with this setting, it sounded awfully fake, so I decided to transform by only a third of an octave, as it is what seemed the most accurate translation from male to female according to my ear (I listened to the pitch shifted audio with various shifting intensities). It would have been interesting to know in advance if the utterance was spoken by a male or a female, to apply the corresponding transformation. Nonetheless, I did not have this information so I simply applied both shifting (+ and – 1/3 octave) to each recording. This might have created some unrealistic audio. But I do not think it would lower the accuracy of my CNN. 
NB: Of course I did not augment the test set. 

### Results:
The last model (model3) was trained with the augmented dataset, and the results are globally the same than without the last data augmentation step: average= 63.98 and sd= 0.05. This step is resource-consuming and looks useless. 
Conclusion:
Speech Emotion Recognition has been performed on the AESDD dataset using Mel Spectograms and CNN models. The best model (Model3) was able to recognize the 5 emotions with an accuracy of almost 2 thirds (64%), whereas a random classification would reach an accuracy of 20% as there are 5 distinct classes. I believe this result is pretty good for a simple Neural Network design with only a few hundred audio recordings. The data Augmentation step did not bring a concrete improvement to the accuracy of the Model3.

## Possible improvements
### Data usage:
-	Data Leakage: Some utterance where recorded 2 times with the same actor. This problem concerns 14 utterances (among ~600). My work did not take this into account, but this could create data leakage between train and test set, artificially increasing slightly the test accuracy. 
-	The database contains the recordings of solely 6 actors, so the real-world accuracy of the model might be poor. Indeed, for a same emotion and a same actor, different utterances can be distributed in both train and test set. We could assume that those audios, even following a different text, have very similar characteristics, due to the voice intonation of the actors and speech habits that are likely the same across utterances. 
### Model architecture:
-	Increase the depth of Model3 even more
-	LSTM CNNs, use other features of audio recordings such as MFCCs or Chromagram. 
-	For each model, it would have been great to have enough time and computational power to test on more than 5 folds for cross-validation, as the standard deviation of accuracy can be quite high.
### Data augmentation:
-	The SpecAugment algorithm that masks some frequencies and times in the spectrogram could also have been used. I did not have the time but it would have interesting: 




