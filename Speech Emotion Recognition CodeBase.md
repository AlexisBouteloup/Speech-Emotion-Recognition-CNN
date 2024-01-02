```python
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random
```


```python
from tqdm import tqdm
import os
import librosa
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.signal import wiener
```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras
```

### Data Exploration


```python
def check_sampling_rate_consistency(path):
    
    """
    check if all audio files have the same sampling rate in the path

    Args:
        path (str): path containing the files to check

    Returns:
        str: whether all audios in .wav in the path have the same sampling rate  
        
    """
    sampling_rates = set()
    
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):  
        # Skip if it's not a wav file
            if not file.endswith('.wav'):
                continue
            # Load audio and stretch it to length 1s
            audio_path = os.path.join(root, file)
            audio, sr = librosa.load(path= audio_path, sr=None)
            
            # Add the sampling rate to the set
            sampling_rates.add(sr)

    # If there is only one unique sampling rate, all samples have the same rate
    if len(sampling_rates) == 1:
        print(f"All audio samples have the same sampling rate: {sampling_rates.pop()} Hz")
    else:
        print("Audio samples have different sampling rates.")

# Example usage
audio_folder_path = 'C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database'
check_sampling_rate_consistency(audio_folder_path)



```

--> All audio samples have the same sampling rate: 44100 Hz


```python
path = 'C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database'
lengths=[]
for root, dirs, files in os.walk(path):
    for file in tqdm(files):  
    # Skip if it's not a wav file
        if not file.endswith('.wav'):
            continue
        # Load audio 
        audio_path = os.path.join(root, file)
        audio, sr = librosa.load(path=audio_path, sr=None)
        lengths.append(len(audio)/sr)
        if len(audio)>11*sr:
            print(audio_path)
#print(lengths)
```


```python
sns.histplot(lengths,color='pink')
plt.gca().set_title('length distribution of the recording (in seconds)')
```


```python
print(np.mean(lengths))
np.median(lengths)
```

### Data Preprocessing and Mel Spectrogam 


```python
def add_spect_to_features(feature_list, label_list, audio, file, sr, augment=False): #directly after loading audio
        
    """
    Apply this function directly after loading audio
    Computes the Mel Spectogram of an audio + normalization

    Args:
        feature_list: list of features (audios) already convert to Mel spectograms, to append with new audio's spectogram
        label_list: corresponding label of emotion 
        audio: audio file loaded with librosa.load()
        file: path to the file
        sr: sampling rate
        augment (boolean): True to augment the dataset with pitch shifting

    Returns:
        feature_list, label_list (list): features and labels lists updated with a new subsample/segment of recording  
        
    """
    
    audio_norm = librosa.util.normalize(audio)
    # Calculate features and get the label from the filename
    mels = librosa.feature.melspectrogram(y=audio_norm, sr=sr, n_fft=2048, hop_length=1024)
    mels_db = librosa.power_to_db(S=mels, ref=1.0)
       
    #print(mels_db.shape)
    mels_db = mels_db.reshape((128,130,1))
    feature_list.append(mels_db)
    label_list.append(file[0])
    
    if augment==True:
        audio_pitch_1 = librosa.effects.pitch_shift(audio_norm, sr=sr, bins_per_octave=12,n_steps=-4)
        add_spect_to_features(feature_list, label_list, audio_pitch_1, augment=False)
        audio_pitch_2 = librosa.effects.pitch_shift(audio_norm, sr=sr, bins_per_octave=12,n_steps=-4)
        add_spect_to_features(feature_list, label_list, audio_pitch_2, augment=False)
    return feature_list, label_list
```


```python
def audio_preprocessing (y, feature_list, label_list, file, sr, augment):
    
    """
    Cuts segments of equal duration (=3s) into the recordings: number of segments depends on recording's length
    Uses the add_spect_to_features to compute their Mel spectogram

    Args:
        y: audio file loaded with librosa.load()
        feature_list: list of features (audios) already convert to Mel spectograms, to append with new audio's spectogram
        label_list: corresponding label of emotion 
        file: path to the file
        sr: sampling rate of audio
        augment (boolean): True to augment the dataset with pitch shifting

    Returns:
        feature_list, label_list (list): features and labels lists updated with segment of a new recording 
        
    """
    
    if len(y) <= 3*sr:
        nb_zeros = 3*sr-len(y)
        y = np.concatenate([y, np.zeros(nb_zeros)])
        feature_list, label_list = add_spect_to_features(feature_list, label_list, y, file, sr, augment=augment)
    if len(y) > 3*sr:
        audio_1 = y[:3*sr]
        audio_2 = y[-3*sr:]
        feature_list, label_list = add_spect_to_features(feature_list, label_list, audio_1, file, sr, augment=augment)
        feature_list, label_list = add_spect_to_features(feature_list, label_list, audio_2, file, sr, augment=augment)
    if 6*sr < len(y) <= 9*sr:
        middle=len(y)/2
        audio_3 = y[int(middle-1.5*sr):int(middle+1.5*sr)]
        feature_list, label_list = add_spect_to_features(feature_list, label_list, audio_3, file, sr, augment=augment)
        #print(len(y))
    if len(y) > 9*sr:
        middle=len(y)/2
        audio_3 = y[int(middle-3*sr):int(middle)]
        audio_4 = y[int(middle):int(middle+3*sr)]
        feature_list, label_list = add_spect_to_features(feature_list, label_list, audio_3, file, sr, augment=augment)
        feature_list, label_list = add_spect_to_features(feature_list, label_list, audio_4, file, sr, augment=augment)
    return feature_list, label_list
```


```python
def labels_encoding(labels):
        
    """
    One-hot encodings of string labels to the rigth format to feed the CNN

    Args:
        labels(list of strings): list of labels associated to each audio signal

    Returns:
        encoded_labels(list): one-hot encoded labels 
        
    """
    
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels = np.array([label_mapping[label] for label in labels])
    # One-hot encode labels
    encoded_labels = to_categorical(labels, num_classes=len(label_mapping))
    return encoded_labels
```


```python
def generate_train_test(augment=False, test_frac=0.2, path='C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database'):
    
    """
    uses audio_preprocessing, add_spect_to_features, and labels_encoding to prepare the raw data and generates the train and 
    test sets that will be used for CNN

    Args:
        augment (boolean): True to augment the dataset with pitch shifting
        test_frac(float): expected proportion of the recordings to be assigned to test set
        path(str): path to the folder containing the audio files 

    Returns:
        train_features, train_labels, test_features, test_labels (list): list if features and labels for both train et test sets
        to use in CNN training and assessment
        
    """
    
    train_feature_list = []
    train_label_list = []

    test_feature_list = []
    test_label_list = []

    # Iterate over all files in given source path

    for root, dirs, files in os.walk(path):
        for file in tqdm(files):  
        # Skip if it's not a wav file
            if not file.endswith('.wav'):
                continue
            # Load audio and stretch it to length 1s
            #print(root)
            #print(file)
            audio_path = os.path.join(root, file)
            #print(audio_path)
            y, sr = librosa.load(path=audio_path, sr=None)
            #Wiener filter
            #y = wiener(y)
            #print(len(y)/sr)

            #to choose between train and test set
            rand = random.uniform(0, 1)
            if rand < test_frac:
                test_feature_list, test_label_list = audio_preprocessing (y, test_feature_list, test_label_list, file, sr, augment=False)
                #augment=False as we never augment the test set!
            else:
                train_feature_list, train_label_list = audio_preprocessing (y, train_feature_list, train_label_list, file, sr, augment=augment)

    train_features = np.array(train_feature_list)
    train_labels = np.array(train_label_list)
    train_labels=labels_encoding(train_labels)

    test_features = np.array(test_feature_list)
    test_labels = np.array(test_label_list)
    test_labels=labels_encoding(test_labels)

    
    return train_features, train_labels, test_features, test_labels
```


```python
#on my data:
train_features, train_labels, test_features, test_labels = generate_train_test(augment=False, test_frac=0.2, path='C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database')
print((len(test_labels)+len(train_labels))/603)
len(test_features)+len(train_features)
```

1108 segments, there are 84% more segments/subsamples of 3 seconds than initial recording (603)

### CNN


```python
model1 = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(128,130,1)),
        keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(5, activation=keras.activations.softmax)
    ])
model1.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model1.save_weights('model1.h5')
print(model1.summary())
```


```python
model2 = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(128,130,1)),
        keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dense(5, activation=keras.activations.softmax)
    ])
model2.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model2.save_weights('model2.h5')
print(model2.summary())
```


```python
model3 = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(128,130,1)),
        keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dense(5, activation=keras.activations.softmax)
    ])
model3.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model3.save_weights('model3.h5')
print(model3.summary())
```


```python
train_labels
```

### Cross validation


```python
def cross_validation(n_folds, model, weigths, augment=False, test_frac=0.2, path='C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database'):
    
    """
    Evaluates a given Keras model by cross-validation 

    Args:
        n_folds(int): number of folds to use for corss-validation. At each iteration, a recording is randomly sent to either
        test or train set.
        model: Keras model to be trained
        weigths: files with initial weigths, e.g. 'model1.h5'
        augment (boolean): True to augment the dataset with pitch shifting
        test_frac(float): expected proportion of the recordings to be assigned to test set
        path(str): path to the folder containing the audio files 

    Returns:
        accuracy(list): accuracy on each of the n_folds trainings
        average_accuracy: average of accuracy of n_folds trainings
        sd_accuracy: standard deviation of accuracy across n_folds trainings
        
    """
    
    accuracy=[]
    
    for i in range(n_folds):

        #reinitialise the weigths
        model.load_weights(weigths)
        
        #Data importation, pre-processing and conversion to Mel Spectograms
        train_features, train_labels, test_features, test_labels = generate_train_test(augment=False, test_frac=0.2, path='C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database')
        
        #CNN training
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        history = model.fit(x=train_features, y=train_labels, validation_split=0.01, epochs=20, batch_size=64) 
        ##validation_split=0.01 as 1 cannot put 0, but this is useless as their is obvious data leakage between train
        ##and validation set (not with test set, of course). Then the validation_set is not used

        #Trained CNN used for classification
        y_predicted = np.argmax(model.predict(x=test_features), axis=1)
        y_true = np.argmax(test_labels, axis=1)

        accuracy.append((y_predicted==y_true).sum()/len(y_true))
        print(accuracy)
        average_accuracy = sum(accuracy)/n_folds
        sd_accuracy = np.std(accuracy)
    print('final accuracy',accuracy, 'average=', average_accuracy, 'sd=', sd_accuracy)
    return accuracy, average_accuracy, sd_accuracy

```


```python
#exemple
cross_validation(2, model1, 'model1.h5')
```

### Confusion matrices (Recall and precision)


```python
def confusion_matrices(test_features, test_labels, model): 
    
    """
    prints confusion matrices for already trained model for test set 

    Args:
        test_features(list): features of test set
        test_labels(list): labels of test set
        model: Keras model already trained
        
    """
    
    y_predicted = np.argmax(model.predict(x=test_features), axis=1)
    y_true = np.argmax(test_labels, axis=1)

    print((y_predicted==y_true).sum()/len(y_true))
    print(len(y_true))

    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, normalize='true',cmap='Reds') #recall
    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, normalize='pred',cmap='Blues') #precision

    label_names=['anger','disgust','fear','happiness','sadness']
    confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_predicted) #not normalized
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    sns.heatmap(confusion_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
```

### Test audio representations on one recording


```python
y, sr = librosa.load('C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database/disgust/d01 (1).wav')  # Load audio file using Librosa

plt.plot(y);
plt.title('Audio Signal');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');
```


```python
audio_path = 'C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database/disgust/d01 (1).wav'
audio, sr = librosa.load(path=audio_path, sr=None)
audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr*0.1)
#print(len(audio))
audio = librosa.util.normalize(audio)
# Calculate features and get the label from the filename
mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
print(mels.shape)
mels_db = librosa.power_to_db(S=mels, ref=1.0)
print(mels_db.shape)
librosa.display.specshow(mels_db, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Mel Spectrogram disgust');
plt.colorbar(format='%+2.0f dB');
```
WIENER

```python
y, sr = librosa.load('C:/Users/jadea/Downloads/Acted Emotional Speech Dynamic Database/disgust/d01 (1).wav')  # Load audio file using Librosa

# Apply Wiener filter
enhanced_signal = wiener(y)

plt.plot(enhanced_signal);
plt.title('Signal with Wiener filter');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');
```


```python
n_fft = 2048
ft = np.abs(librosa.stft(y, hop_length = len(y)+1))
#ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');
```


```python
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram');
```


```python
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
mfccs = librosa.power_to_db(mfccs, ref=np.max)
librosa.display.specshow(mfccs, y_axis='mel', fmax=8000, x_axis='time');
plt.title('MFCC Spectrogram');
plt.colorbar(format='%+2.0f dB');
```
