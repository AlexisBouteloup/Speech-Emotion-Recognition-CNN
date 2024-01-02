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

    WARNING:tensorflow:From C:\Users\jadea\anaconda3\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    


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

    0it [00:00, ?it/s]
    100%|██████████| 121/121 [00:04<00:00, 25.95it/s]
    100%|██████████| 122/122 [00:00<00:00, 215.01it/s]
    100%|██████████| 120/120 [00:00<00:00, 230.54it/s]
    100%|██████████| 119/119 [00:00<00:00, 198.21it/s]
    100%|██████████| 121/121 [00:00<00:00, 208.63it/s]

    All audio samples have the same sampling rate: 44100 Hz
    

    
    

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
    # As labels are not integer, we cannot directly use  tf.one_hot, we first have to convert them to numerical.
    
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
    #used audio_preprocessing, add_spect_to_features, and labels_encoding to prepare the data and generates the train and 
    #test sets that will be used for CNN
    
    train_feature_list = []
    train_label_list = []

    test_feature_list = []
    test_label_list = []

    # Iterate over all files in given source path
    print('Preparing feature dataset and labels.')

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

    Preparing feature dataset and labels.
    

    0it [00:00, ?it/s]
    100%|██████████| 121/121 [00:02<00:00, 44.81it/s]
    100%|██████████| 122/122 [00:03<00:00, 39.76it/s]
    100%|██████████| 120/120 [00:02<00:00, 50.57it/s]
    100%|██████████| 119/119 [00:02<00:00, 47.50it/s]
    100%|██████████| 121/121 [00:02<00:00, 46.19it/s]

    1.8374792703150913
    

    
    




    1108



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

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_11 (Conv2D)          (None, 128, 130, 16)      160       
                                                                     
     max_pooling2d_11 (MaxPooli  (None, 64, 65, 16)        0         
     ng2D)                                                           
                                                                     
     conv2d_12 (Conv2D)          (None, 64, 65, 32)        4640      
                                                                     
     max_pooling2d_12 (MaxPooli  (None, 32, 32, 32)        0         
     ng2D)                                                           
                                                                     
     flatten_4 (Flatten)         (None, 32768)             0         
                                                                     
     dropout_4 (Dropout)         (None, 32768)             0         
                                                                     
     dense_11 (Dense)            (None, 64)                2097216   
                                                                     
     dense_12 (Dense)            (None, 5)                 325       
                                                                     
    =================================================================
    Total params: 2102341 (8.02 MB)
    Trainable params: 2102341 (8.02 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    None
    


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

    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_17 (Conv2D)          (None, 128, 130, 16)      160       
                                                                     
     max_pooling2d_17 (MaxPooli  (None, 64, 65, 16)        0         
     ng2D)                                                           
                                                                     
     conv2d_18 (Conv2D)          (None, 64, 65, 32)        4640      
                                                                     
     max_pooling2d_18 (MaxPooli  (None, 32, 32, 32)        0         
     ng2D)                                                           
                                                                     
     conv2d_19 (Conv2D)          (None, 32, 32, 32)        9248      
                                                                     
     max_pooling2d_19 (MaxPooli  (None, 16, 16, 32)        0         
     ng2D)                                                           
                                                                     
     flatten_6 (Flatten)         (None, 8192)              0         
                                                                     
     dropout_6 (Dropout)         (None, 8192)              0         
                                                                     
     dense_17 (Dense)            (None, 64)                524352    
                                                                     
     dense_18 (Dense)            (None, 32)                2080      
                                                                     
     dense_19 (Dense)            (None, 5)                 165       
                                                                     
    =================================================================
    Total params: 540645 (2.06 MB)
    Trainable params: 540645 (2.06 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    None
    


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

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_13 (Conv2D)          (None, 128, 130, 16)      160       
                                                                     
     max_pooling2d_13 (MaxPooli  (None, 64, 65, 16)        0         
     ng2D)                                                           
                                                                     
     conv2d_14 (Conv2D)          (None, 64, 65, 32)        4640      
                                                                     
     max_pooling2d_14 (MaxPooli  (None, 32, 32, 32)        0         
     ng2D)                                                           
                                                                     
     conv2d_15 (Conv2D)          (None, 32, 32, 64)        18496     
                                                                     
     max_pooling2d_15 (MaxPooli  (None, 16, 16, 64)        0         
     ng2D)                                                           
                                                                     
     conv2d_16 (Conv2D)          (None, 16, 16, 128)       73856     
                                                                     
     max_pooling2d_16 (MaxPooli  (None, 8, 8, 128)         0         
     ng2D)                                                           
                                                                     
     flatten_5 (Flatten)         (None, 8192)              0         
                                                                     
     dropout_5 (Dropout)         (None, 8192)              0         
                                                                     
     dense_13 (Dense)            (None, 128)               1048704   
                                                                     
     dense_14 (Dense)            (None, 64)                8256      
                                                                     
     dense_15 (Dense)            (None, 32)                2080      
                                                                     
     dense_16 (Dense)            (None, 5)                 165       
                                                                     
    =================================================================
    Total params: 1156357 (4.41 MB)
    Trainable params: 1156357 (4.41 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    None
    


```python
train_labels
```


```python
history = model1.fit(x=train_features, y=train_labels, validation_split=0.01, epochs=20, batch_size=64)
#if doesn't work: re-run the cell defining model1
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
    
    #use augment=True for data augmentation
    #re-define the model before runing each time
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
cross_validation(2, model2, 'model2.h5')
```

    Preparing feature dataset and labels.
    

    0it [00:00, ?it/s]
    100%|██████████| 121/121 [00:02<00:00, 44.99it/s]
    100%|██████████| 122/122 [00:02<00:00, 44.26it/s]
    100%|██████████| 120/120 [00:02<00:00, 50.53it/s]
    100%|██████████| 119/119 [00:02<00:00, 40.52it/s]
    100%|██████████| 121/121 [00:03<00:00, 37.94it/s]
    

    Epoch 1/20
    14/14 [==============================] - 4s 182ms/step - loss: 2.5652 - accuracy: 0.2082 - val_loss: 1.4343 - val_accuracy: 0.7778
    Epoch 2/20
    14/14 [==============================] - 2s 168ms/step - loss: 1.5600 - accuracy: 0.2651 - val_loss: 1.1762 - val_accuracy: 0.6667
    Epoch 3/20
    14/14 [==============================] - 2s 166ms/step - loss: 1.4200 - accuracy: 0.3663 - val_loss: 1.3116 - val_accuracy: 0.5556
    Epoch 4/20
    14/14 [==============================] - 2s 164ms/step - loss: 1.2340 - accuracy: 0.4676 - val_loss: 1.0629 - val_accuracy: 0.5556
    Epoch 5/20
    14/14 [==============================] - 2s 174ms/step - loss: 1.0999 - accuracy: 0.5734 - val_loss: 1.1860 - val_accuracy: 0.6667
    Epoch 6/20
    14/14 [==============================] - 2s 169ms/step - loss: 0.9825 - accuracy: 0.6212 - val_loss: 0.9626 - val_accuracy: 0.7778
    Epoch 7/20
    14/14 [==============================] - 2s 168ms/step - loss: 0.8448 - accuracy: 0.6542 - val_loss: 0.7366 - val_accuracy: 0.6667
    Epoch 8/20
    14/14 [==============================] - 2s 167ms/step - loss: 0.7275 - accuracy: 0.7383 - val_loss: 1.4464 - val_accuracy: 0.6667
    Epoch 9/20
    14/14 [==============================] - 2s 166ms/step - loss: 0.5990 - accuracy: 0.7838 - val_loss: 1.7072 - val_accuracy: 0.6667
    Epoch 10/20
    14/14 [==============================] - 2s 165ms/step - loss: 0.4584 - accuracy: 0.8373 - val_loss: 2.3138 - val_accuracy: 0.6667
    Epoch 11/20
    14/14 [==============================] - 2s 171ms/step - loss: 0.3954 - accuracy: 0.8623 - val_loss: 1.5933 - val_accuracy: 0.6667
    Epoch 12/20
    14/14 [==============================] - 2s 168ms/step - loss: 0.3297 - accuracy: 0.8692 - val_loss: 0.9084 - val_accuracy: 0.7778
    Epoch 13/20
    14/14 [==============================] - 2s 172ms/step - loss: 0.2646 - accuracy: 0.9033 - val_loss: 1.1396 - val_accuracy: 0.6667
    Epoch 14/20
    14/14 [==============================] - 2s 170ms/step - loss: 0.2037 - accuracy: 0.9329 - val_loss: 1.7701 - val_accuracy: 0.6667
    Epoch 15/20
    14/14 [==============================] - 2s 168ms/step - loss: 0.2185 - accuracy: 0.9170 - val_loss: 1.9025 - val_accuracy: 0.6667
    Epoch 16/20
    14/14 [==============================] - 2s 166ms/step - loss: 0.1652 - accuracy: 0.9454 - val_loss: 1.0390 - val_accuracy: 0.7778
    Epoch 17/20
    14/14 [==============================] - 2s 165ms/step - loss: 0.1454 - accuracy: 0.9522 - val_loss: 1.6232 - val_accuracy: 0.6667
    Epoch 18/20
    14/14 [==============================] - 2s 174ms/step - loss: 0.1052 - accuracy: 0.9681 - val_loss: 1.9575 - val_accuracy: 0.6667
    Epoch 19/20
    14/14 [==============================] - 2s 174ms/step - loss: 0.0924 - accuracy: 0.9704 - val_loss: 2.1024 - val_accuracy: 0.6667
    Epoch 20/20
    14/14 [==============================] - 2s 179ms/step - loss: 0.1231 - accuracy: 0.9545 - val_loss: 1.5533 - val_accuracy: 0.6667
    7/7 [==============================] - 0s 32ms/step
    [0 3 2 3 0 0 0 2 0 0 3 0 0 0 0 0 0 0 2 2 2 3 0 0 3 3 0 3 3 2 3 2 3 0 0 0 0
     2 1 3 2 2 1 1 1 1 1 1 2 2 3 2 4 4 2 1 2 1 1 0 0 1 2 1 1 2 1 0 1 0 1 4 1 1
     1 3 1 3 2 3 3 3 1 3 2 1 2 2 0 2 3 2 4 2 2 2 1 2 1 2 1 2 2 4 2 2 2 1 2 1 2
     2 2 3 2 2 2 4 2 3 3 2 4 2 2 3 2 3 1 1 3 1 3 0 3 3 2 3 3 3 3 3 2 2 3 3 3 3
     3 0 3 2 2 3 2 2 2 1 1 0 2 2 3 4 4 2 3 3 0 0 3 3 2 3 0 4 4 4 4 4 4 2 2 4 4
     4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 2 4 4 4 4 4 4 4 4 4 4]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4
     4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
    [0.6045454545454545]
    Preparing feature dataset and labels.
    

    0it [00:00, ?it/s]
    100%|██████████| 121/121 [00:02<00:00, 48.79it/s]
    100%|██████████| 122/122 [00:03<00:00, 39.99it/s]
    100%|██████████| 120/120 [00:02<00:00, 47.99it/s]
    100%|██████████| 119/119 [00:02<00:00, 40.34it/s]
    100%|██████████| 121/121 [00:03<00:00, 37.99it/s]
    

    Epoch 1/20
    14/14 [==============================] - 4s 182ms/step - loss: 2.2777 - accuracy: 0.2162 - val_loss: 1.1541 - val_accuracy: 1.0000
    Epoch 2/20
    14/14 [==============================] - 2s 161ms/step - loss: 1.5366 - accuracy: 0.2962 - val_loss: 0.9477 - val_accuracy: 0.7778
    Epoch 3/20
    13/14 [==========================>...] - ETA: 0s - loss: 1.4091 - accuracy: 0.3714


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[40], line 2
          1 #exemple
    ----> 2 cross_validation(2, model2, 'model2.h5')
    

    Cell In[37], line 37, in cross_validation(n_folds, model, weigths, augment, test_frac, path)
         34 #CNN training
         35 model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    ---> 37 history = model.fit(x=train_features, y=train_labels, validation_split=0.01, epochs=20, batch_size=64) 
         38 ##validation_split=0.01 as 1 cannot put 0, but this is useless as their is obvious data leakage between train
         39 ##and validation set (not with test set, of course). Then the validation_set is not used
         40 
         41 #Trained CNN used for classification
         42 y_predicted = np.argmax(model.predict(x=test_features), axis=1)
    

    File ~\anaconda3\Lib\site-packages\keras\src\utils\traceback_utils.py:65, in filter_traceback.<locals>.error_handler(*args, **kwargs)
         63 filtered_tb = None
         64 try:
    ---> 65     return fn(*args, **kwargs)
         66 except Exception as e:
         67     filtered_tb = _process_traceback_frames(e.__traceback__)
    

    File ~\anaconda3\Lib\site-packages\keras\src\engine\training.py:1807, in Model.fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1799 with tf.profiler.experimental.Trace(
       1800     "train",
       1801     epoch_num=epoch,
       (...)
       1804     _r=1,
       1805 ):
       1806     callbacks.on_train_batch_begin(step)
    -> 1807     tmp_logs = self.train_function(iterator)
       1808     if data_handler.should_sync:
       1809         context.async_wait()
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\util\traceback_utils.py:150, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        148 filtered_tb = None
        149 try:
    --> 150   return fn(*args, **kwargs)
        151 except Exception as e:
        152   filtered_tb = _process_traceback_frames(e.__traceback__)
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\polymorphic_function.py:832, in Function.__call__(self, *args, **kwds)
        829 compiler = "xla" if self._jit_compile else "nonXla"
        831 with OptionalXlaContext(self._jit_compile):
    --> 832   result = self._call(*args, **kwds)
        834 new_tracing_count = self.experimental_get_tracing_count()
        835 without_tracing = (tracing_count == new_tracing_count)
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\polymorphic_function.py:868, in Function._call(self, *args, **kwds)
        865   self._lock.release()
        866   # In this case we have created variables on the first call, so we run the
        867   # defunned version which is guaranteed to never create variables.
    --> 868   return tracing_compilation.call_function(
        869       args, kwds, self._no_variable_creation_config
        870   )
        871 elif self._variable_creation_config is not None:
        872   # Release the lock early so that multiple threads can perform the call
        873   # in parallel.
        874   self._lock.release()
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\tracing_compilation.py:139, in call_function(args, kwargs, tracing_options)
        137 bound_args = function.function_type.bind(*args, **kwargs)
        138 flat_inputs = function.function_type.unpack_inputs(bound_args)
    --> 139 return function._call_flat(  # pylint: disable=protected-access
        140     flat_inputs, captured_inputs=function.captured_inputs
        141 )
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\concrete_function.py:1323, in ConcreteFunction._call_flat(self, tensor_inputs, captured_inputs)
       1319 possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
       1320 if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
       1321     and executing_eagerly):
       1322   # No tape is watching; skip to running the function.
    -> 1323   return self._inference_function.call_preflattened(args)
       1324 forward_backward = self._select_forward_and_backward_functions(
       1325     args,
       1326     possible_gradient_type,
       1327     executing_eagerly)
       1328 forward_function, args_with_tangents = forward_backward.forward()
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\atomic_function.py:216, in AtomicFunction.call_preflattened(self, args)
        214 def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
        215   """Calls with flattened tensor inputs and returns the structured output."""
    --> 216   flat_outputs = self.call_flat(*args)
        217   return self.function_type.pack_output(flat_outputs)
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\polymorphic_function\atomic_function.py:251, in AtomicFunction.call_flat(self, *args)
        249 with record.stop_recording():
        250   if self._bound_context.executing_eagerly():
    --> 251     outputs = self._bound_context.call_function(
        252         self.name,
        253         list(args),
        254         len(self.function_type.flat_outputs),
        255     )
        256   else:
        257     outputs = make_call_op_in_graph(
        258         self,
        259         list(args),
        260         self._bound_context.function_call_options.as_attrs(),
        261     )
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\context.py:1486, in Context.call_function(self, name, tensor_inputs, num_outputs)
       1484 cancellation_context = cancellation.context()
       1485 if cancellation_context is None:
    -> 1486   outputs = execute.execute(
       1487       name.decode("utf-8"),
       1488       num_outputs=num_outputs,
       1489       inputs=tensor_inputs,
       1490       attrs=attrs,
       1491       ctx=self,
       1492   )
       1493 else:
       1494   outputs = execute.execute_with_cancellation(
       1495       name.decode("utf-8"),
       1496       num_outputs=num_outputs,
       (...)
       1500       cancellation_manager=cancellation_context,
       1501   )
    

    File ~\anaconda3\Lib\site-packages\tensorflow\python\eager\execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         51 try:
         52   ctx.ensure_initialized()
    ---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         54                                       inputs, attrs, num_outputs)
         55 except core._NotOkStatusException as e:
         56   if name is not None:
    

    KeyboardInterrupt: 


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
    
    #model has to be already trained
    
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


```python
confusion_matrices(test_features, test_labels, model1)
```

### Test on one recording


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


```python

```


```python

    
```
