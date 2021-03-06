import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pydub import AudioSegment # pip install pydub
import splitfolders # pip install split-folders
import warnings
warnings.filterwarnings('ignore')
# Part 1 - using AudioSegment from pydub to split the audio files

   # Create Empty Directories
genres = 'blues classical country disco pop hiphop metal reggae rock'.split()
for g in genres:
  audio = os.path.join('Data3sec',f'{g}')
  os.makedirs(audio)
  # Split each audio file into 10 parts of 3 seconds so we get 1000 audio file each genre, 10000 in total
i = 0
for g in genres:
  j=0
  #print(f"{g}")
  for filename in os.listdir(os.path.join('Data/genres_original',f"{g}")):

    song  =  os.path.join(f'Data/genres_original/{g}',f'{filename}')
    j = j+1
    for w in range(0,10):
      i = i+1
      t1 = 3*(w)*1000
      t2 = 3*(w+1)*1000
      newAudio = AudioSegment.from_wav(song)
      new = newAudio[t1:t2]
      new.export(f'Data3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")


# Part 2 : Genrating Spectograms using Librosa

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(10,10))
genres = 'metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'Spec_data/{g}').mkdir(parents = True, exist_ok = True)
    for filename in os.listdir(f'./Data3sec/{g}'):
        songname = f'./Data3sec/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y,NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides ='default',mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'Spec_data/{g}/{filename[:-3].replace(".","")}.png')
        plt.clf()       

# Part 3 : Preprocess and Split Data into training set and validation set 
        
splitfolders.ratio('Spec_data', output="Data_split", seed=1337, ratio=(0.8,0.2))  
       
    # Create data generators for both training and testing set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Data_split/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 color_mode='rgba',
                                                 class_mode = 'categorical')


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Data_split/val',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            color_mode='rgba',
                                            class_mode = 'categorical')

# Part 4 - Building the CNN

# Initialising the CNN
classes =10
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 4]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes)))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set and Save our model into HDF5 file
cnn.fit(x = training_set, validation_data = test_set, epochs = 70)
cnn.save('MyMusicRecognizer2.h5')

new_model = tf.keras.models.load_model('MyMusicRecognizer.h5')
new_model.summary()
loss, acc = new_model.evaluate(training_set,  verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

from keras.preprocessing import image
test_image = image.load_img('sample2.png',color_mode='rgba', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

training_set.class_indices
result = new_model.predict(test_image)
print(result) 

