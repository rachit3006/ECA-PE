from ast import main
import numpy as np
import pandas as pd
import math
from data_combination import custom_train_test_split
import keras
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.models import model_from_json
from keras.callbacks import *
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score

def weight_(y_train, mu = 0.15):
    counter = dict(Counter(y_train))
    unque_counts_weights = create_class_weight(counter, mu=mu)
    unque_counts_weights_new = unque_counts_weights.items()
    unque_counts_weights_new_sort= dict(sorted(unque_counts_weights_new))
    return unque_counts_weights_new_sort

def create_class_weight(labels_dict,mu=0.15):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

def network(model='resnet50',lr=0.00001):
    resnet50_features = VGGFace(model=model, include_top=False, input_shape=(224, 224, 3), pooling='avg')
    x = keras.layers.Dense(units = 1024, activation = 'relu')(resnet50_features.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(7, activation = 'softmax')(x)         
    model = keras.models.Model(resnet50_features.input, x)
    model.compile(
        optimizer = keras.optimizers.Adam(
            lr=lr, 
            beta_1=0.9, 
            beta_2=0.999, 
            decay=1e-5, 
            epsilon=1e-8, 
            amsgrad=True),
        loss = 'categorical_crossentropy',
        metrics =['acc']
    )
    return model

class Metrics(Callback):
    def __init__(self, verbose=0, patience=0):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.patience = patience
        self.path_save_model = 'models/resnet50/'

    def on_train_begin(self, logs={}):
        self.val_recalls = []
        self.val_recall = 0
        self.stop_flag_recall = 0
        self.num_epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        self.num_epochs += 1
        
        val_predict=self.model.predict_generator(self.model.valid_generator, verbose=1)
        val_predict=np.argmax(val_predict,axis=1)
        val_targ = self.model.valid_generator.labels
        
        score = recall_score(val_targ, val_predict, average='macro')
        self.val_recalls.append(score)
        logs['val_recall'] = score

        if score > self.val_recall:
            self.stop_flag_recall = 0
            self.model.save(self.path_save_model + '/weights2.h5')
            if self.verbose > 0:
                print('Recall improved from {} to {}'.format(self.val_recall, score))
                self.val_recall = score
        else:
            self.stop_flag_recall += 1
            if self.verbose > 0:
                print('Recall did not improve.')

        if self.stop_flag_recall > self.patience:
            if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))
            self.model.stop_training = True
           
        print('max_val_recall {}\n'.format(max(self.val_recalls)))
        print('current_val_recall {}\n'.format(score))

# path to the folder with images grouped by video name
images = 'data/afnet_fer_combination'

df_train, df_valid = custom_train_test_split()

datagen_train = ImageDataGenerator(preprocessing_function=utils.preprocess_input,
                                   rotation_range=40,
                                   horizontal_flip=True,
                                   brightness_range=[0.2,1.0])

datagen_valid = ImageDataGenerator(preprocessing_function=utils.preprocess_input)

size_img = (224,224)
bs = 16

train_generator = datagen_train.flow_from_dataframe(dataframe=df_train,
                                                    directory=images,
                                                    x_col='pth',
                                                    y_col='label',
                                                    target_size=size_img,
                                                    batch_size=bs,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    )

valid_generator = datagen_valid.flow_from_dataframe(dataframe=df_valid,
                                                    directory=images,
                                                    x_col='pth',
                                                    y_col='label',
                                                    target_size=size_img,
                                                    batch_size=bs,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    )

labels_train = train_generator.labels
weight = weight_(labels_train, 0.34)

metrics = Metrics(verbose=1, patience=6)

callbacks = [
             metrics
            ]

model = network()
model.valid_generator = valid_generator

model.fit(
    train_generator,
    epochs = 60,
    verbose = True,
    validation_data=valid_generator,
    callbacks = callbacks,
    class_weight=weight
    )

fer_json = model.to_json()
with open(metrics.path_save_model + "/model2.json", "w") as json_file:
    json_file.write(fer_json)
print('\n' + metrics.path_save_model)