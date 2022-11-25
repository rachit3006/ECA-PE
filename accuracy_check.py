import pandas as pd
from data_combination import custom_train_test_split
import keras
from keras_vggface import utils
from keras.preprocessing.image import ImageDataGenerator

# path to the folder with images grouped by video name
images = 'data/afnet_fer_combination'

df_train, df_valid = custom_train_test_split()

datagen_valid = ImageDataGenerator(preprocessing_function=utils.preprocess_input)

size_img = (224, 224)
bs = 32

valid_generator = datagen_valid.flow_from_dataframe(dataframe=df_valid,
                                                    directory=images,
                                                    x_col='pth',
                                                    y_col='label',
                                                    target_size=size_img,
                                                    batch_size=bs,
                                                    class_mode='categorical',
                                                    shuffle=False)

labels = valid_generator.labels

model = keras.models.load_model('models/resnet50/weights.h5')

score = model.evaluate(valid_generator)
print("Accuracy :",score[1])