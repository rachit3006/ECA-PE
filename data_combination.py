import os
import pandas as pd
from sklearn.model_selection import train_test_split

def custom_train_test_split():
    images = 'data/afnet_fer_combination'
    filename = 'labels.csv'
    main_df = pd.read_csv('data/affectnet/' + filename)
    main_df = main_df[main_df.label != "contempt"]

    df_train, df_valid = train_test_split(main_df, test_size=0.2)

    fer_emotions = ['anger','disgust','fear','sad','surprise']

    train_dict = {"pth":[], "label":[]}
    test_dict = {"pth":[], "label":[]}

    for emotion in fer_emotions:
        for filename in os.listdir(images+"/train/"+emotion):
            f = os.path.join("train/"+emotion, filename)
            train_dict["pth"].append(f)
            train_dict["label"].append(emotion)
        for filename in os.listdir(images+"/test/"+emotion):
            f = os.path.join("test/"+emotion, filename)
            test_dict["pth"].append(f)
            test_dict["label"].append(emotion)

    df_valid = pd.concat([df_valid,pd.DataFrame(test_dict)], ignore_index = True)
    df_train = pd.concat([df_train,pd.DataFrame(train_dict)], ignore_index = True)
    
    return [df_train, df_valid]