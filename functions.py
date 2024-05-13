import pandas as pd

TWEETS_FOLDER = './preprocessed_data'
LABELS_FOLDER = './datasets'

TRAIN_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_train'
TRAIN_LABELS_PATH = LABELS_FOLDER + '/sentiment/train_labels.txt'
VAL_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_validation.txt'
VAL_LABELS_PATH = LABELS_FOLDER + '/sentiment/val_labels.txt'
TEST_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_test.txt'
TEST_LABELS_PATH = LABELS_FOLDER + '/sentiment/test_labels.txt'

# Load preprocessed tweets
def load_data(tweets_path, labels_path):
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
    return pd.DataFrame({'tweet': tweets, 'label': labels})

def load_datasets(use_joined=True): 
    train_path = TRAIN_TWEETS_PATH
    valid_path = VAL_TWEETS_PATH
    test_path = TEST_TWEETS_PATH
    if use_joined: 
        train_path = train_path + "_joined"
        valid_path = valid_path + "_joined"
        test_path = test_path + "_joined"
        
    train_path = train_path + ".txt"
    valid_path = valid_path + ".txt"
    test_path = test_path  + ".txt"
    train_frame = load_data(tweets_path=train_path, labels_path=TRAIN_LABELS_PATH)
    validation_frame = load_data(tweets_path=valid_path, labels_path=VAL_LABELS_PATH)
    test_frame = load_data(tweets_path=test_path, labels_path=TEST_LABELS_PATH)
    return train_frame, validation_frame, test_frame
