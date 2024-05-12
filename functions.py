import pandas as pd

TWEETS_FOLDER = './preprocessed_data'
LABELS_FOLDER = './datasets'

TRAIN_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_train.txt'
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

def load_datasets(): 
    train_frame = load_data(tweets_path=TRAIN_TWEETS_PATH, labels_path=TRAIN_LABELS_PATH)
    validation_frame = load_data(tweets_path=VAL_TWEETS_PATH, labels_path=VAL_LABELS_PATH)
    test_frame = load_data(tweets_path=TEST_TWEETS_PATH, labels_path=TEST_LABELS_PATH)
    return train_frame, validation_frame, test_frame
