import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from gensim.models import Word2Vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import datapath
import gensim.downloader as api

import spacy

from datetime import datetime


SPACY_CORPUS = "en_core_web_sm"

TWEETS_FOLDER = './preprocessed_data'
LABELS_FOLDER = './datasets'

TRAIN_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_train'
TRAIN_LABELS_PATH = LABELS_FOLDER + '/sentiment/train_labels.txt'
VAL_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_validation'
VAL_LABELS_PATH = LABELS_FOLDER + '/sentiment/val_labels.txt'
TEST_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_test'
TEST_LABELS_PATH = LABELS_FOLDER + '/sentiment/test_labels.txt'

# Load preprocessed tweets
def load_data(tweets_path, labels_path):
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
    return pd.DataFrame({'tweet': tweets, 'label': labels})

def load_datasets(vectorizer_name="", use_joined=True): 
    if vectorizer_name in {"word2vec", "fasttext"}:
        use_joined = False

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



class VectorizerWrapper():
    # TODO It is not beautiful but it works usually one would need the vectorizers to be fixed variables 
    # and the vectorizers implementing fit_transform and _transform
    sklearn_vectorizers = {"tfidf", "count"}
    
    def __init__(self, vectorizer_name="tfidf") -> None:
        self.vectorizer_name = vectorizer_name
        self.vectorizer = None
        self.dims = None
    
    def fit_transform(self, data, **kwargs): 
        if self.vectorizer_name == "count":
            self.vectorizer = CountVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        elif self.vectorizer_name == "spacy": 
            nlp = spacy.load(SPACY_CORPUS, disable=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"])
            self.vectorizer = nlp
            return self.transform(data) 
        elif self.vectorizer_name == "word2vec":
            # Load the pre-trained Word2Vec model
            #model_path = "./datasets/glove.twitter.27B.25d.txt"  # Replace 'path_to_pretrained_model' with the actual path to the model file
            #word2vec_model =  KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
            self.dims = 100
            word2vec_model = api.load("glove-twitter-"+ str(self.dims))
            # word2vec_model = Word2Vec.load(model_path)
            self.vectorizer = word2vec_model 
            return self.transform(data) 
        elif self.vectorizer_name == "doc2vec":
            def tagged_document(data_ls_ls):
                for i, word_list in enumerate(data_ls_ls):
                    yield TaggedDocument(word_list, [i])
            data = list(tagged_document(data))
            doc2vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
            
            doc2vec_model.build_vocab(data)
            doc2vec_model.train(data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

            self.vectorizer = doc2vec_model
            return self.transform(data)
        elif self.vectorizer_name == "fasttext":
            import fasttext
            import fasttext.util
            model_path = "./datasets/cc.en.300.bin"
            fasttext_model = fasttext.load_model(model_path)
            self.dims= 100 # setting dims here manually 
            fasttext.util.reduce_model(fasttext_model, self.dims)
            print(fasttext_model.get_dimension())
            self.vectorizer = fasttext_model
            return self.transform(data)
        elif self.vectorizer_name == "tfidf": 
            self.vectorizer = TfidfVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        else:
            self.vectorizer = TfidfVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        
    def transform(self, data): 
        if self.vectorizer_name in self.sklearn_vectorizers: 
            return self.vectorizer.transform(data)
        elif self.vectorizer_name == "spacy": 
            print("using spacy")
            #return [self.vectorizer(doc).vector for doc in data]
            return [doc.vector for doc in self.vectorizer.pipe(data)]
        elif self.vectorizer_name == "word2vec": 
            print("using word2vec")
            matrix = []
            for doc in data: 
                word_vectors = []
                for word in doc:
                    try: 
                        word_vectors.append(self.vectorizer[word])
                    except KeyError:
                        pass
                #word_vectors = [self.vectorizer[word] for word in doc]
                if len(word_vectors) == 0: 
                    word_vectors.append([0] * self.dims)
                matrix.append(np.mean(word_vectors, axis=0))
            return matrix
        elif self.vectorizer_name == "doc2vec": 
            print("using doc2vec")
            matrix = [self.vectorizer.infer_vector(doc) for doc in data]
            return matrix
        elif self.vectorizer_name == "fasttext":
            print("using fasttext")
            matrix = []
            for doc in data: 
                word_vectors = [self.vectorizer.get_word_vector(word) for word in doc]
                matrix.append(np.mean(word_vectors, axis=0))
            return matrix 
        else: 
            return self.vectorizer.transform(data) 


def plot_confusion_matrix(model, X, y, additional_title:str):
    titles_options = [
    ("Confusion matrix, without normalization" + additional_title, None),
    ("Normalized confusion matrix"+ additional_title, "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=["negative", "neutral", "positive"],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()


def analyze_model(model, X_val, val_labels, X_test, test_labels):
    print("Analyzing the model:")
    print("0 = negative, 1= neutral, 2=positive")

    # Predict on validation data
    val_predictions = model.predict(X_val)
    val_f1 = f1_score(val_labels, val_predictions, average="weighted")
    print(f'Validation F1: {val_f1:.2f}')

    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Validation Accuracy: {val_accuracy:.2f}')

    val_recall = recall_score(val_labels, val_predictions, average="weighted")
    print(f'Validation Recall: {val_recall:.2f}')

    plot_confusion_matrix(model=model, X=X_val, y=val_labels, additional_title="- for val data")

    # Predict on test data
    # 0	negative
    # 1	neutral
    # 2	positive
    test_predictions = model.predict(X_test)
    test_f1 = f1_score(test_labels, test_predictions, average="weighted")
    print(f'Test F1: {test_f1:.2f}')

    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy:.2f}')

    test_recall = recall_score(test_labels, test_predictions, average="weighted")
    print(f'Validation Recall: {test_recall:.2f}')

    plot_confusion_matrix(model=model, X=X_test, y=test_labels, additional_title="- for test data")

    return {"test_accuracy": test_accuracy, "test_f1": test_f1, "test_recall": test_recall,
                "val_accuracy": val_accuracy, "val_f1": val_f1, "val_recall": val_recall}


def write_to_file(estimator_name, vect_name, best_params: dict, analyze_results: dict, params_grid: dict= {}):
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')

    filename =  estimator_name + "_grid_search_" + formatted_time + ".txt"
    with open('models/' + filename, 'w') as file:
        file.write(estimator_name + "\n")
        file.write(vect_name + "\n")
        for param, value in best_params.items():
            file.write(f"{param}: {value}\n")

        file.write("Metrics achieved: \n")
        for metric, value in analyze_results.items():
            file.write(f"{metric}: {value}\n")
        
        file.write("Parameters of grid search: \n")
        for param, param_vals in params_grid.items():
            file.write(f"{param}: {param_vals}\n")

    print("Best parameters written to '{}' at time: {}.".format(filename, formatted_time))


def meta_grid_search(model, parameters:dict, vectorizer_dict: dict, train_data, val_data, test_data, scoring_metric='accuracy'):
    X_train_orig = train_data['tweet']
    X_val_orig = val_data['tweet']
    X_test_orig = test_data['tweet']
    train_labels = train_data['label']
    val_labels = val_data['label']
    test_labels = test_data['label']
    
    for vect_name, vect_args in vectorizer_dict.items(): 
        vectorizer = VectorizerWrapper(vectorizer_name=vect_name)
        
        #fit transform train data
        X_train = vectorizer.fit_transform(X_train_orig, **vect_args)
        # Transform the validation and test data
        X_val = vectorizer.transform(X_val_orig)
        X_test = vectorizer.transform(X_test_orig)
        
        grid_clf = GridSearchCV(model, parameters, verbose= True, scoring=scoring_metric)
        grid_clf.fit(X_train, train_labels)
        print(sorted(grid_clf.cv_results_.keys()))
        
        best_estimator = grid_clf.best_estimator_

        best_params = grid_clf.best_params_
        estimator_name = best_estimator.__class__.__name__
        
        analyze_results = analyze_model(model=best_estimator, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)

        write_to_file(estimator_name=estimator_name, vect_name=vect_name, best_params=best_params, analyze_results=analyze_results, params_grid=parameters)

        return grid_clf


