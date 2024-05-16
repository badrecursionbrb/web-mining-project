# web-mining-project

Issue with installing fasttext on Windows (non-unix) may be resolved using: 
pip install fasttext-wheel

For MAC install the certificate in order for the gensim API to work: 
https://stackoverflow.com/questions/42098126/mac-osx-python-ssl-sslerror-ssl-certificate-verify-failed-certificate-verify

## Installation

For installation of certain files refer to the installation.py file: 


import nltk
nltk.download('stopwords')


import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
#ft = fasttext.load_model('cc.en.300.bin')# %%

## Optuna Dashboard

after executing the following command
  optuna-dashboard sqlite:///<sql-lite-db-name>.db


The Optuna Dashboard is reachable by the URL: 
 http://localhost:8080/


