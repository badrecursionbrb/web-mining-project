# in order to download the stopwords list

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

# convert the glove vectors from nltk to gensim word2vec form: 
# 1. move to target directory 
# 2. execute the following command
# python -m gensim.scripts.glove2word2vec -i glove.twitter.27B.25d.txt -o glove-twitter-27B-25d-w2v.txt.