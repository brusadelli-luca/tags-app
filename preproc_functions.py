# Traitement de texte
import numpy as np
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
import bs4
from nltk.stem import WordNetLemmatizer

# Stop words
STOP_W = list(tuple(stopwords.words('english')))

# Tokenizer
tokenizr = RegexpTokenizer(r'[a-zA-Z]+')


def tokenizer_fct(sentence, html=False):
    """sentence (str) => tokens (list)"""

    if html:
        sentence_clean = bs4.BeautifulSoup(sentence, features="html.parser").text.lower()
    else:
        sentence_clean = sentence.lower()

    tokens = tokenizr.tokenize(sentence_clean)

    return tokens


def stop_word_filter_fct(list_words, stop_w=None):
    """tokens (list) => tokens (list)"""
    if stop_w is None:
        stop_w = STOP_W
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if 2 < len(w) < 22]

    return filtered_w2


# Lemmatizer (base d'un mot)
def lemma_fct(list_words):
    """tokens (list) => tokens (list)"""
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]

    return lem_w


# Fonction de préparation du texte pour le bag of words avec lemmatization (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_lem_fct(sentence, html=False):
    """sentence (str) => tokens (list)"""
    tokens = tokenizer_fct(sentence, html)
    sw = stop_word_filter_fct(tokens)
    lem_w = lemma_fct(sw)

    return lem_w


# Fonction de préparation du texte pour le bag of words avec lemmatization (Tf_idf)
# Simplifiée par rapport à la version précédente, permet d'utiliser les paramètres tokenizer, stop_words, lowercase, plus efficaces.
def tfidf_transform_fct(sentence):
    """sentence (str) => tokens (list)"""
    tokens = tokenizr.tokenize(bs4.BeautifulSoup(sentence, features="html.parser").text)
    lem_w = lemma_fct(tokens)

    return lem_w

# On peut également utiliser [gensim.utils.simple_preprocess(text) for text in sentences]


# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(sentence, html=False):
    tokens = tokenizer_fct(sentence, html)

    return ' '.join(tokens)


def feature_tfidf(sentence, tfidf_vect):
    features = tfidf_vect.transform(sentence)
    return features


def feature_w2v(sentence, tokenizer_wi, embed_model):
    maxlen = 19
    features = [tokenizer_wi[w] for w in tokenizer_fct(sentence)] + [0] * maxlen
    features = np.array([np.array(features[:19])])
    embeddings = embed_model.predict(features)

    return embeddings
