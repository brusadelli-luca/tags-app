"""
GET : Sends data in simple or unencrypted form to the server.
HEAD : Sends data in simple or unencrypted form to the server without body. 
PUT : Replaces target resource with the updated content.
DELETE : Deletes target resource provided as URL.
"""
import pickle
from flask import Flask, jsonify, request

from preproc_functions import tfidf_transform_fct, feature_tfidf, transform_bow_lem_fct, feature_w2v

MODEL_SUFFIX = 'tfidf'


def load_stuff(model_suffix):
    """fonction qui charge les objets nécessaires pour la prédiction des tags
    """

    if model_suffix == 'tfidf':
        # Vectorizer
        tfidf_vect = pickle.load(open('tfidf_vect_fit_title.pkl', 'rb'))

        return tfidf_vect

    elif model_suffix == 'word2vec':
        # W2V Tokenizer
        tokenizer_wi = pickle.load(open('tokenizer_word_index.pkl', 'rb'))

        # Chargement du modèle Word2Vec
        w2v_model = pickle.load(open('w2v_model_train_title.pkl', 'rb'))

        embed_model = pickle.load(open('embed_model_title.pkl', 'rb'))

        return tokenizer_wi, w2v_model, embed_model


TFIDF_VECT = load_stuff(MODEL_SUFFIX)


def model_load(name_suff):
    """fonction qui charge le modèle entrainé, l'explainer, le dataset sur lequel va porter l'api et le détail des features
    utilisés par le modèle"""

    name_suff = name_suff + '_sup'

    # Read dictionary pkl file
    with open('ovr_sgdc_{}.pkl'.format(name_suff), 'rb') as fp:
        return pickle.load(fp)


MODEL = model_load(MODEL_SUFFIX)


def pred_fct(model, sentence):
    """Predict the tags of a sentence
    """
    feature_sentence = None
    tags_feat = None

    if MODEL_SUFFIX == 'tfidf':
        prep_sentence = tfidf_transform_fct(sentence)

        feature_sentence = feature_tfidf(prep_sentence, TFIDF_VECT)

    elif MODEL_SUFFIX == 'word2vec':
        prep_sentence = transform_bow_lem_fct(sentence)

        feature_sentence = feature_w2v(prep_sentence, tokenizer_wi, embed_model)

    if feature_sentence is not None:

        tags_feat = model.predict(feature_sentence)

    mlb = pickle.load(open('mlb_binarizer.pkl', 'rb'))

    tags_predict = mlb.inverse_transform(tags_feat)[0]

    return tags_predict


app = Flask(__name__)

@app.route('/hello')
def hello_world():
    return 'HELLO WORLD ! Bienvenue sur l\'API du Projet 4 !'


@app.route('/')
def bienvenue():
    return 'Bienvenue sur l\'API du Projet 4 !'


@app.route('/predict_tags', methods=['POST', 'GET'])
def predict_tags():
    global MODEL
    if request.method == 'POST':
        sentence = request.args.get('sentence')
        return jsonify({'response' : pred_fct(MODEL, sentence)})

    else:
        sentence = request.args.get('sentence')
        return jsonify({'response' : pred_fct(MODEL, sentence)})


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
