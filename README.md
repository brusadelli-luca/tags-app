# API de Production pour la génération de tags
Ce dépôt contient le code source de l'API de Production pour la génération de tags.

Le code principal est dans le fichier `tags_app.py`.

Les fonctions de prétraitement sont dans le fichier `preproc_functions.py`.

Le fichier `requirements.txt` contient la liste des dépendances Python nécessaires pour exécuter l'API.

A titre expérimental, les fichiers du modèle sérialisé sont fournis dans le dépôt :
- `tfidf_vect_fit_title.pkl`
- `ovr_sgdc_tfidf_sup.pkl`'
- `mlb_binarizer.pkl`
- `sw.pkl`

 
Les fichiers suivants sont utilisés pour le déploiement sur Heroku :
- `Procfile`
- `.python-version`


## Installation
Pour installer les dépendances, exécutez la commande suivante:
```bash
pip install -r requirements.txt
```

## Utilisation
L'API est accessible à l'adresse `https://tags-app-0a80a95df38d.herokuapp.com/`
```

Exemple de requête POST:
```https://tags-app-0a80a95df38d.herokuapp.com/predict_tags?sentence='How do I learn Python ?'```

Exemple de réponse:
```json
{
  "tags": [
    "python"
  ]
}
```

Un dashboard est disponible à l'adresse `https://tags-dashboard-3447887db869.herokuapp.com/`