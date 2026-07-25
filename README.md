# TP - Diagnostic de la rouille du mais

Projet de classification binaire de feuilles de mais :

- `0` : feuille saine ;
- `1` : feuille atteinte de rouille.

## Organisation

- `dataset/` : images classees dans `saines/` et `malades/` ;
- `src/` : extraction des caracteristiques et modeles ;
- `models/` : modeles entraines ;
- `outputs/figures/` : graphiques et matrices de confusion ;
- `outputs/tables/` : DataFrame de caracteristiques et resultats ;
- `uploads/` : historique local de l'application Streamlit ;
- `tests/` : tests des fonctions importantes.

## Installation sous Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execution

Audit des images et verification du masque de feuille :

```powershell
python scripts/audit_dataset.py
```

Extraction des trois caracteristiques de la partie 1 :

```powershell
python scripts/extract_features.py
```

Le tableau demande par l'enonce est genere dans `outputs/tables/features.csv`.
`pct_rouille` est exprime en pourcentage (de 0 a 100). La variable personnelle
`heterogeneite_saturation` mesure la variation de saturation HSV dans la feuille.

Recherche des meilleurs seuils avec l'indice Max-Minority de la partie 2 :

```powershell
python scripts/evaluate_max_minority_splits.py
```

Les seuils et puretes obtenus sont enregistres dans
`outputs/tables/max_minority_splits.csv`.

Entrainement et comparaison des quatre modeles de la partie 3 :

```powershell
python scripts/train_and_compare_models.py
```

Les modeles sont sauvegardes dans `models/`, les metriques dans
`outputs/tables/model_comparison.csv` et les graphiques dans `outputs/figures/`.
La documentation detaillee du traitement et de l'analyse se trouve dans
[`docs/ETAPES_TP.md`](docs/ETAPES_TP.md).

Lancement de l'application Streamlit de la partie 4 :

```powershell
streamlit run app.py
```

L'application accepte les images PNG/JPG/JPEG, calcule les trois caracteristiques,
affiche le diagnostic et conserve une galerie locale dans `uploads/`.

Tests automatiques :

```powershell
python -m unittest discover -s tests -v
```

Verification globale de tous les livrables generes :

```powershell
python scripts/verify_project.py
```

Le detail de la provenance et des limites des images se trouve dans `dataset/README.md`.
