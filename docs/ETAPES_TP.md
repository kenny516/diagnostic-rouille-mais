# TP - Diagnostic de la rouille du maïs

Ce document décrit le pipeline complet, depuis les images jusqu'à la comparaison
des modèles de classification. Il sert à la fois de guide d'exécution et de
justification des choix techniques.

## 1. Objectif et structure des classes

Le but est de prédire l'état d'une feuille de maïs :

- `0` : feuille saine ;
- `1` : feuille malade, présentant des symptômes de rouille.

Le jeu local contient 800 images équilibrées : 400 saines et 400 malades. Les
images malades proviennent de la classe *Common Rust* de PlantVillage. Elles
constituent un proxy visuel de la rouille Polysora, mais ne prouvent pas que le
pathogène est précisément *Puccinia polysora*.

## 2. Installation et reproductibilité

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les expériences utilisent `random_state = 42`. Le découpage apprentissage/test
est stratifié afin de conserver la même proportion de feuilles saines et malades.

## 3. Audit et prétraitement des images

Commande :

```powershell
python scripts/audit_dataset.py
```

L'audit vérifie notamment :

- la lisibilité et les dimensions des images ;
- les doublons exacts avec une empreinte SHA-256 ;
- la luminosité et la netteté ;
- la proportion de fond sombre ;
- la surface retenue par le masque de feuille.

Résultat : 800 images valides, aucun doublon et 9 images signalées comme très
floues. Elles sont conservées pour ne pas modifier silencieusement le jeu de
données ; leur statut se trouve dans `outputs/tables/image_audit.csv`.

### 3.1 Pourquoi isoler la feuille ?

Les images malades possèdent en moyenne environ 31 % de fond noir, contre presque
0 % pour les images saines. Sans correction, un modèle pourrait reconnaître le
fond plutôt que la maladie.

La fonction `create_leaf_mask()` applique les opérations suivantes :

1. sélection des pixels non noirs ;
2. fermeture morphologique pour combler les petits trous ;
3. ouverture morphologique pour éliminer le bruit ;
4. conservation du plus grand contour, supposé être la feuille.

Tous les descripteurs sont ensuite calculés uniquement dans ce masque. Pour les
images saines cadrées entièrement sur la feuille, le masque couvre naturellement
presque toute l'image.

## 4. Partie 1 - Extraction des caractéristiques

Commande :

```powershell
python scripts/extract_features.py
```

Le fichier obtenu est `outputs/tables/features.csv` et contient :

```text
ID_Image | pct_rouille | rugosite | heterogeneite_saturation | label_malade
```

### 4.1 Pourcentage de rouille `pct_rouille`

L'image BGR est convertie en HSV avec OpenCV. Les tons brun-orange sont isolés
avec les bornes OpenCV suivantes :

```text
HSV minimum = [3, 55, 25]
HSV maximum = [30, 255, 255]
```

La caractéristique vaut :

```text
pct_rouille = 100 × pixels_rouille_dans_feuille / pixels_de_feuille
```

Le dénominateur exclut donc le fond noir.

### 4.2 Rugosité `rugosite`

Le calcul suit ces étapes :

1. conversion en niveaux de gris ;
2. léger flou gaussien 3 × 3 pour réduire le bruit isolé ;
3. gradients horizontaux et verticaux avec Sobel ;
4. magnitude `sqrt(Gx² + Gy²)` ;
5. moyenne des magnitudes à l'intérieur de la feuille.

Le masque est légèrement érodé avant la moyenne. Cette précaution empêche le bord
feuille/fond de devenir artificiellement une zone très rugueuse.

### 4.3 Variable personnelle `heterogeneite_saturation`

Cette variable est l'écart-type de la saturation HSV, normalisée entre 0 et 1,
dans la feuille. Une feuille saine tend à être plus homogène. Les pustules, zones
brunes et halos chlorotiques créent au contraire un mélange plus variable.

Médianes observées :

| Classe | `pct_rouille` | `rugosite` | `heterogeneite_saturation` |
|---|---:|---:|---:|
| Saine | 0,0000 | 31,6727 | 0,1105 |
| Malade | 16,4655 | 46,2655 | 0,1472 |

## 5. Partie 2 - Indice Max-Minority

Pour un nœud contenant `N` observations, la pureté est la proportion de la classe
majoritaire :

```text
P(t) = max(n0 / N, n1 / N)
```

Pour un seuil qui crée les groupes gauche `G` et droite `D` :

```text
P_split = |G|/N × P(G) + |D|/N × P(D)
```

La fonction `trouver_meilleur_split(X_column, y)` :

1. trie la variable et les labels associés ;
2. place les seuils au milieu de deux valeurs consécutives distinctes ;
3. calcule les effectifs cumulés des classes ;
4. évalue chaque pureté pondérée ;
5. retourne le seuil qui maximise cette pureté.

Le tri coûte `O(N log N)` et le balayage des seuils coûte `O(N)`. En cas d'égalité,
le plus petit seuil est choisi pour rendre le résultat reproductible.

Commande :

```powershell
python scripts/evaluate_max_minority_splits.py
```

Sur le jeu complet, `pct_rouille` fournit le meilleur premier split : seuil
`0,6269 %` et pureté pondérée `0,97375`.

## 6. Partie 3 - Modèles de classification

Commande :

```powershell
python scripts/train_and_compare_models.py
```

### 6.1 Découpage des données

- apprentissage : 640 images, soit 80 % ;
- test : 160 images, soit 20 % ;
- stratification : 80 saines et 80 malades dans le test ;
- graine aléatoire : 42 ;
- les trois caractéristiques sont calculées avant le découpage, sans utiliser les
  labels ni apprendre de paramètres statistiques globaux.

L'affectation exacte de chaque image est conservée dans
`outputs/tables/data_split.csv`.

### 6.2 Arbre Max-Minority maison

À chaque nœud, l'arbre :

1. calcule la classe majoritaire et la pureté courante ;
2. cherche le meilleur seuil de chaque variable disponible ;
3. choisit le couple variable/seuil ayant la meilleure pureté pondérée ;
4. envoie `X <= seuil` à gauche et `X > seuil` à droite ;
5. recommence récursivement.

La récursion s'arrête si le nœud est pur, si la profondeur maximale 5 est atteinte,
si le nœud contient moins de deux observations ou si aucun split n'améliore la
pureté. L'importance d'une variable cumule son gain de pureté pondéré par le
nombre d'observations du nœud.

### 6.3 Forêt Max-Minority maison

La forêt contient 100 arbres. Pour chaque arbre :

1. un bootstrap de 640 lignes est tiré avec remplacement grâce à
   `np.random.Generator.choice` ;
2. à chaque nœud, `sqrt(3) = 1` variable est tirée aléatoirement ;
3. un arbre Max-Minority de profondeur maximale 5 est construit ;
4. la prédiction finale est obtenue par vote majoritaire.

Le bagging diminue la variance : une erreur ou un seuil instable dans un arbre
unique a moins d'influence après agrégation de nombreux arbres diversifiés.

### 6.4 Modèles scikit-learn

La comparaison utilise :

- `DecisionTreeClassifier(criterion="gini", max_depth=5)` ;
- `RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5)`.

La profondeur est identique aux modèles maison afin que la comparaison porte
principalement sur le critère et la méthode d'ensemble.

### 6.5 Métriques

La classe positive est la feuille malade (`1`).

```text
Accuracy  = prédictions correctes / toutes les prédictions
Précision = vrais positifs / (vrais positifs + faux positifs)
Rappel    = vrais positifs / (vrais positifs + faux négatifs)
```

Le rappel est particulièrement important : un faux négatif laisse une feuille
malade non détectée et peut favoriser la propagation de la maladie.

### 6.6 Résultats sur le test

| Modèle | Accuracy | Précision | Rappel | Faux positifs | Faux négatifs |
|---|---:|---:|---:|---:|---:|
| Arbre maison Max-Minority | 0,9938 | 0,9877 | 1,0000 | 1 | 0 |
| Forêt maison Max-Minority | 0,9875 | 0,9875 | 0,9875 | 1 | 1 |
| Arbre scikit-learn Gini | 0,9750 | 0,9634 | 0,9875 | 3 | 1 |
| Forêt scikit-learn Gini | 0,9875 | 0,9875 | 0,9875 | 1 | 1 |

Sur ce découpage précis, l'arbre maison obtient le meilleur résultat et ne manque
aucune feuille malade. Les deux forêts donnent le même résultat. Cela ne signifie
pas qu'un arbre unique est généralement plus robuste : la différence ne porte ici
que sur une image et le jeu de données est simple et contrôlé.

### 6.7 Recommandation agronomique

Pour la démonstration du TP, le modèle recommandé est l'arbre Max-Minority maison,
car son rappel vaut 1,00 et sa matrice de confusion ne contient aucun faux négatif.
Le fichier correspondant est `models/arbre_max_minority.joblib`.

Cette recommandation est uniquement expérimentale. Un déploiement réel à
Madagascar exige au minimum des photos de terrain locales, des cas confirmés de
*Puccinia polysora*, plusieurs téléphones et conditions lumineuses, ainsi qu'une
validation externe. En l'état, l'application doit être présentée comme une aide au
diagnostic et non comme un diagnostic agronomique certain.

## 7. Fichiers produits

### Tableaux

- `outputs/tables/image_audit.csv` : audit par image ;
- `outputs/tables/features.csv` : caractéristiques finales ;
- `outputs/tables/max_minority_splits.csv` : meilleurs premiers seuils ;
- `outputs/tables/model_comparison.csv` : métriques des quatre modèles ;
- `outputs/tables/test_predictions.csv` : prédictions image par image ;
- `outputs/tables/feature_importances.csv` : importance des variables ;
- `outputs/tables/data_split.csv` : affectation apprentissage/test.
- `outputs/tables/project_verification.json` : preuve de cohérence globale des livrables.

### Figures

- `outputs/figures/leaf_mask_examples.png` ;
- `outputs/figures/image_quality_distributions.png` ;
- `outputs/figures/model_performance.png` ;
- `outputs/figures/confusion_matrices.png` ;
- `outputs/figures/feature_importances.png`.

### Modèles

- `models/arbre_max_minority.joblib` ;
- `models/foret_max_minority.joblib` ;
- `models/arbre_scikit_gini.joblib` ;
- `models/foret_scikit_gini.joblib` ;
- `models/modele_deploiement.pkl` : modèle et métadonnées chargés par Streamlit ;
- `models/metadata.json` : ordre des variables et paramètres reproductibles.

## 8. Partie 4 - Application Streamlit

Lancer l'application depuis la racine du projet :

```powershell
streamlit run app.py
```

Le traitement d'une image téléversée suit exactement la même chaîne que
l'entraînement :

1. lecture des octets PNG/JPEG avec `cv2.imdecode` ;
2. construction du masque de feuille ;
3. calcul de `pct_rouille`, `rugosite` et `heterogeneite_saturation` ;
4. création d'une ligne Pandas dans l'ordre mémorisé pendant l'entraînement ;
5. prédiction avec l'arbre Max-Minority chargé par `pickle.load` ;
6. affichage du diagnostic et des trois valeurs calculées.

Le modèle est mis en cache avec `st.cache_resource`, ce qui évite de le recharger
à chaque interaction. L'application rappelle explicitement qu'il s'agit d'un
prototype pédagogique.

### 8.1 Galerie d'historique

Après une analyse :

- l'image est sauvegardée dans `uploads/` ;
- le diagnostic, les caractéristiques et la date UTC sont ajoutés à
  `uploads/history.json` ;
- les résultats sont présentés avec `st.columns(3)` ;
- l'empreinte SHA-256 empêche une même image d'être ajoutée plusieurs fois lors
  des réexécutions automatiques de Streamlit.

Les fichiers du dossier `uploads/` sont locaux et ignorés par Git afin de ne pas
publier les photos des utilisateurs.

## 9. Refaire toute l'expérience

```powershell
python scripts/audit_dataset.py
python scripts/extract_features.py
python scripts/evaluate_max_minority_splits.py
python scripts/train_and_compare_models.py
python -m unittest discover -s tests -v
python scripts/verify_project.py
streamlit run app.py
```

Les quatre parties demandées dans l'énoncé sont alors reproductibles depuis les
images brutes jusqu'à l'application web locale.
