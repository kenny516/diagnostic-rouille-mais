# Jeu de donnees du TP

## Structure

- `malades/` : 400 images de feuilles de mais atteintes de rouille.
- `saines/` : 400 images de feuilles de mais saines.
- Toutes les images sont au format JPEG et ont une resolution de 256 x 256 pixels.

## Sources

Le jeu Kaggle retenu est **Corn or Maize Leaf Disease Dataset** :
https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

Pour eviter de telecharger les classes inutiles, les images ont ete recuperees directement
depuis le depot PlantVillage utilise par ce jeu Kaggle :
https://github.com/spMohanty/PlantVillage-Dataset

Un sous-ensemble deterministe et equilibre de 400 images par classe a ete selectionne parmi :

- `Corn_(maize)___Common_rust_` : 1 192 images disponibles ;
- `Corn_(maize)___healthy` : 1 162 images disponibles.

Reference principale : Mohanty, Hughes et Salathe (2016), "Using Deep Learning for
Image-Based Plant Disease Detection", Frontiers in Plant Science.

## Etiquettes

- `saines` -> `label_malade = 0`
- `malades` -> `label_malade = 1`

## Limites importantes

1. La classe malade de Kaggle/PlantVillage correspond a la **rouille commune du mais**
   (`Puccinia sorghi`) et non a une identification garantie de la rouille Polysora
   (`Puccinia polysora`). Elle sera utilisee comme proxy visuel pour ce TP, et cette limite
   devra etre mentionnee dans le rapport.
2. Les images malades contiennent souvent un fond noir, contrairement aux images saines.
   Pour eviter que le modele apprenne ce fond, les caracteristiques HSV et Sobel devront etre
   calculees uniquement sur les pixels appartenant a la feuille.
3. Les donnees PlantVillage sont prises dans des conditions controlees. Une evaluation sur
   quelques photos de terrain sera necessaire avant de presenter l'application comme un outil
   deployable a Madagascar.
