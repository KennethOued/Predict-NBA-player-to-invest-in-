## Projet 'Is is Worth investing in this NBA player ?'

***
L’objectif est de fournir un classifier permettant de prédire 
qu’un joueur vaut le coup d’investir sur lui car il va durer plus de 5 ans en NBA en s’appuyant sur ses statistiques sportives. Ce modèle vise a conseiller des investisseur cherchant à capitaliser sur de futurs talents de la NBA


## Table of Contents
1. [General Info](#general-info)
2. [Execution](#exécution)
3. [Prétaitement-modèle](#Prétaitement-modèle)
4. [Pistes d'amélioration](#Améliorations)


<a name="general-info"></a>
### General Info
***
Ce livrable est composé de plusieurs fichiers: 

- app.py: code de l'API, construite en utilisant la bibliothèque FastAPI de python

-  streamlit_nba.py: code de la web app construite en utilisant la bibliothèque Streamlit de python

- test.py: code de la construction du classifier. Il regroupe plusieurs fonctions: train (pour l'entrainement de modèle), find_best_model (pour le choix du meilleur modèle parmi une listes de modèles testés, qui me semblaient adaptés au use-case) et score_classifier (pour effectuer une prediction sur notre test set)

- tools.py: code comprenant une liste de fonctions, variables utiles que j'ai utilisé dans les fichiers python mentionnés plus hauts. Ce sont entre-autres: clean_data (pour nettoyer, modifier notre dataset), models (liste de modèles utilisés), model_hyperparameters (hyperparamètres de ces modèles) et preprocessing_pipeline (pipeline de preprocessing)

- requirements.txt: un fichier dans lequel vous trouverez listée, l'ensemble des bibliothèques à installer, utiles à la réalisation de ce projet

- run.bat: l'exécutable. Il permet de lancer le backend contruit au format d’API REST, et le frontend (l'interface utilisateur)


<a name="exécution"></a>
### Exécution
***
Vous pouvez lancer le projet en ouvrant simplement un terminal (CMD, GitBash,...) et éxécuter la commande */run.bat* (J'utilise Windows). Cette commande exécute à la fois le fichier python test.py pour entrainement & choix du modèle, puis lance l'API et la web app streamlit...
L'exécution a duré 3mns sur mon computer.

<a name="Prétaitement-modèle"></a>
### Prétaitement-modèle

- Prétaitement: De mon dataset, je n'ai gardé que la première occurence de chaque élément qui apparait sur plusieurs lignes ; également j'ai supprimé des données que j'ai jugé aberantes, correspondant au cas où le même joueur (Name) présente des stats différentes et/ou output différents d'une ligne du dataset à une autre. 
Nous effectuons une imputation à la moyenne pour les Nan présents dans la colonne 3P% du dataset; nous recalculons par la même occasion les valeurs des colonnes FT%, FG%, 3P%  qui présentaient quelques erreurs

- Modèles: Plusieurs modèles ont été entrainés, fine-tunés à l'aide d'hyperparamètres que nous passons à la méthode GridSearchCV pour la calibration. Le modèle choisi est celui ayant la meilleure performance pour la métrique **precision** dans notre jeu de validation.
Nous avons privilégié cette métrique car l'idée étant de conseiller des investir sur le choix de joueurs sur lesquels investir, le modèle est appréciable à sa capacité à identifier correctement les Vrais Positifs (joueurs sur qui investir).
Nous avons contruit une pipeline incluant cette étape de preprocessing (features scaling et features selectons) et le meilleur modèle


<a name="Améliorations"></a>
### Pistes d'amélioration
***

Ci-après des pistes d'amélioration:

- retourner les données de test sur lesquelles le modèle se trompe et les analyser

- Tester d'autres méthode de feature selections (ACP, LDA,...),
automatiser le choix de k (pour la feature selection SelectKbest)

- tester d'autres modèles

- Améliorer la gestion d'erreurs sur la web app 

