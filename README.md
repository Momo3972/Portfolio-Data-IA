# Mohamed Lamine OULD BOUYA  
**Data Scientist - Data Engineer - Data Analyst - IA**

> Je conçois des solutions data robustes, explicables et utiles : de l’ingestion à la mise en production, en passant par la modélisation ML/IA et la data visualisation.

Rueil-Malmaison  
ouldbouya.mohamedlamine@gmail.com  
+33 7 60 15 54 08  
Objectif : **stage de fin d’études (6 mois)** dès que possible pour un début au plus tard le 10 mars 2026  
Intérêts : Machine Learning - Data visulisation - IA générative - Qualité des données - Deep learning - Cloud - ETL

---

## À propos de moi

Actuellement en **Mastère Spécialisé Expert Big Data Engineer à l'Université de Technologie de Troyes (UTT)**, je combine un parcours **ingénieur** (analyse de risques, reporting automatisé, data qualité) et une solide formation en **science des données**.  
Je m’intéresse particulièrement à la création de pipelines de données robustes, à l'analyse et la visualisation via API, à l'IA générative, à la modélisation et la mise en production de modèles IA.

---

## Projets phares

---

### 1. [Classification d’Images CIFAR-10 (MLP, CNN et Transfer Learning EfficientNetB0)](https://github.com/Momo3972/deepvision-cifar10-classifier)

Développement d’un système complet de classification d’images CIFAR-10 incluant un CNN construit from scratch et un modèle EfficientNetB0 utilisant le Transfer Learning, la Data Augmentation et un Fine-Tuning avancé. Création d’un pipeline reproductible + application Streamlit de démonstration.

### **Stack**
Python, TensorFlow/Keras, NumPy, Scikit-learn, Matplotlib, Google Colab, Streamlit

### **Objectif**
Comparer deux approches d’apprentissage profond pour mesurer l’impact du Transfer Learning :

- CNN baseline entraîné from scratch 
- EfficientNetB0 pré-entraîné (ImageNet) et Fine-Tuning  

Objectif : démontrer les gains en performance et en généralisation.

### **Méthodes utilisées**
- Préparation des données : normalisation, split stratifié, Data Augmentation 
- Entraînement ML : CNN custom (Conv2D → MaxPool → Dropout → BatchNorm)  
- Transfer Learning : EfficientNetB0, fine-tuning progressif, callbacks (`EarlyStopping`, `ReduceLROnPlateau`)  
- Analyse complète : courbes d’apprentissage, matrices de confusion, F1-scores par classe  
- Validation robuste sur données jamais vues  

### **Principaux résultats**
- **CNN baseline** : accuracy ≈ **0.70**  
- **EfficientNetB0** : accuracy ≈ **0.93**, loss ≈ **0.16**, **gain +0.23** en accuracy  
- Forte amélioration sur les classes complexes grâce au fine-tuning + Data Augmentation  

### **Livrables**
- Modèle final : `best_model_efficientnet_aug.h5`  
- Notebook complet : `Projet_Vision_CIFAR10.ipynb`  
- Application Streamlit : `app.py`  
- Courbes d’apprentissage + matrices de confusion  
- Environnement reproductible (`requirements.txt`) + README documenté  

---

### 2. [Credit Default MLOps Pipeline](https://github.com/Momo3972/credit-default-mlops-pipeline)

Développement d’un pipeline MLOps end-to-end production-ready pour la prédiction du défaut de paiement, intégrant l’entraînement du modèle, le versioning MLflow, le déploiement API et le monitoring temps réel.

- **Stack** : Python, FastAPI, Scikit-learn, MLflow, Docker, Docker Compose, MinIO (S3), PostgreSQL, Prometheus, Grafana, Linux, WSL2 (Ubuntu), Machine virtuelle
- **Objectif** : Pipeline MLOps de bout en bout pour la prédiction du défaut de paiement : entraînement, tracking MLflow, API FastAPI, services Dockerisés, CI/CD et monitoring en production
- **Méthodes utilisées** :
  - Entraînement et évaluation du modèle de scoring crédit
  - Tracking des expériences et versioning via MLflow Tracking ET Model Registry
  - Promotion du modèle en Production via alias MLflow
  - Déploiement du modèle via API FastAPI containerisée
  - Exposition des métriques applicatives avec Prometheus
  - Visualisation et observabilité via dashboards Grafana auto-provisionnés
  - Orchestration complète de l’infrastructure avec Docker Compose
  - Pipeline 100 % reproductible (infra, ML, serving, monitoring)
- **Livrables** :
  - API FastAPI de scoring crédit (/predict, /health, /meta)
  - Modèle versionné et traçable dans MLflow
  - Stack Docker complète : MLflow, MinIO, PostgreSQL, API, Prometheus, Grafana
  - Dashboards Grafana prêts à l’emploi
  - Documentation technique : architecture, monitoring, runbook et checklist de démo reproductible

---

### 3. [Everflow API Analytics](https://github.com/Momo3972/Everflow-API-Analytics)

Développement d’un mini-système analytique pour visualiser la performance marketing via l’API Everflow, incluant l’extraction de données, le calcul de métriques clés (profit), et la génération automatique de graphiques et d’un rapport Markdown.

- **Stack** : Python, Pandas, Matplotlib, Requests, Everflow API

- **Méthodes utilisées** :
  - Connexion sécurisée à l’API Everflow (authentification via clé API)
  - Extraction et transformation de statistiques agrégées (offres, affiliés, annonceurs)
  - Calcul du profit (revenue - payout)
  - Génération automatique de graphiques analytiques
  - Export automatique d’un rapport Markdown
  - Structuration modulaire : `src/`, `mock_data/`, `out/`

- **Livrables** :
  - Notebook complet : *Everflow-API-Analytics.ipynb*
  - Rapport Markdown généré automatiquement
  - Fichiers de sortie dans `out/` :
    - profits par offre
    - profits par affilié
    - profits par annonceur
    - rapport global (*REPORT*)

- **Résultat** :
  Déploiement d’un mini-dashboard analytique automatisé permettant une visualisation rapide et exploitable des performances marketing via l’API Everflow.

---

### 4. [NOAA Weather - Industrial End-to-End MLOps Pipeline](https://github.com/Momo3972/noaa-weather-mlops-pipeline)

Conception et déploiement d’un pipeline MLOps industriel end-to-end pour la prévision de températures à partir de données NOAA, couvrant l’ingestion automatisée, l’entraînement supervisé, la gouvernance des modèles, le déploiement API et le monitoring de la dérive des données.

- **Stack** : Python, Scikit-learn, FastAPI, MLflow (Tracking et Model Registry), Apache Airflow, EvidentlyAI, Docker, Docker Compose, GitHub Actions, Linux, WSL2
- **Objectif** : Mettre en place une infrastructure MLOps complète et automatisée, reproduisant un workflow industriel réel : orchestration des pipelines ML, gestion du cycle de vie des modèles, déploiement en production et observabilité continue des données
- **Méthodes utilisées** :
  - Ingestion et préparation automatisées des données météorologiques NOAA
  - Feature engineering et entraînement d’un modèle de régression Random Forest
  - Tracking des expériences, métriques et artefacts via MLflow
  - Gouvernance des modèles avec MLflow Model Registry
  - Promotion automatique du modèle en Production via alias MLflow
  - Déploiement du modèle via une API FastAPI containerisée
  - Orchestration des workflows de réentraînement avec Apache Airflow
  - Détection et analyse de la dérive des données avec EvidentlyAI
  - CI/CD automatisé (tests, linting, build et déploiement Docker)
  - Stack entièrement Dockerisée et reproductible
- **Livrables** :
  - API FastAPI de prédiction des températures (`/predict`, `/health`)
  - Modèles versionnés, traçables et promus dans MLflow
  - DAG Airflow de réentraînement planifié
  - Rapports de data drift générés automatiquement
  - Stack multi-conteneurs opérationnelle (MLflow, Airflow, API, monitoring)
  - Documentation complète et preuves d’exécution (CI/CD, orchestration, monitoring)

---

### 5. [Chatbot RAG IA Générative](https://github.com/Momo3972/chatbot-rag-ia-gen)

Développement d’un chatbot IA utilisant une architecture RAG et une interface Web, permettant d’interroger dynamiquement une base documentaire PDF et d’obtenir des réponses contextualisées

- **Stack** : Python, LLM (LangChain / OpenAI), RAG, Gradio / Streamlit  

- **Objectif** : permettre à un utilisateur d’interroger des documents (PDF, textes) et d’obtenir des réponses précises, contextualisées et sourcées  

- **Méthodes utilisées** :  
  - Conception d’une chaîne RAG complète : *indexation, embeddings, retrieval, génération*  
  - Intégration d’une API IA (OpenAI / autres LLMs)  
  - Création d’une interface Web interactive (Gradio / Streamlit)  
  - Tests de pertinence et ajustements de la chaîne (chunking, embeddings, scoring de similarité)  

- **Livrables** :  
  - Application Web interactive prête à l’emploi  
  - Chaîne RAG complète (*embedding -> retrieval -> génération*)  
  - Évaluation de la pertinence des réponses et amélioration de la qualité du chatbot   

---

### 6. [Détection de fraude bancaire](https://github.com/Momo3972/projet-fraude)

Analyse et modélisation de transactions bancaires pour identifier des signaux faibles de fraude dans un contexte de données fortement déséquilibrées

- **Stack** : Python, Pandas, NumPy, Scikit-learn, XGBoost  
- **Objectif** : améliorer le rappel de la classe frauduleuse sans dégrader la précision, dans un dataset où les fraudes représentent <1 % des transactions  
- **Méthodes utilisées** :  
  - Analyse exploratoire (EDA) des variables financières et temporelles  
  - Préparation des données : nettoyage, encodage, feature engineering  
  - Gestion du déséquilibre via SMOTE  
  - Entraînement et optimisation de modèles : Régression Logistique, Random Forest, XGBoost 
  - Évaluation avancée : F1-score, AUC-ROC, courbes précision-rappel, matrice de confusion  
  - Sélection du meilleur modèle basé sur sa capacité à détecter les fraudes rares  
- **Résultat** : amélioration du F1-score et meilleure détection des transactions frauduleuses minoritaires  
- **Livrables** :  
  - Notebook complet `fraude_detection.ipynb`  
  - Visualisations : matrices de confusion, ROC/PR curves, importances des features (`reports/figures/`)  
  - Fichier de métriques JSON (`reports/metrics/metrics.json`)  
  - Jeux de données nettoyés (`train_split.csv`, `test_split.csv`)  
  - README structuré documentant toute la démarche   

---

### 7. [Dashboard Power BI - Analyse de la performance commerciale](https://github.com/Momo3972/powerbi-global-superstore-dashboard)

Création d’un tableau de bord interactif pour analyser les ventes, profits et performances commerciales du dataset Global Superstore

- **Stack** : Power BI Desktop, Power Query, DAX, Excel  
- **Objectif** : fournir un tableau de bord professionnel permettant :
  - d’analyser l’évolution du chiffre d’affaires,
  - d’identifier les pays contributeurs,
  - de visualiser la répartition des ventes par catégories de produits,
  - et de suivre les KPIs essentiels (ventes, profits, volume, marges).

- **Méthodes utilisées** :
  - Analyse des besoins métier et identification des indicateurs clés (KPI)
  - Nettoyage, transformation et modélisation des données via Power Query
  - Modélisation en étoile (tables de faits et dimensions)
  - Création de mesures DAX : Total Ventes, Total Profit, Quantité vendue, Marge
  - Visualisations avancées :
    - Graphique temporel des ventes (année / mois)
    - Top 10 des pays par chiffre d’affaires
    - Répartition des ventes par catégorie de produits
  - Filtres dynamiques : année, segment client, catégorie produit, pays
  - Page d’infobulle (tooltip) personnalisée pour contextualiser les ventes
  - Page “À propos” documentant la démarche analytique

- **Livrables** :
  - Tableau de bord Power BI complet : analyse des performances commerciales
  - Visualisations interactives + filtres dynamiques + infobulle contextualisée
  - Documentation claire (README + page dédiée dans Power BI)

---

### 8. [Prédiction de la présence d'un réservoir pétrolier](https://github.com/Momo3972/oil-reservoir-prediction-ml)

Prédiction de la présence d’un réservoir pétrolier à partir de données géologiques et sismiques simulées - avec analyse d’interprétabilité SHAP pour valider la cohérence géologique

- **Stack** : Python, Pandas, NumPy, Scikit-learn, Random Forest, XGBoost, Matplotlib, SHAP
- **Objectif** : prédire la présence d’hydrocarbures avant forage, en exploitant des caractéristiques géologiques (porosité, type de roche, piège, profondeur, distance aux champs existants, signature sismique)
- **Méthodes utilisées** :
  - Analyse exploratoire (EDA) géologique
  - Préparation des données & encodage des variables catégorielles
  - Entraînement et optimisation d’un modèle (GridSearchCV)
  - Comparaison de trois modèles : Logistic Regression, Random Forest optimisé, XGBoost optimisé
  - Évaluation approfondie (Accuracy, Recall, Precision, F1, ROC-AUC)
  - Interprétabilité avancée avec SHAP :
    - Summary Plot (vue globale des variables)
    - Force Plot (explication locale d’une observation)
    - Bar Plot (importance moyenne des features)
- **Résultat** :
  - Le **Random Forest optimisé** obtient le meilleur score (**ROC-AUC ≈ 0.87**)
  - Les variables les plus déterminantes sont :
    - **Seismic_Score** (signal sismique fort -> structures favorables)
    - **Rock_Type** (grès / calcaire -> bons réservoirs)
    - **Trap_Type** (anticline / faille / dôme -> accumulation d’hydrocarbures)
    - **Porosity** et **Permeability** (qualité du réservoir)
    - **Distance aux champs existants**
  - L’analyse SHAP confirme que le modèle prend des décisions **géologiquement cohérentes**
- **Livrables** :
  - Notebook complet d'analyse et modélisation (`oil-prediction.ipynb`)
  - Modèle optimisé exporté : `best_random_forest_oil_reservoir.joblib`
  - Visualisations : matrice de confusion, ROC curve, summary SHAP, force plot, barplot SHAP
  - README complet documentant la démarche scientifique et géologique

---

### 9. [Analyse de l’Espérance de Vie (2000–2015)](https://github.com/Momo3972/analyse-esperance-de-vie)

Analyse statistique complète des déterminants de l’espérance de vie mondiale (OMS), incluant une pipeline reproductible (Makefile), plusieurs modèles prédictifs et un rapport automatisé

- **Stack** : R, tidyverse, ggplot2, glmnet, randomForest, corrplot, RMarkdown, Makefile  
- **Objectif** : comprendre les facteurs influençant l’espérance de vie et comparer plusieurs modèles prédictifs (Régression linéaire, LASSO, Stepwise AIC, Random Forest)

- **Méthodes utilisées** :  
  - Nettoyage avancé des données (naniar, imputation médiane)  
  - Analyse exploratoire : corrélogramme, histogrammes, scatterplots  
  - Modélisation prédictive : LM, LASSO, Stepwise, Random Forest  
  - Évaluation des modèles (RMSE, R²) et interprétation des variables importantes  
  - Automatisation du workflow via Makefile → génération d’un rapport Word professionnel

- **Principaux résultats** :  
  - Random Forest = meilleur modèle (**R² ≈ 0.96**, RMSE Test ≈ **1.9**)  
  - Variables déterminantes : `hiv_aids`, `adult_mortality`, `income_composition`, `schooling`  
  - Forte influence des facteurs socio-économiques et sanitaires

- **Livrables** :  
  - Dataset propre + modèles sauvegardés  
  - Figures (corrélogramme, importance des variables, comparaisons de modèles)  
  - Rapport complet généré automatiquement : `rapport_final.docx`  
  - Pipeline reproductible (Makefile et scripts)

---

## Compétences techniques

| Domaine | Compétences |
|----------|-------------|
| **Langages** | Python, SQL, R, Excel |
| **Machine Learning** | Scikit-learn, XGBoost, PCA, SMOTE |
| **Deep Learning / Computer vision** | MLP, CNN, Transfer Learning, EfficientNet, Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Google Colab |
| **Visualisation** | Power BI, Tableau, Plotly, Matplotlib, Seaborn |
| **Base de données** | MySQL, MongoDB |
| **Cloud / Big Data** | Google Cloud Platform (GCP), Snowflake, Databricks |
| **Data Engineering** | ETL, Pipelines, FastAPI, MLflow (Tracking et Model Registry), Apache Airflow, EvidentlyAI, GitHub Actions, Docker, Docker Compose, MinIO (S3), PostgreSQL, Prometheus, Grafana, Linux, WSL2 (Ubuntu), Machine virtuelle, ingestion multi-source, EDA, data quality |
| **Outils / Méthodo** | Git, VS Code, Jupyter, tests unitaires, documentation |

---

## Formation & Certifications

- **Mastère Spécialisé - Expert Big Data Engineer**, UTT Paris (2024–2025)  
- **Certificat Concepteur Développeur en Data Science**, Jedha Paris (2024)  
- **Lean Six Sigma Black Belt**, Cubic Partners Paris (2019)  
- **Master QSE**, EISTI Cergy (2017)  
- **Master Géosciences**, Université Paris-Saclay (2014)

---

## Expériences professionnelles

### 🔹 **AERGON - Ingénieur d’études** (2019 - aujourd’hui)
- Réalisation d’études en sécurité et sûreté nucléaire
- Audit technique et réglementaire en environnement industriel
- Analyse de risques et automatisation de reporting
- Manipulation de jeux de données réglementaires
**Environnement technologique** : Word, Excel, VBA
**Compétences** : rigueur, qualité des données, automatisation, data reporting

### 🔹 **IRD - Ingénieur stagiaire (modélisation numérique)** (2013)
- Intégration et interpolation de données physiques 3D sous GOCAD  
- Génération de modèles par inversion et analyses exploratoires
**Environnement technologique** : Gocad, Word, Excel
**Compétences** : traitement de données, interpolation, modélisation scientifique

---

## Soft Skills
- Proactivité · Curiosité intellectuelle · Fiabilité  
- Esprit d’équipe · Communication claire · Aisance relationnelle  
- Sens de la rigueur et du résultat  

---

## Centres d’intérêt
Lecture technique & IA | Football | Cuisine | Poésie  

---

## Me retrouver
- **Portfolio en ligne** -> [momo3972.github.io/Portfolio-Data-IA](https://momo3972.github.io/Portfolio-Data-IA/)
- **GitHub** -> [github.com/Momo3972](https://github.com/Momo3972) 

---

*Je cherche à rejoindre une équipe data ambitieuse pour transformer les données en valeur métier réelle, en combinant rigueur analytique, esprit d’ingénierie et créativité IA.*
