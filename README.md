# Mohamed Lamine OULD BOUYA  
**Data Scientist · Data Engineer · Data Analyst · IA**

> Je conçois des solutions data robustes, explicables et utiles : de l’ingestion à la mise en production, en passant par la modélisation ML/IA et la data visualisation.

Rueil-Malmaison  
ouldbouya.mohamedlamine@gmail.com  
+33 7 60 15 54 08  
Objectif : **stage de fin d’études (6 mois)** à partir de début janvier 2026  
Intérêts : Machine Learning - Data visulisation - IA générative - Qualité des données - Deep learning - Cloud - ETL

---

## À propos de moi

Actuellement en **Mastère Spécialisé Expert Big Data Engineer à l'Université de Technologie de Troyes (UTT)**, je combine un parcours **ingénieur** (analyse de risques, reporting automatisé, data qualité) et une solide formation en **science des données**.  
Je m’intéresse particulièrement à la création de pipelines de données robustes, à l'analyse et la visualisation via API, à l'IA générative, à la modélisation et la mise en production de modèles IA.

---

## Projets phares

### 1. [Everflow API Analytics](https://github.com/Momo3972/Everflow-API-Analytics)
> Création d’un mini-dashboard analytique pour visualiser la performance marketing via l’API Everflow
- **Stack** : Python, Pandas, Matplotlib, Requests, Everflow API  
- **Méthodes utilisées** :
  - Connexion sécurisée à l’API Everflow (authentification via clé API)
  - Extraction et transformation de statistiques agrégées sur une plage de dates données
  - Calcul du profit (revenue - payout)
  - Génération automatique de graphiques (offres, affiliés, annonceurs)
  - Export automatique d’un rapport Markdown  
- **Résultat** : mini-dashboard analytique et réutilisable
Notebook : `Everflow-API-Analytics.ipynb`

### 2. [Classification d’Images CIFAR-10 (CNN et Transfer Learning EfficientNetB0)](https://github.com/Momo3972/deepvision-cifar10-classifier)
> Développement d’un système complet de classification d’images basé sur le dataset CIFAR-10, incluant un modèle CNN construit from scratch et un modèle EfficientNetB0 utilisant le Transfer Learning et le Fine-Tuning
- **Stack** : Python, TensorFlow / Keras, NumPy, Matplotlib, Scikit-learn, Google Colab  
- **Objectif** : Construire et comparer deux approches pour classifier les images CIFAR-10 :
 - un CNN baseline simple entraîné from scratch
 - un modèle EfficientNetB0 pré-entraîné sur ImageNet puis affiné (fine-tuning) pour CIFAR-10
L’objectif est de mesurer l’impact du Transfer Learning sur la performance finale 
- **Méthodes utilisées** :
Exploration & Préparation
- Visualisation d'exemples du dataset CIFAR-10
- Normalisation des images
- Création de pipelines d’entraînement, validation et test (tf.data)

Modèle CNN (baseline)
- Architecture personnalisée : Conv2D → MaxPool → Dropout → Dense
- Entraînement complet sur CIFAR-10
- Analyse des courbes d’apprentissage (accuracy / loss)

Transfer Learning - EfficientNetB0
- Chargement d’un modèle pré-entraîné (ImageNet)
- Première phase : backbone gelé + classification head personnalisée
- Deuxième phase : fine-tuning complet
- Suivi des performances sur 2 phases concaténées

Évaluation
- Rapport de classification (precision, recall, f1-score)
- Matrice de confusion détaillée
- Comparaison finale CNN vs EfficientNetB0
- Gain absolu d’accuracy sur le test set

**Résultat**

CNN baseline :
- Test accuracy ≈ 0.70
- Test loss ≈ 0.86

EfficientNetB0 (Transfer Learning) :
- Test accuracy ≈ 0.95
- Test loss ≈ 0.16
- Gain absolu ≈ +0.24

Le modèle EfficientNetB0 offre une amélioration nette sur toutes les classes, confirmée par les matrices de confusion et scores F1.

- **Livrables** :

Notebook complet : 01_cifar10_cnn.ipynb
Modèles entraînés :
- cnn_baseline_cifar10.h5
- efficientnetb0_tl_cifar10.h5
Rapport automatisé PDF
README du projet avec structure et explication
Visualisations : courbes d’apprentissage, matrices de confusion

**Résumé**

Ce projet démontre l’intérêt du Transfer Learning en vision par ordinateur et met en évidence l’écart de performance entre un CNN traditionnel et un modèle moderne pré-entraîné

---

### 3. [Chatbot RAG IA](https://github.com/Momo3972/chatbot-rag-ia-gen)
> Développement d’un Chatbot IA avec RAG et interface Web (Assistant conversationnel intelligent relié à une base documentaire)
- **Stack** : Python, LLM (LangChain / OpenAI), RAG, Gradio / Streamlit  
- **Objectif** : permettre à un utilisateur d’interroger dynamiquement des documents PDF et d’obtenir des réponses contextuelles 
- **Méthodes utilisées** :
 - Conception d’une chaîne RAG complète (indexation, embeddings, retrieval et génération)
 - Intégration d’API IA et création d’une interface Web interactive (Gradio / Streamlit)
- **Livrables** :
 - Application Web interactive  
 - Chaîne RAG complète (embedding + retrieval + génération)  
 - Évaluation basique de la pertinence des réponses  

---

### 4. [Détection de fraude bancaire](https://github.com/Momo3972/projet-fraude)
> Analyse et modélisation de transactions bancaires pour identifier des signaux faibles de fraude
- **Stack** : Python, Pandas, NumPy, Scikit-learn, XGBoost  
- **Objectif** : améliorer le **rappel** sans dégrader la **précision** sur classes rares  
- **Méthodes utilisées** :
  - Nettoyage, préparation et analyse des données (EDA, catégorielle et temporelle)
  - Gestion du déséquilibre via **SMOTE**
  - Entraînement et optimisation de modèles : Régression logistique, Random Forest, XGBoost ; évaluation avec F1-score, AUC-ROC, précision-rappel
- **Résultat** : amélioration du score **F1** et meilleure détection de fraudes rares  

---

### 5. [Dashboard Power BI - Analyse de la performance commerciale](https://github.com/Momo3972/projet-powerbi-superstore)

> Création d’un tableau de bord interactif pour analyser les ventes, profits et performances commerciales du dataset Global Superstore

- **Stack** : Power BI Desktop, Power Query, DAX, Excel  
- **Objectif** : fournir un tableau de bord professionnel permettant :
  - d’analyser l’évolution du chiffre d’affaires,
  - d’identifier les pays contributeurs,
  - de visualiser la répartition des ventes par catégories de produits,
  - et de suivre les KPIs essentiels (ventes, profits, volume, marges).

- **Méthodes utilisées** :
  - Analyse des besoins métier et identification des indicateurs clés (KPI)
  - Nettoyage, transformation et modélisation des données via **Power Query**
  - Modélisation en étoile (**tables de faits et dimensions**)
  - Création de mesures DAX : Total Ventes, Total Profit, Quantité vendue, Marge
  - Visualisations avancées :
    - Graphique temporel des ventes (année / mois)
    - Top 10 des pays par chiffre d’affaires
    - Répartition des ventes par catégorie de produits
  - Filtres dynamiques : année, segment client, catégorie produit, pays
  - Page d’infobulle (tooltip) personnalisée pour contextualiser les ventes
  - Page “À propos” documentant la démarche analytique

- **Livrables** :
  - Tableau de bord Power BI complet : **analyse des performances commerciales**
  - Visualisations interactives + filtres dynamiques + infobulle contextualisée
  - Documentation claire (README + page dédiée dans Power BI)

---

### 6. [Prédiction d'un réservoir pétrolier](https://github.com/Momo3972/oil-reservoir-prediction-ml)

> Prédiction de la présence d’un réservoir pétrolier à partir de données géologiques et sismiques simulées - avec analyse d’interprétabilité SHAP pour valider la cohérence géologique

- **Stack** : Python, Pandas, NumPy, Scikit-learn, Random Forest, XGBoost, Matplotlib, SHAP
- **Objectif** : prédire la présence d’hydrocarbures avant forage, en exploitant des caractéristiques géologiques (porosité, type de roche, piège, profondeur, distance aux champs existants, signature sismique)
- **Méthodes utilisées** :
  - Analyse exploratoire (EDA) géologique
  - Préparation des données & encodage des variables catégorielles
  - Entraînement et optimisation d’un modèle (GridSearchCV)
  - Comparaison de trois modèles : **Logistic Regression**, **Random Forest optimisé**, **XGBoost optimisé**
  - Évaluation approfondie (Accuracy, Recall, Precision, F1, ROC-AUC)
  - Interprétabilité avancée avec **SHAP** :
    - Summary Plot (vue globale des variables)
    - Force Plot (explication locale d’une observation)
    - Bar Plot (importance moyenne des features)
- **Résultat** :
  - Le **Random Forest optimisé** obtient le meilleur score (**ROC-AUC ≈ 0.87**)
  - Les variables les plus déterminantes sont :
    - **Seismic_Score** (signal sismique fort -> structures favorables)
    - **Rock_Type** (grès / calcaire -> bons réservoirs)
    - **Trap_Type** (anticline / faille / dôme → accumulation d’hydrocarbures)
    - **Porosity** et **Permeability** (qualité du réservoir)
    - **Distance aux champs existants**
  - L’analyse SHAP confirme que le modèle prend des décisions **géologiquement cohérentes**
- **Livrables** :
  - Notebook complet d'analyse & modélisation (`oil-prediction.ipynb`)
  - Modèle optimisé exporté : `best_random_forest_oil_reservoir.joblib`
  - Visualisations : matrice de confusion, ROC curve, summary SHAP, force plot, barplot SHAP
  - README complet documentant la démarche scientifique et géologique

---

## Compétences techniques

| Domaine | Compétences |
|----------|-------------|
| **Langages** | Python, SQL, R, Excel |
| **Machine Learning** | Scikit-learn, XGBoost, PCA, SMOTE |
| **Deep Learning / Computer vision** | CNN, Transfer Learning, EfficientNet, Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Google Colab |
| **Visualisation** | Power BI, Tableau, Plotly, Matplotlib, Seaborn |
| **Base de données** | MySQL, MongoDB |
| **Cloud / Big Data** | Google Cloud Platform (GCP), Snowflake, Databricks |
| **Data Engineering** | ETL, pipelines, ingestion multi-source, EDA, data quality |
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
- **LinkedIn** -> [linkedin.com/in/mohamedlamineouldbouya](https://linkedin.com/in/mohamed-lamineould-bouya-ab465211b)  

---

> *Je cherche à rejoindre une équipe data ambitieuse pour transformer les données en valeur métier réelle, en combinant rigueur analytique, esprit d’ingénierie et créativité IA.*
