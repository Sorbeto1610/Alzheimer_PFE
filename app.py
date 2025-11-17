import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import nltk
import torch
from torch import nn
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Suppression de l'avertissement du watcher torch
import os
os.environ['STREAMLIT_WATCHER_SUPPRESS_TORCH_WARNING'] = '1'

nltk.download('punkt')
nltk.download('punkt_tab')

st.set_page_config(
    page_title="PFE Alzheimer",
    layout="centered"  # Utilise 'wide' pour une largeur complète
)

# Titre de l'application
st.title("DÉTECTION PRÉCOCE DE LA MALADIE D’ALZHEIMER PAR ETUDE DE L’EXPRESSION ORALE RETRANSCRITE À L’ÉCRIT")

# Sommaire dans la barre latérale
st.sidebar.header("Sommaire")
st.sidebar.markdown("[INTRODUCTION](#introduction)")
st.sidebar.markdown("[AMBITION INITIALE](#ambition-initiale)")
st.sidebar.markdown("[NOTRE APPROCHE](#notre-approche)")
st.sidebar.markdown("[ANALYSE DES DONNEES INITIALE](#analyse-des-donnees-initiale)")
st.sidebar.markdown("[TRANSCRIPTION](#transcription)")
st.sidebar.markdown("[DATASET DES SILENCES](#dataset-des-silences)")
st.sidebar.markdown("[MODELES ET PERFORMANCES](#modeles-et-performances)")
st.sidebar.markdown("[CHOIX DE LA TRANSCRIPTION](#choix-de-la-transcription)")
st.sidebar.markdown("[CHOIX BERT ET SBERT](#choix-bert-et-sbert)")
st.sidebar.markdown("[CHOIX EMBEDDING](#choix-embedding)")
st.sidebar.markdown("[CHOIX EARLY STOPPING](#choix-early-stopping)")
st.sidebar.markdown("[MODELE SEQUENTIEL POUR ANALYSE DES SILENCES](#modèle-sequentiel-danalyse-des-silences)")
st.sidebar.markdown("[FUSION DES MODELES DE REGRESSION LINEAIRE](#fusion-des-modeles-de-regression-lineaire)")
st.sidebar.markdown("[DEMONSTRATION](#demonstration)")
st.markdown("""
### INTRODUCTION
La maladie d’Alzheimer est une maladie neurodégénérative entraînant un
déclin progressif des fonctions cognitives. La détection précoce joue un rôle crucial
pour permettre une prise en charge rapide, des interventions ciblées et une
meilleure gestion de la maladie. Cette recherche vise à détecter automatiquement
les premiers signes de la maladie d’Alzheimer en analysant l’expression orale
transcrite sous forme écrite.
Ce projet s’appuie sur des données linguistiques obtenues à partir d’enregistrements
audio de patients à différents stades de la maladie, convertis en texte à
l’aide de technologies de reconnaissance vocale. L’analyse porte sur des caractéristiques
linguistiques telles que la diversité lexicale, la structure syntaxique et
les erreurs langagières, considérées comme des indicateurs potentiels de la maladie.
L’objectif principal est de concevoir un modèle d’intelligence artificielle
capable d’identifier efficacement les sujets présentant un risque accru de développer
la maladie d’Alzheimer. Pour cela, une approche méthodologique rigoureuse
sera adoptée, incluant une revue bibliographique, la collecte et le traitement des
données, ainsi qu’une comparaison des performances de différents algorithmes.
Ce travail contribue à l’amélioration des outils de diagnostic dans le domaine
des neurosciences, tout en soulignant le rôle prometteur de l’intelligence
artificielle dans le secteur de la santé.
""")

# Créer deux colonnes
col1, col2, col3, col4, col5 = st.columns(5)

# Colonne 1 : Votre photo et présentation
with col2:
    st.image("./Gabriel_pp.jpg", caption="Gabriel CHABREDIER", use_container_width=True)

# Colonne 2 : Photo de votre collègue et présentation
with col4:
    st.image("./Valentine_pp.jpeg", caption="Valentine GOBERT", use_container_width=True)

st.markdown("""
### AMBITION INITIALE
Après avoir étudié les travaux de recherche paru sur le sujet, nous nous sommes fixé comme objectif d'atteindre une performance dépassant les 80% de précision et de F1-score.
            
## NOTRE APPROCHE
            
### ANALYSE DES DONNEES INITIALE
En commençant notre projet, notre première demande a été d’avoir accès à la donnée d’origine pour avoir une meilleure compréhension du point de départ. Notre projet se base sur deux groupes d’audio.
""")

# Créer deux colonnes
col1, col2, col3, col4, col5 = st.columns(5)

# Styles CSS pour les boutons
button_style = """
<style>
.button {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 150px;
    height: 50px;
    border-radius: 10px;
    background-color: #90ee90; /* Vert clair */
    color: black;
    font-size: 16px;
    border: none;
}
</style>
"""

# Injecter le CSS
st.markdown(button_style, unsafe_allow_html=True)

# Contenu des colonnes avec des boutons stylisés
with col2:
    st.markdown("**166 audios d'entraînement**")

with col4:
    st.markdown("")
    st.markdown("**71 audios de test**")

st.markdown("""
### TRANSCRIPTION
En attendant l’accès aux audios, nous avons reçu de la part de notre encadrant les transcriptions qu’il avait déjà réalisées. En lisant les transcriptions et en écoutant les audios, nous avons pu
constater que la qualité des audios est variable et que certaines transcriptions en pâtissent. Les transcriptions de notre encadrant étaient capable d’entendre les propos du patient à travers des bruits qui nous rendaient la transcription impossible à faire par nous-même (au vu de la faible quantité d’audio nous avions envisagé de
faire une transcription à l’oreille). Nous avons travaillé à faire notre propre model de transcription, nous l’avons comparé aux transcriptions de notre encadrant et avons constaté que nos sortis sont similaires. En comparant nos transcriptions et nos audios, nous nous sommes rendu compte qu’une information disparaissait
lors de cette étape : ”Les temps de pauses et silences”. Par défaut, les méthodes de transcriptions avancés corrigent certaines incohérences et ne mentionnent pas les pauses et silences présents dans les audios. Alors nous nous sommesfixés comme objectif de développer une nouvelle transcription conservant ces informations.
""")
st.divider()
# Charger le fichier Excel
df = pd.read_excel("stacking_train.xlsx")

# Toggle pour choisir entre CN et AD
isAD = st.toggle("CN ou AD")

# Créer les colonnes
col1, col2, col3 = st.columns(3)

if isAD:
    transcription_id = "adrso077"  # L'ID à chercher
    # Jouer l'audio
    
else:
    transcription_id = "adrso312"

with col1:
        st.markdown(f"**Audio du sujet {transcription_id}**")
        st.audio(f"./{transcription_id}.wav")

transcription_text_encadrant = df.loc[df['id'] == transcription_id, 'transcript'].values
with col2:
        st.markdown("**Transcription de notre encadrant**")
        st.markdown(transcription_text_encadrant[0])  # Affiche le premier résultat trouvé
    
transcription_text = df.loc[df['id'] == transcription_id, 'Transcription'].values

with col3:
        st.markdown("**Notre Transcription**")
        st.markdown(transcription_text[0])  # Affiche le premier résultat trouvé

st.divider()

st.markdown("""
### DATASET DES SILENCES
 En plus de cette transcription nous
 avons extrait la longueur de chaque audio et la liste de ses silences. Une fois les données extraites, nous avons voulu prouver que l’analyse des silences était pertinente à notre projet. Nous avons alors crée un nouveau dataset
 ajoutant divers ratios entre ”Silences” et ”audio duration”.

""")

st.divider()
st.markdown("**Dataframe avec silences**")
df_train = pd.read_excel("transcriptions_finale.xlsx")
st.dataframe(df_train)
st.image("./Box_plot_silences.png", caption="[Box plots] Temps cumulé de silence par sujet - Nombre de Silence par sujet - Durée moyenne du silence par sujet", use_container_width=True)
col1, col2, col3 = st.columns(3)
with col1:
     st.image("./Silences_distribution_gaussian.png", caption="Distribution des Silences en fonction de leur longueur - AD et CN", use_container_width= True)
with col2:
     st.image("./Distribution_silences_tempstotal.png", caption="Distribution et Gaussien du ratio entre le silence cummulé et la longueur de l’audio - AD et CN", use_container_width= True)
with col3:
     st.image("./Distribution_tempscumulé_silences.png", caption="Distribution et Gaussien du temps cumulé de silence par sujet - AD et CN", use_container_width=True)
st.markdown("""
Avec toutes ces représentations graphiques, on constate un décalage entre
les sujets AD et CN. Cela nous prouve donc que nous avons des données ex-
ploitables, bien que l’on ne puisse pas affirmer qu’elles soient significatives pour
la prise de décision.
""")
st.divider()
st.markdown("""
### MODELES ET PERFORMANCES
#### CHOIX DE LA TRANSCRIPTION
Dans un premier temps nous avons voulu comparer les performances des
modeles selon la transcription utilisée. Nous avons alors réalisée plusieurs mod-
èles utilisant des embeddings BERT et SBERT. Les embedding ayant une di-
mension limitées, ils tronquent la fin d’un texte si celui-ci dépasse son nombre
maximum de token. Avec BERT nous avons jusqu’à 512 tokens et 256 pour
SBERT. Nous avons donc développer un tracker de token pour segmenter nos
transcriptions et ainsi s’assurer d’éviter la perte d’information. La segmentation
était pertinente pour les embeddings de SBERT car la limite était basse mais cette
segmentation n’était pas nécéssaire pour les embeddings de BERT.
""")
with st.expander("**Transcription Encadrant**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image("./plot/Bert_encadrant_plot.png", caption="Modele de classification binaire BERT utilisant les transcriptions de notre encadrant [Early Stopping : epoch 8]", use_container_width=True)
    with col2:
        st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (1st) [Early stopping : epoch 388]", use_container_width=True)
with st.expander("**Transcription Silences**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image("./plot/bert_plot.png", caption="Modele de classification binaire BERT utilisant les transcriptions incluant les silences [Early Stopping : epoch 7]", use_container_width=True)
    with col2:
        st.image("./plot/1st_embedding_SBERT_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions incluant les silences (1st) [Early stopping : epoch 476]", use_container_width=True)
st.markdown("""
En comparant ces quatres modèles on peut constater que l’apprentissage
semble légèrement mieux se passer avec les transcriptions de notre encadrant.
On peut émettre plusieurs hypothèses, notre façon d’inclure les silences peut per-
turber l’interprétation de la transcription et altérer l’embedding ; en voulant con-
server les erreurs des patients nous avons pris une température élèvée pour notre
transcription, au vu de la qualité variable des audios il est possible que nos tran-
scriptions des personnes contrôles(CN) furent fortement déformées ; les silences
pouvant arriver au milieu de phrase, ils ont du engendrer une perte de contexte
qui a pu impacter les embeddings.
Cependant, bien que notable, les différences entre les performances des
modèles restent faibles.
""")

# Charger les données depuis le fichier Excel
df_perf_model = pd.read_excel("entire_model_info.xlsx")

model_selected = [
    "BERT.pth",
    "BERT_encadrant.pth",
    "1st_embedding_SBERT.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Affichage dans Streamlit
st.markdown("**Performances des modèles selon la transcription utilisée**")
st.dataframe(filtered_df)

st.markdown("""
#### CHOIX BERT ET SBERT
SBERT est une adaptation de BERT spécifiquement conçue pour générer
des représentations de phrases au lieu de mots. Cela permet d’obtenir des représentations mieux adaptées à la détection de similarité sémantique. En théorie, SBERT
est donc plus approprié pour le clustering de textes. Pour le confirmer nous allons
comparer les performances de plusieurs modèles :
""")
with st.expander("**BERT**", expanded=False):
    st.image("./plot/Bert_encadrant_plot.png", caption="Modele de classification binaire BERT utilisant les transcriptions de notre encadrant [Early Stopping : epoch 8]", use_container_width=True)
with st.expander("**SBERT (1st & 2nd)**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (1st) [Early stopping : epoch 388]", use_container_width=True)
        with col2:
            st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (2nd) [Early stopping : epoch 2024]", use_container_width=True)
st.markdown("""
A travers les performances de ces trois modèles, on remarque une tendance
à l’overfitting chez BERT, et les indicateurs de performance sont globalement
supérieurs avec SBERT. Cela confirme notre hypothèse, SBERT est donc plus
adapté à notre cas de classification.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant.pth",
    "BERT_encadrant.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Affichage dans Streamlit
st.markdown("**Performances des modèles selon la transcription utilisée**")
st.dataframe(filtered_df)

st.markdown("""
#### CHOIX EMBEDDING
En parallèle lorsque nous avons réalisé nos modèles BERT et SBERT nous
avons rencontré un problème. Les embedding ayant une dimension limitée, ils
tronquent la fin d’un texte si celui-ci dépasse son nombre maximum de token.
Avec BERT nous avons jusqu’à 512 tokens et 256 pour SBERT. Nous avons
donc développé un tracker de token pour segmenter nos transcriptions et ainsi
s’assurer d’éviter la perte d’information. La segmentation était pertinente pour
les embeddings de SBERT car la limite était basse mais cette segmentation n’était
pas nécéssaire pour les embeddings de BERT. Etant impossible d’encoder notre
transcription en une fois, nous avons tenté et comparé deux approches.
""")  

with st.expander("**1st Embedding**", expanded=False):
    st.markdown("""
Pour l’approche ”1st embedding”, nous avons considéré chaque morceau
de texte comme un nouvel individu. Cela a permis une augmentation de donnée,
passant de 166 individus à 223 d’entrainement et de 71 à 101 de test.
""")  
    st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (1st) [Early stopping : epoch 388]", use_container_width=True)

with st.expander("**2nd Embedding**", expanded=False):
    st.markdown("""
L’approche ”2nd Embedding” est une approche de moyennage d’embedding.
Faire la moyenne des embeddings permet de conserver le sens des représentations
car cela combine les vecteurs tout en maintenant les relations sémantiques. Cette
approche atténue le bruit en équilibrant les variations individuelles et permet de
préserver les caractéristiques du texte. Cependant, le sens deviens plus global et
perd de sa précision.
""")
    st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (2nd) [Early stopping : epoch 2024]", use_container_width=True)

st.markdown("""
Il est difficile de trancher entre les deux approches. Les paramètres de
performance sont très proches, avec un léger avantage pour le ”2nd embedding”.
Bien que l’augmentation de données apportée par le ”1st embedding” soit ap-
préciable, son impact sur les performances reste limitée. Par conséquent, il est
préférable d’opter pour le ”2nd embedding”. Cette seconde approche garantit
une intégration complète des propos de chaque individu, tout en évitant la défor-
mation de nos données due à la fragmentation de nos transcriptions.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Affichage dans Streamlit
st.markdown("**Performances des modèles selon le type d'embedding**")
st.dataframe(filtered_df)

st.markdown("""
#### CHOIX EARLY STOPPING
Lors de l’entraînement de nos modèles, nous avons mis en place un sys-
tème d’early stopping. Ce mécanisme nous permet de définir un nombre d’époques,
appelé patience. Si la performance du modèle ne s’améliore pas pendant ce nom-
bre d’époques, l’entraînement s’arrête.
Grâce à cette approche, nous sauvegardons le modèle ayant obtenu la meilleure performance. Cela nous aide à éviter le surapprentissage et à nous
assurer que le modèle continue de s’améliorer selon notre indicateur de perfor-
mance. De plus, cela nous offre la flexibilité nécessaire pour ajuster d’autres
hyperparamètres, tels que le taux de dropout, le nombre de couches et le taux
d’apprentissage. Dans un premier temps, notre objectif était d’atteindre une bonne
précision (accuracy), ce qui nous a conduit à choisir cet indicateur de performance
pour notre stratégie d’early stopping. Cependant, afin de déterminer si ce choix
est approprié, nous allons comparer nos modèles en modifiant l’indicateur de per-
formance utilisé pour l’early stopping.
""") 
with st.expander("**Accuracy (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (2nd) [Early stopping (Accuracy): epoch 2024]", use_container_width=True)
with st.expander("**Loss (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_loss.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (2nd) [Early stopping (Loss) : epoch 3040]", use_container_width=True)
with st.expander("**F1-score (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_f1-score.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (2nd) [Early stopping (F1-score) : epoch 869]", use_container_width=True)

st.markdown("""
Tous les modèles partent initialement du même réseau de neurones. Cepen-
dant, nous observons que le choix de l’indicateur de performance utilisé pour
l’early stopping a un impact significatif sur les résultats.
D’une part, le modèle qui utilise la minimisation de la loss finit par surap-
prendre. D’autre part, celui qui se concentre sur la maximisation du F1-score
montre une performance équilibrée entre l’ensemble d’entraînement et l’ensemble
de test.
Cependant, au final, c’est le modèle qui a utilisé l’accuracy comme critère
d’early stopping qui obtient les meilleurs indicateurs de performance globaux.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant_loss.pth",
    "2nd_embedding_SBERT_encadrant.pth",
    "2nd_embedding_SBERT_encadrant_F1-score.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Affichage dans Streamlit
st.markdown("**Performances des modèles selon indicateur en Early Stopping**")
st.dataframe(filtered_df)

st.markdown("""
#### MODELE SEQUENTIEL POUR ANALYSE DES SILENCES
Jusqu’à présent, nous avons discuté de notre jeu de données de transcrip-
tion. Nous avons également un autre jeu de données à analyser, celui concernant
nos silences. Avant d’évaluer la performance des modèles, nous allons examiner
la corrélation entre nos variables et nos étiquettes.
""")
st.image("./plot/Correlation_matrix.png", caption="Matrice de corrélation entre les variables de silence et notre label.", use_container_width=True)
st.markdown("Nous pouvons observer que les données sont pertinentes, mais elles ne sont pas nécessairement significatives pour la prise de décision.")
st.image("./plot/Binary_Classifier_model_Silences_plot.png", caption="Modele de classification binaire basé sur les Silences [Early stopping (accuracy) : epoch 171]", use_container_width=True)
st.image("./plot/Shap_summary_plot_binary_classifier.png", caption="Shap summary plot de l'impact des variables sur la sortie du modèle de classification bianire", use_container_width=True)

st.markdown("""
Nous observons que les performances de ces modèles ne sont pas à la hau-
teur par rapport à celles des modèles SBERT utilisant les transcriptions fournies
par notre encadrant. Cela suggère néanmoins que nos données permettent cette
classification binaire, même si les résultats ne sont pas optimaux."
""")

model_selected = [
    "Binary_Classifier_model_Silences.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Affichage dans Streamlit
st.markdown("**Performances du modèle de classification binaire**")
st.dataframe(filtered_df)

st.markdown("""
### FUSION DES MODELES DE REGRESSION LINEAIRE 
À l’origine, nous avions prévu d’inclure nos informations sur les silences
directement dans nos transcriptions. Cependant, nous avons constaté que cela
nuisaient aux performances. Nous avons donc séparé ces informations et créé un
modèle de classification pour chaque type de donnée d’entrée.
Nous nous interrogeons maintenant sur l’éventuelle amélioration des per-
formances si nous fusionnons ces modèles. Pour explorer cette possibilité, nous
avons réalisé des modèles de stacking.
Pour faciliter cette fusion, nous avons adapté nos meilleurs modèles de
classification binaire en modèles de régression linéaire.
""")

model_selected = [
    "2nd_embedding_linear_regression_SBERT_encadrant.pth",
    "2nd_embedding_linear_regression_SBERT_encadrant_loss.pth",
    "2nd_embedding_linear_regression_SBERT_encadrant_RMSE.pth",
    "Linear_Regression_model_Silences.pth"
]
# Filtrer pour inclure uniquement les modèles sélectionnés
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)].drop('F1 Score', axis=1)

# Affichage dans Streamlit
st.markdown("**Performances des modèles de régression linéaire**")
st.dataframe(filtered_df)

# Charger les données depuis le fichier CSV
df_metrics = pd.read_csv("./linear_regression_metamodel_metrics.csv")

# Séparer les données pour Train et Test
train_data = df_metrics[df_metrics['Dataset'] == 'Train'].drop('Dataset', axis=1)
test_data = df_metrics[df_metrics['Dataset'] == 'Test'].drop('Dataset', axis=1)

# Affichage dans Streamlit
st.markdown("### Métriques du Metamodel : Entraînement vs Test")

# Tableau pour les données d'entraînement
st.markdown("#### Données d'Entraînement")
st.dataframe(train_data)

# Tableau pour les données de test
st.markdown("#### Données de Test")
st.dataframe(test_data)


st.markdown("""
## DEMONSTRATION

""")

# Charger le fichier Excel
df_test = pd.read_excel("stacking_test.xlsx")

# Créer une select box pour choisir un patient
patient_id = st.selectbox("Choisissez un patient", options=df_test['id'].unique())
with st.expander("**Audio et Transcription**", expanded=False): 
    # Créer les colonnes
    col1, col2, col3 = st.columns(3)

    with col1:
            st.markdown(f"**Audio du sujet {patient_id}**")
            st.audio(f"./audio_test/{patient_id}.wav")

    transcription_text_encadrant = df_test.loc[df_test['id'] == patient_id, 'transcript'].values
    with col2:
            st.markdown("**Transcription de notre encadrant**")
            st.markdown(transcription_text_encadrant[0])  # Affiche le premier résultat trouvé
        
    transcription_text = df_test.loc[df_test['id'] == patient_id, 'Transcription'].values

    with col3:
            st.markdown("**Notre Transcription**")
            st.markdown(transcription_text[0])  # Affiche le premier résultat trouvé

#######################
def segment_text(text):
    # Tokenise le texte en phrases
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Ajoute la phrase actuelle au chunk courant
        current_chunk.append(sentence)
        # Compte le nombre de mots dans le chunk courant
        word_count = sum(len(s.split()) for s in current_chunk)

        # Si le chunk dépasse le nombre maximum de mots, on le sauvegarde
        # J'ai choisi 120 car comme ça on ne peut pas dépasser les 256 word piece sans que ce soit tronqué.
        if word_count >= 120:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Ajoute le dernier chunk s'il contient des phrases
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

to_drop = ['id','Transcription','Silences','Alzheimer', 'transcript', 'addressfname']
x1 =df_test.loc[df_test['id'] == patient_id].drop(to_drop,axis=1)
y= df_test.loc[df_test['id'] == patient_id]['Alzheimer']
scaler_loaded = joblib.load('stacked_scaler.pkl')
X1 = scaler_loaded.transform(x1)
transcript =df_test.loc[df_test['id'] == patient_id]['transcript'].values[0]
chunks = segment_text(transcript)
chunk_embeddings = []  # Pour stocker les embeddings des chunks
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
for chunk in chunks:
    embedding = st_model.encode(chunk) 
    chunk_embeddings.append(embedding)
if chunk_embeddings:  # Vérifier que la liste n'est pas vide
    X2 = np.mean(chunk_embeddings, axis=0)  

X1_tensor = torch.tensor(X1, dtype=torch.float32)
X2_tensor = torch.tensor(X2, dtype=torch.float32)
class ImprovedRegressionModel1(nn.Module):
    def __init__(self, input_size):
        super(ImprovedRegressionModel1, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x) 
      
class ImprovedRegressionModel2(nn.Module):
    def __init__(self, input_size):
        super(ImprovedRegressionModel2, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.dropout1 = nn.Dropout(0.55)
        self.fc2 = nn.Linear(64, 32)           
        self.dropout2 = nn.Dropout(0.6)        
        self.fc3 = nn.Linear(32, 1)         

    def forward(self, x):
        x = torch.nn.functional.softplus(self.fc1(x)) 
        x = self.dropout1(x)                            
        x = torch.nn.functional.softplus(self.fc2(x))  
        x = self.dropout2(x)                            
        return self.fc3(x)                              

input_dim1 = 5
model1 = ImprovedRegressionModel1(input_dim1)
model1.load_state_dict(torch.load('./Linear_Regression_model_Silences.pth', map_location=torch.device('cpu')))
model1.eval()

input_dim2 = 384
model2 = ImprovedRegressionModel2(input_dim2)
model2.load_state_dict(torch.load('./2nd_embedding_linear_regression_SBERT_encadrant_loss.pth', map_location=torch.device('cpu')))
model2.eval()

with torch.no_grad():
     pred1 = model1(X1_tensor)
     pred2 = model2(X2_tensor)

# Clamp predictions to ensure they are between 0 and 1
pred1_clamped = torch.clamp(pred1, 0, 1)
pred2_clamped = torch.clamp(pred2, 0, 1)
# Convert prediction to a NumPy array and extract the value
pred1_percentage = pred1_clamped.item() * 100
pred2_percentage = pred2_clamped.item() * 100

#st.dataframe(y)
#######################
with st.expander("**Données de silence**", expanded=False):
     st.dataframe(x1)
with st.expander("**Modèles et prédictions**", expanded=False):
     col1, col2 = st.columns(2)
     with col1:
        st.subheader("Silence Model")
        a, b, c = st.columns(3)
        with b:
            st.subheader(f"{pred1_percentage:.1f}%")
        cols = st.columns(10)
        cols[0].write(0)
        cols[9].write(1)
        st.progress(pred1_clamped.item())
     with col2:
        st.subheader("Transcription Model")
        a, b, c = st.columns(3)
        with b:
            st.subheader(f"{pred2_percentage:.1f}%")
        cols = st.columns(10)
        cols[0].write(0)
        cols[9].write(1)
        st.progress(pred2_clamped.item())
     
# Load the meta-model
loaded_meta_model = joblib.load('meta_model.pkl')

meta_feature = np.column_stack((pred1.cpu().numpy(), pred2.cpu().numpy()))
meta_prediction = loaded_meta_model.predict(meta_feature)
meta_prediction_tensor = torch.tensor(meta_prediction, dtype=torch.float32)
meta_prediction_clamped = torch.clamp(meta_prediction_tensor, 0, 1)

# Afficher le résultat
meta_prediction_percentage = meta_prediction_clamped.item() * 100
with st.expander("**Metamodel prediction**", expanded=False):
    st.subheader("Metamodel")
    a, b, c = st.columns(3)
    with b:
        st.title(f"{meta_prediction_percentage:.2f}%")
    cols = st.columns(10)
    cols[0].write(0)
    cols[9].write(1)
    st.progress(meta_prediction_clamped.item())


model_names = ['Silence Model', 'Transcription Model', 'Metamodel']
results = [pred1_clamped.item(), pred2_clamped.item(), meta_prediction_clamped.item()]

emojis = []
for result in results:
    predicted_class = 1 if float(result) >= 0.5 else 0
    emojis.append('👍' if predicted_class == y.item() else '👎')

results_df = pd.DataFrame({
    'Modèles': model_names,
    'Résultats': results,
    'Prédiction Correcte': emojis
})
with st.expander("**Diagnostic**", expanded=False):
    diagnostic = "Patient Alzheimer" if y.item() == 1 else "Patient Contrôle"
    st.title("Diagnostic : "+str(diagnostic))
    st.write(results_df)

#if st.button("Send balloons!"):
#    st.balloons()
# Merci BEYONCE
