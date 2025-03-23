import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PFE Alzheimer",
    layout="wide"  # Utilise 'wide' pour une largeur complète
)

# Titre de l'application
st.title("DÉTECTION PRÉCOCE DE LA MALADIE D’ALZHEIMER PAR ETUDE DE L’EXPRESSION ORALE RETRANSCRITE À L’ÉCRIT")

# Sommaire dans la barre latérale
st.sidebar.header("Sommaire")
st.sidebar.markdown("[Introduction](#introduction)")
st.sidebar.markdown("[Ambition initial](#ambition-initial)")
st.sidebar.markdown("[Notre Approche](#notre-approche)")
st.sidebar.markdown("[Analyse des données d’origine](#analyse-des-données-dorigine)")


st.markdown("""
### Introduction
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
    st.markdown("""    
    Je suis [votre rôle ou spécialité]. J'ai travaillé sur ce projet en me concentrant sur [vos contributions spécifiques].
    """)

# Colonne 2 : Photo de votre collègue et présentation
with col4:
    st.image("./Valentine_pp.jpg", caption="Valentine GOBERT", use_container_width=True)
    st.markdown("""
    Je suis [rôle ou spécialité de votre collègue]. J'ai contribué à ce projet en [contributions de votre collègue].
    """)

st.markdown("""
### Ambition initial
Après avoir étudié les travaux de recherche paru sur le sujet, nous nous sommes fixé comme objectif d'atteindre une performance dépassant les 80% de précision et de F1-score.
            
## Notre Approche
            
### Analyse des données d’origine
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
### Transcription
En attendant l’accès aux audios, nous avons reçu de la part de notre encadrant les transcriptions qu’il avait déjà réalisées. En lisant les transcriptions et en écoutant les audios, nous avons pu
constater que la qualité des audios est variable et que certaines transcriptions en pâtissent. Les transcriptions de notre encadrant étaient capable d’entendre les propos du patient à travers des bruits qui nous rendaient la transcription impossible à faire par nous-même (au vu de la faible quantité d’audio nous avions envisagé de
faire une transcription à l’oreille). Nous avons travaillé à faire notre propre model de transcription, nous l’avons comparé aux transcriptions de notre encadrant et avons constaté que nos sortis sont similaires. En comparant nos transcriptions et nos audios, nous nous sommes rendu compte qu’une information disparaissait
lors de cette étape : ”Les temps de pauses et silences”. Par défaut, les méthodes de transcriptions avancés corrigent certaines incohérences et ne mentionnent pas les pauses et silences présents dans les audios. Alors nous nous sommesfixés comme objectif de développer une nouvelle transcription conservant ces informations.
""")
st.divider()
# Charger le fichier Excel
df_train = pd.read_excel("transcriptions_finale.xlsx")
df_train_encadrant = pd.read_csv("train_scraped_encadrant.csv")

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

transcription_text_encadrant = df_train_encadrant.loc[df_train_encadrant['addressfname'] == transcription_id, 'transcript'].values
with col2:
        st.markdown("**Transcription de notre encadrant**")
        st.markdown(transcription_text_encadrant[0])  # Affiche le premier résultat trouvé
    
transcription_text = df_train.loc[df_train['id'] == transcription_id, 'Transcription'].values

with col3:
        st.markdown("**Notre Transcription**")
        st.markdown(transcription_text[0])  # Affiche le premier résultat trouvé

st.divider()

st.markdown("""
### Dataset des Silences
 En plus de cette transcription nous
 avons extrait la longueur de chaque audio et la liste de ses silences. Une fois les données extraites, nous avons voulu prouver que l’analyse des silences était pertinente à notre projet. Nous avons alors crée un nouveau dataset
 ajoutant divers ratios entre ”Silences” et ”audio duration”.

""")

st.divider()
st.markdown("**Dataframe avec silences**")
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
### Modèles et performances
#### Choix de la transcription
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
    st.image("./plot/Bert_encadrant_plot.png", caption="Modele de classification binaire BERT utilisant les transcriptions de notre encadrant [Early Stopping : epoch 8]", use_container_width=True)
    st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="Modele de classification binaire SBERT utilisant les transcriptions de notre encadrant (1st) [Early stopping : epoch 388]", use_container_width=True)
with st.expander("**Transcription Silences**", expanded=False):
    st.image("./plot/bert_plot.png", caption="Modele de classification binaire BERT utilisant les transcriptions incluant les silences [Early Stopping : epoch 7]", use_container_width=True)
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
#### Choix entre BERT et SBERT
SBERT est une adaptation de BERT spécifiquement conçue pour générer
des représentations de phrases au lieu de mots. Cela permet d’obtenir des représen-
tations mieux adaptées à la détection de similarité sémantique. En théorie, SBERT
est donc plus approprié pour le clustering de textes. Pour le confirmer nous allons
comparer les performances de plusieurs modèles :
""")

if st.button("Send balloons!"):
    st.balloons()
