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

# Suppress torch watcher warning
import os
os.environ['STREAMLIT_WATCHER_SUPPRESS_TORCH_WARNING'] = '1'

nltk.download('punkt')
nltk.download('punkt_tab')

st.set_page_config(
    page_title="Alzheimer PFE",
    layout="centered"  # Use 'wide' for full width
)

# Application title
st.title("EARLY DETECTION OF ALZHEIMER'S DISEASE THROUGH THE STUDY OF ORAL EXPRESSION TRANSCRIBED IN TEXT FORM")

# Table of contents in the sidebar
st.sidebar.header("Table of Contents")
st.sidebar.markdown("[INTRODUCTION](#introduction)")
st.sidebar.markdown("[INITIAL AMBITION](#initial-ambition)")
st.sidebar.markdown("[OUR APPROACH](#our-approach)")
st.sidebar.markdown("[INITIAL DATA ANALYSIS](#initial-data-analysis)")
st.sidebar.markdown("[TRANSCRIPTION](#transcription)")
st.sidebar.markdown("[SILENCES DATASET](#silences-dataset)")
st.sidebar.markdown("[MODELS AND PERFORMANCES](#models-and-performances)")
st.sidebar.markdown("[TRANSCRIPTION CHOICE](#transcription-choice)")
st.sidebar.markdown("[BERT AND SBERT CHOICE](#bert-and-sbert-choice)")
st.sidebar.markdown("[EMBEDDING CHOICE](#embedding-choice)")
st.sidebar.markdown("[EARLY STOPPING CHOICE](#early-stopping-choice)")
st.sidebar.markdown("[SEQUENTIAL MODEL FOR SILENCE ANALYSIS](#sequential-model-for-silence-analysis)")
st.sidebar.markdown("[LINEAR REGRESSION MODELS FUSION](#linear-regression-models-fusion)")
st.sidebar.markdown("[DEMONSTRATION](#demonstration)")
st.markdown("""
### INTRODUCTION
Alzheimer's disease is a neurodegenerative disease causing a progressive
decline in cognitive functions. Early detection plays a crucial role in enabling
rapid care, targeted interventions, and better disease management. This research
aims to automatically detect the early signs of Alzheimer's disease by analyzing
oral expression transcribed into written form.
This project is based on linguistic data obtained from audio recordings
of patients at different stages of the disease, converted to text using speech
recognition technologies. The analysis focuses on linguistic characteristics such
as lexical diversity, syntactic structure, and language errors, considered as
potential indicators of the disease.
The main objective is to design an artificial intelligence model capable of
effectively identifying subjects at increased risk of developing Alzheimer's disease.
To achieve this, a rigorous methodological approach will be adopted, including
a literature review, data collection and processing, as well as a comparison of
different algorithm performances.
This work contributes to the improvement of diagnostic tools in the field
of neuroscience, while highlighting the promising role of artificial intelligence
in the healthcare sector.
""")

# Create five columns
col1, col2, col3, col4, col5 = st.columns(5)

# Column 1: Your photo and presentation
with col2:
    st.image("./Gabriel_pp.jpg", caption="Gabriel CHABREDIER", use_container_width=True)

# Column 2: Your colleague's photo and presentation
with col4:
    st.image("./Valentine_pp.jpeg", caption="Valentine GOBERT", use_container_width=True)

st.markdown("""
### INITIAL AMBITION
After studying the research papers published on the subject, we set ourselves the goal of achieving a performance exceeding 80% accuracy and F1-score.
            
## OUR APPROACH
            
### INITIAL DATA ANALYSIS
When starting our project, our first request was to have access to the original data to gain a better understanding of the starting point. Our project is based on two groups of audio files.
""")

# Create five columns
col1, col2, col3, col4, col5 = st.columns(5)

# CSS styles for buttons
button_style = """
<style>
.button {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 150px;
    height: 50px;
    border-radius: 10px;
    background-color: #90ee90; /* Light green */
    color: black;
    font-size: 16px;
    border: none;
}
</style>
"""

# Inject CSS
st.markdown(button_style, unsafe_allow_html=True)

# Column content with styled buttons
with col2:
    st.markdown("**166 training audios**")

with col4:
    st.markdown("")
    st.markdown("**71 test audios**")

st.markdown("""
### TRANSCRIPTION
While waiting for access to the audio files, we received from our supervisor the transcriptions he had already made. By reading the transcriptions and listening to the audio files, we noticed that the quality of the audio files is variable and that some transcriptions suffer from it. Our supervisor's transcriptions were able to hear the patient's words through noise that made it impossible for us to transcribe (given the small number of audio files, we had considered doing manual transcription). We worked on creating our own transcription model, compared it to our supervisor's transcriptions, and found that our outputs were similar. By comparing our transcriptions and our audio files, we realized that information was disappearing during this step: "Pause and silence times". By default, advanced transcription methods correct certain inconsistencies and do not mention the pauses and silences present in the audio files. So we set ourselves the goal of developing a new transcription preserving this information.
""")
st.divider()
# Load Excel file
df = pd.read_excel("stacking_train.xlsx")

# Toggle to choose between CN and AD
isAD = st.toggle("CN or AD")

# Create columns
col1, col2, col3 = st.columns(3)

if isAD:
    transcription_id = "adrso077"  # The ID to search for
    # Play audio
    
else:
    transcription_id = "adrso312"

with col1:
        st.markdown(f"**Audio of subject {transcription_id}**")
        st.audio(f"./{transcription_id}.wav")

transcription_text_encadrant = df.loc[df['id'] == transcription_id, 'transcript'].values
with col2:
        st.markdown("**Our supervisor's transcription**")
        st.markdown(transcription_text_encadrant[0])  # Display the first result found
    
transcription_text = df.loc[df['id'] == transcription_id, 'Transcription'].values

with col3:
        st.markdown("**Our Transcription**")
        st.markdown(transcription_text[0])  # Display the first result found

st.divider()

st.markdown("""
### SILENCES DATASET
 In addition to this transcription, we extracted the length of each audio file and the list of its silences. Once the data was extracted, we wanted to prove that silence analysis was relevant to our project. We then created a new dataset
 adding various ratios between "Silences" and "audio duration".

""")

st.divider()
st.markdown("**Dataframe with silences**")
df_train = pd.read_excel("transcriptions_finale.xlsx")
st.dataframe(df_train)
st.image("./Box_plot_silences.png", caption="[Box plots] Cumulative silence time per subject - Number of Silences per subject - Average silence duration per subject", use_container_width=True)
col1, col2, col3 = st.columns(3)
with col1:
     st.image("./Silences_distribution_gaussian.png", caption="Distribution of Silences according to their length - AD and CN", use_container_width= True)
with col2:
     st.image("./Distribution_silences_tempstotal.png", caption="Distribution and Gaussian of the ratio between cumulative silence and audio length - AD and CN", use_container_width= True)
with col3:
     st.image("./Distribution_tempscumulé_silences.png", caption="Distribution and Gaussian of cumulative silence time per subject - AD and CN", use_container_width=True)
st.markdown("""
With all these graphical representations, we can see a difference between
AD and CN subjects. This proves that we have exploitable data, although we
cannot confirm that it is significant for decision-making.
""")
st.divider()
st.markdown("""
### MODELS AND PERFORMANCES
#### TRANSCRIPTION CHOICE
First, we wanted to compare the model performances according to the
transcription used. We then created several models using BERT and SBERT embeddings.
Since embeddings have a limited dimension, they truncate the end of a text if it
exceeds the maximum number of tokens. With BERT we have up to 512 tokens
and 256 for SBERT. We therefore developed a token tracker to segment our
transcriptions and ensure that we avoid information loss. The segmentation
was relevant for SBERT embeddings because the limit was low, but this
segmentation was not necessary for BERT embeddings.
""")
with st.expander("**Supervisor's Transcription**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image("./plot/Bert_encadrant_plot.png", caption="BERT binary classification model using our supervisor's transcriptions [Early Stopping: epoch 8]", use_container_width=True)
    with col2:
        st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (1st) [Early stopping: epoch 388]", use_container_width=True)
with st.expander("**Silences Transcription**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image("./plot/bert_plot.png", caption="BERT binary classification model using transcriptions including silences [Early Stopping: epoch 7]", use_container_width=True)
    with col2:
        st.image("./plot/1st_embedding_SBERT_plot.png", caption="SBERT binary classification model using transcriptions including silences (1st) [Early stopping: epoch 476]", use_container_width=True)
st.markdown("""
By comparing these four models, we can see that learning seems slightly
better with our supervisor's transcriptions. Several hypotheses can be made: our
way of including silences may disturb the interpretation of the transcription and
alter the embedding; by wanting to preserve the patients' errors, we took a high
temperature for our transcription, given the variable quality of the audio files, it is
possible that our transcriptions of control people (CN) were strongly distorted;
since silences can occur in the middle of a sentence, they must have caused a
loss of context that could have impacted the embeddings.
However, although notable, the differences between the model performances
remain small.
""")

# Load data from Excel file
df_perf_model = pd.read_excel("entire_model_info.xlsx")

model_selected = [
    "BERT.pth",
    "BERT_encadrant.pth",
    "1st_embedding_SBERT.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Display in Streamlit
st.markdown("**Model performances according to the transcription used**")
st.dataframe(filtered_df)

st.markdown("""
#### BERT AND SBERT CHOICE
SBERT is an adaptation of BERT specifically designed to generate sentence
representations instead of words. This allows obtaining representations better suited
to semantic similarity detection. In theory, SBERT is therefore more appropriate for
text clustering. To confirm this, we will compare the performances of several models:
""")
with st.expander("**BERT**", expanded=False):
    st.image("./plot/Bert_encadrant_plot.png", caption="BERT binary classification model using our supervisor's transcriptions [Early Stopping: epoch 8]", use_container_width=True)
with st.expander("**SBERT (1st & 2nd)**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (1st) [Early stopping: epoch 388]", use_container_width=True)
        with col2:
            st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (2nd) [Early stopping: epoch 2024]", use_container_width=True)
st.markdown("""
Through the performances of these three models, we notice a tendency
to overfit with BERT, and the performance indicators are generally higher with
SBERT. This confirms our hypothesis: SBERT is therefore more suitable for our
classification case.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant.pth",
    "BERT_encadrant.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Display in Streamlit
st.markdown("**Model performances according to the transcription used**")
st.dataframe(filtered_df)

st.markdown("""
#### EMBEDDING CHOICE
In parallel, when we created our BERT and SBERT models, we encountered
a problem. Since embeddings have a limited dimension, they truncate the end
of a text if it exceeds the maximum number of tokens. With BERT we have up to
512 tokens and 256 for SBERT. We therefore developed a token tracker to segment
our transcriptions and ensure that we avoid information loss. The segmentation
was relevant for SBERT embeddings because the limit was low, but this segmentation
was not necessary for BERT embeddings. Since it was impossible to encode our
transcription all at once, we tried and compared two approaches.
""")  

with st.expander("**1st Embedding**", expanded=False):
    st.markdown("""
For the "1st embedding" approach, we considered each piece of text as a new
individual. This allowed data augmentation, going from 166 individuals to 223
for training and from 71 to 101 for testing.
""")  
    st.image("./plot/1st_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (1st) [Early stopping: epoch 388]", use_container_width=True)

with st.expander("**2nd Embedding**", expanded=False):
    st.markdown("""
The "2nd Embedding" approach is an embedding averaging approach.
Averaging embeddings allows preserving the meaning of representations because
it combines vectors while maintaining semantic relationships. This approach
attenuates noise by balancing individual variations and allows preserving text
characteristics. However, the meaning becomes more global and loses precision.
""")
    st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (2nd) [Early stopping: epoch 2024]", use_container_width=True)

st.markdown("""
It is difficult to decide between the two approaches. The performance
parameters are very close, with a slight advantage for the "2nd embedding".
Although the data augmentation provided by the "1st embedding" is appreciable,
its impact on performance remains limited. Therefore, it is preferable to opt for
the "2nd embedding". This second approach ensures complete integration of each
individual's statements, while avoiding data distortion due to the fragmentation
of our transcriptions.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant.pth",
    "1st_embedding_SBERT_encadrant.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Display in Streamlit
st.markdown("**Model performances according to embedding type**")
st.dataframe(filtered_df)

st.markdown("""
#### EARLY STOPPING CHOICE
When training our models, we implemented an early stopping system. This
mechanism allows us to define a number of epochs, called patience. If the model's
performance does not improve during this number of epochs, training stops.
With this approach, we save the model that achieved the best performance.
This helps us avoid overfitting and ensure that the model continues to improve
according to our performance indicator. Additionally, this gives us the necessary
flexibility to adjust other hyperparameters, such as the dropout rate, the number
of layers, and the learning rate. Initially, our goal was to achieve good accuracy,
which led us to choose this performance indicator for our early stopping strategy.
However, to determine whether this choice is appropriate, we will compare our
models by modifying the performance indicator used for early stopping.
""") 
with st.expander("**Accuracy (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_plot.png", caption="SBERT binary classification model using our supervisor's transcriptions (2nd) [Early stopping (Accuracy): epoch 2024]", use_container_width=True)
with st.expander("**Loss (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_loss.png", caption="SBERT binary classification model using our supervisor's transcriptions (2nd) [Early stopping (Loss): epoch 3040]", use_container_width=True)
with st.expander("**F1-score (2nd)**", expanded=False):
    st.image("./plot/2nd_embedding_SBERT_encadrant_f1-score.png", caption="SBERT binary classification model using our supervisor's transcriptions (2nd) [Early stopping (F1-score): epoch 869]", use_container_width=True)

st.markdown("""
All models initially start from the same neural network. However, we observe
that the choice of performance indicator used for early stopping has a significant
impact on the results.
On one hand, the model that uses loss minimization ends up overfitting. On
the other hand, the one that focuses on F1-score maximization shows balanced
performance between the training set and the test set.
However, in the end, it is the model that used accuracy as the early stopping
criterion that achieves the best overall performance indicators.
""")

model_selected = [
    "2nd_embedding_SBERT_encadrant_loss.pth",
    "2nd_embedding_SBERT_encadrant.pth",
    "2nd_embedding_SBERT_encadrant_F1-score.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Display in Streamlit
st.markdown("**Model performances according to Early Stopping indicator**")
st.dataframe(filtered_df)

st.markdown("""
#### SEQUENTIAL MODEL FOR SILENCE ANALYSIS
So far, we have discussed our transcription dataset. We also have another
dataset to analyze, the one concerning our silences. Before evaluating the model
performances, we will examine the correlation between our variables and our labels.
""")
st.image("./plot/Correlation_matrix.png", caption="Correlation matrix between silence variables and our label.", use_container_width=True)
st.markdown("We can observe that the data is relevant, but it is not necessarily significant for decision-making.")
st.image("./plot/Binary_Classifier_model_Silences_plot.png", caption="Binary classification model based on Silences [Early stopping (accuracy): epoch 171]", use_container_width=True)
st.image("./plot/Shap_summary_plot_binary_classifier.png", caption="Shap summary plot of the impact of variables on the binary classification model output", use_container_width=True)

st.markdown("""
We observe that the performances of these models are not up to par compared
to those of SBERT models using the transcriptions provided by our supervisor.
This nevertheless suggests that our data allows this binary classification, even if
the results are not optimal.
""")

model_selected = [
    "Binary_Classifier_model_Silences.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)]

# Display in Streamlit
st.markdown("**Binary classification model performance**")
st.dataframe(filtered_df)

st.markdown("""
### LINEAR REGRESSION MODELS FUSION 
Originally, we had planned to include our silence information directly in our
transcriptions. However, we found that this harmed performance. We therefore
separated this information and created a classification model for each type of
input data.
We now wonder whether performance could be improved if we merge these
models. To explore this possibility, we created stacking models.
To facilitate this fusion, we adapted our best binary classification models
into linear regression models.
""")

model_selected = [
    "2nd_embedding_linear_regression_SBERT_encadrant.pth",
    "2nd_embedding_linear_regression_SBERT_encadrant_loss.pth",
    "2nd_embedding_linear_regression_SBERT_encadrant_RMSE.pth",
    "Linear_Regression_model_Silences.pth"
]
# Filter to include only selected models
filtered_df = df_perf_model[df_perf_model['Model Name'].isin(model_selected)].drop('F1 Score', axis=1)

# Display in Streamlit
st.markdown("**Linear regression model performances**")
st.dataframe(filtered_df)

# Load data from CSV file
df_metrics = pd.read_csv("./linear_regression_metamodel_metrics.csv")

# Separate data for Train and Test
train_data = df_metrics[df_metrics['Dataset'] == 'Train'].drop('Dataset', axis=1)
test_data = df_metrics[df_metrics['Dataset'] == 'Test'].drop('Dataset', axis=1)

# Display in Streamlit
st.markdown("### Metamodel Metrics: Training vs Test")

# Table for training data
st.markdown("#### Training Data")
st.dataframe(train_data)

# Table for test data
st.markdown("#### Test Data")
st.dataframe(test_data)


st.markdown("""
## DEMONSTRATION

""")

# Load Excel file
df_test = pd.read_excel("stacking_test.xlsx")

# Create a select box to choose a patient
patient_id = st.selectbox("Choose a patient", options=df_test['id'].unique())
with st.expander("**Audio and Transcription**", expanded=False): 
    # Create columns
    col1, col2, col3 = st.columns(3)

    with col1:
            st.markdown(f"**Audio of subject {patient_id}**")
            st.audio(f"./audio_test/{patient_id}.wav")

    transcription_text_encadrant = df_test.loc[df_test['id'] == patient_id, 'transcript'].values
    with col2:
            st.markdown("**Our supervisor's transcription**")
            st.markdown(transcription_text_encadrant[0])  # Display the first result found
        
    transcription_text = df_test.loc[df_test['id'] == patient_id, 'Transcription'].values

    with col3:
            st.markdown("**Our Transcription**")
            st.markdown(transcription_text[0])  # Display the first result found

#######################
def segment_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Add the current sentence to the current chunk
        current_chunk.append(sentence)
        # Count the number of words in the current chunk
        word_count = sum(len(s.split()) for s in current_chunk)

        # If the chunk exceeds the maximum number of words, save it
        # I chose 120 so that we cannot exceed 256 word pieces without being truncated.
        if word_count >= 120:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Add the last chunk if it contains sentences
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
chunk_embeddings = []  # To store chunk embeddings
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
for chunk in chunks:
    embedding = st_model.encode(chunk) 
    chunk_embeddings.append(embedding)
if chunk_embeddings:  # Check that the list is not empty
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
with st.expander("**Silence data**", expanded=False):
     st.dataframe(x1)
with st.expander("**Models and predictions**", expanded=False):
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

# Display the result
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
    'Models': model_names,
    'Results': results,
    'Correct Prediction': emojis
})
with st.expander("**Diagnosis**", expanded=False):
    diagnostic = "Alzheimer Patient" if y.item() == 1 else "Control Patient"
    st.title("Diagnosis: "+str(diagnostic))
    st.write(results_df)

#if st.button("Send balloons!"):
#    st.balloons()
# Thank you BEYONCE
