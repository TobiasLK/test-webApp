# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:12:09 2023

@author: maeva
"""



# ---------------------------------------------------#
#           OVERALL SETUP                            #
# ---------------------------------------------------#

# Importing modules
import streamlit as st
import string
import base64
import nltk
import spacy
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from langdetect import detect
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import io

# Load the pre-trained BERT model
BERTmodel = BertForSequenceClassification.from_pretrained('bert-base-uncased')
BERTtokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Downloads the relevant nltk assets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

# Sets stopwords to english
stop_words = set(stopwords.words("english"))



# ---------------------------------------------------#
#           PAGE CONFIGURATIONS                      #
# ---------------------------------------------------#


# Defining some general properties of the app
st.set_page_config(
    page_title="Feedback",
    page_icon="📝",
    layout="wide")

# Adding a background image to the WebApp
'''
 def add_bg_from_local(image_file):
    with open("Background.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

 add_bg_from_local('Background.jpg')
'''


# ---------------------------------------------------#
#           HEADER SECTION                           #
# ---------------------------------------------------#


# Setting a title for the webpage
st.title("Feedback Analysis Tool")

# Text under title
st.write("""
         📝 This application is a Streamlit dashboard that can be used to
         **analyze** and **improve** feedbacks on innovation ideas in a company📝
         """)

# Making a textbox where users can enter the innovation idea
st.header("Please insert your innovation idea here")

idea = st.text_area("Enter text here: ")
submission_date = st.date_input("Enter idea submission date here: ")
st.write("You entered:", idea, " date:", submission_date)

# Making a textbox where users can enter their feedbacks on the innovation idea
st.header("Please insert your feedback on innovation idea here")

feedback = st.text_area("Enter text here:")
feedback_date = st.date_input("Enter feedback submission date here:")
st.write("You entered:", feedback, " date:", feedback_date)

# ----- PRE-PROCESSING TEXT --------------------------------

# Number of words

# Option B seems better
nlp = spacy.load('en_core_web_sm')

if feedback:
    doc = nlp(feedback)
    num_tokens = len(doc)
    st.write("Number of tokens:", num_tokens)



# ---------------------------------------------------#
#           SCORING PARAMETERS                       #
# ---------------------------------------------------#


# ----- Scoring Parameter 1: Novelty Score (NS) ------

# Opening the database file
with open("C:/Users/tobia/Desktop/Maeva Code/feedback_data.txt", "r") as f:
#with open("feedback_data.txt", "r") as f:
    lines = f.readlines()

# Remove stop words from each entry and cleaning the strings
entries = []
for line in lines:
    line = line.translate(str.maketrans('', '', string.punctuation))
    words = line.strip().split()
    filtered_words = [word.lower() for word in words if
                      word.lower() not in stop_words and word not in string.punctuation]
    entry = " ".join(filtered_words)
    entries.append(entry)

# Make list of unique feedback keywords
feedback_keywords = set([word for line in entries for word in line.strip().split()])

# Splitting input sentence into words
input_words = word_tokenize(idea)

# Function that checks how many input words are the same as the keywords
num_matches = len(set(input_words) & feedback_keywords)


def novelty_score(text):
    if not text:
        return 0.0

    novelty_ratio = num_matches / len(word_tokenize(text))
    return round(novelty_ratio, 2)


# ----- Scoring Parameter 2: Workability Score (WS) ------

def workability_score(feedback):
    # Define the constraints or requirements
    required_keywords = ['feasible', 'realistic', 'practical', 'achievable', 'implementable']

    # Check if the idea contains the required keywords
    count = 0
    for keyword in required_keywords:
        if keyword in feedback.lower():
            count += 1

    # Calculate workability score
    score = count / len(required_keywords)
    return round(score, 2)


# ----- Scoring Parameter 3: Relevance Score (RS) ------

def get_relevant_terms(feedback):
    # Tokenize text
    words = nltk.word_tokenize(feedback.lower())

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Calculate term frequency-inverse document frequency
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(words)

    # Get the most important terms based on TF-IDF scores
    tfidf_scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)
    relevant_terms = [word for word, score in sorted(tfidf_scores, key=lambda x: x[1])]

    return relevant_terms[:10]  # Return the top 10 most important terms


def relevance_score(feedback):
    relevant_terms = get_relevant_terms(feedback)
    words = word_tokenize(feedback.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    RK = sum([1 for word in words if word in relevant_terms])
    TK = len(words)
    return RK / TK if TK != 0 else 0


# ----- Scoring Parameter 4: Specificity Score (SPS) ------

# Function that checks adjectives compared to sentence

# Option B seems better
def specificity_score(feedback):
    tokens = nltk.word_tokenize(feedback)
    pos_tags = nltk.pos_tag(tokens)
    num_adjectives = len([word for word, tag in pos_tags if tag.startswith('JJ')])
    total_words = len(tokens)
    if total_words > 0:
        SS = num_adjectives / total_words
    else:
        SS = 0

    return SS


# ----- Scoring Parameter 5: Clarity Score (CS) --------

# Option A seems better
# Function that checks the different sentence requirements
def clarity_score(text):
    count = 0
    # Check if the sentence starts with a capital letter
    if text[0].isupper():
        count += 1

    # Check if the sentence ends with a period
    if text[-1] == ".":
        count += 1

    # Check if the sentence is written in English
    if detect(text) == 'en':
        count += 1

    # Tokenize the sentence and tag the parts of speech
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Check if the sentence contains at least three words including a verb
    verb_found = False
    word_count = 0
    for word, tag in pos_tags:
        if tag.startswith('V'):
            verb_found = True
        if word not in string.punctuation:
            word_count += 1
    if verb_found:
        count += 1
    if word_count > 3:
        count += 1

    return count / 5


# ---- Scoring Parameter 6: Timeliness Score (TS) -----

def timeliness_score(submission_date, feedback_date):

    # Convert submission_date and feedback_date to strings using strftime()
    submission_date_str = submission_date.strftime('%Y-%m-%d')
    feedback_date_str = feedback_date.strftime('%Y-%m-%d')

    # Calculate the number of days between the submission date and feedback date
    days_diff = (datetime.strptime(feedback_date_str, '%Y-%m-%d') - datetime.strptime(submission_date_str,
                                                                                      '%Y-%m-%d')).days

    # Calculate the timeliness score
    if days_diff < 0:
        ts = 0.0
    elif days_diff > 7:
        ts = 0.5
    else:
        ts = 1 - (days_diff / 7)

    return ts


# ----- Scoring Parameter 8: Problem-Solving Score (PS) ------

def extract_probsol_keywords(text):
    # Parse the text using spaCy
    doc = nlp(text)

    # Define the problem solving keywords
    problem_solving_keywords = ['solve', 'problem', 'issue', 'trouble', 'fix', 'repair', 
                                'resolve', 'correct', 'debug','analyze', 'assess', 'brainstorm', 'collaborate', 'combine',
                                'compare', 'contrast', 'critique', 'debug', 'define', 'design',
                                'develop', 'diagnose', 'evaluate', 'experiment', 'explore',
                                'generate', 'hypothesize', 'implement', 'improve', 'innovate',
                                'integrate', 'investigate', 'iterate', 'modify', 'optimize',
                                'overcome', 'plan', 'prioritize', 'prototype', 'refine',
                                'research', 'review', 'simulate', 'solve', 'strategize', 'streamline',
                                'study', 'synthesize', 'test', 'troubleshoot', 'understand',
                                'unify', 'validate', 'verify', 'visualize', 'weigh',
                                'workaround', 'wrench', 'zero in']

    # Extract the problem solving keywords from the text
    keywords = []
    for token in doc:
        if token.lemma_ in problem_solving_keywords:
            keywords.append(token.text)

    return keywords


def problem_solving_score(feedback):
    words = word_tokenize(feedback.lower())
    PSK = sum([1 for word in words if word in extract_probsol_keywords(feedback)])

    if PSK == 0:
        return 0.0
    elif PSK == 1:
        return 0.5
    else:
        return 1.0


# ----- Scoring Parameter 9: Collaboration Score (CLS) ------

def extract_collab_keywords(text):
    # Parse the text using spaCy
    doc = nlp(text)

    # Define the problem solving keywords
    collab_keywords = ['cooperation', 'coordination', 'partnerships', 'teamwork',
                       'networking', 'brainstorming', 'crowdsourcing', 'ideation',
                       'synergy', 'innovation', 'creativity', 'sharing', 'openness',
                       'communication', 'collaboration', 'leadership', 'problem-solving',
                       'cross-functional', 'innovative', 'empowerment', 'flexibility',
                       'team', 'community', 'culture', 'involvement', 'participation',
                       'transparency', 'diversity', 'inclusion', 'trust', 'support',
                       'adaptability', 'accountability', 'vision', 'sustainability',
                       'strategic', 'alliances', 'partners', 'disruption', 'influence',
                       'motivation', 'inspiration', 'learning', 'growth', 'change',
                       'customer-focused', 'digitalization', 'agility', 'experimentation']

    # Extract the problem solving keywords from the text
    keywords = []
    for token in doc:
        if token.lemma_ in collab_keywords:
            keywords.append(token.text)

    return keywords


def collaboration_score(feedback):
    words = word_tokenize(feedback.lower())
    CK = sum([1 for word in words if word in extract_probsol_keywords(feedback)])

    if CK == 0:
        return 0.0
    elif CK == 1:
        return 0.5
    else:
        return 1.0


# ----- Scoring Parameter 10: Crowdsourcing Score (CRS) -----

def extract_crowdsource_keywords(text):
    # Parse the text using spaCy
    doc = nlp(text)

    # Define the problem solving keywords
    crowdsource_keywords = ['open', 'external', 'wisdom', 'peer', 'collective', 'co-creation',
                            'distributed', 'idea', 'citizen', 'communities', 'collaboration',
                            'networks', 'crowdfunding', 'crowdvoting', 'outsourcing', 'volunteer',
                            'source', 'data', 'access', 'science', 'design', 'content',
                            'education', 'government', 'hardware', 'standards', 'platforms',
                            'software', 'hackathons', 'maker', 'feedback', 'testing',
                            'participation', 'marketing', 'analytics', 'engagement', 'development',
                            'intelligence', 'research', 'microtasking', 'smart', 'human',
                            'collective action', 'crowd wisdom', 'crowd innovation']

    # Extract the problem solving keywords from the text
    keywords = []
    for token in doc:
        if token.lemma_ in crowdsource_keywords:
            keywords.append(token.text)

    return keywords


def crowdsourcing_score(feedback):
    words = word_tokenize(feedback.lower())
    CSK = sum([1 for word in words if word in extract_probsol_keywords(feedback)])

    if CSK == 0:
        return 0.0
    elif CSK == 1:
        return 0.5
    else:
        return 1.0


# ----- Scoring Parameter 11: Length Score (LS) ------

# Option A
# Function that calculates the length score
def length_score(feedback):
    x = 10
    y = 100
    L = len(word_tokenize(feedback))
    if L < x or L > y:
        return 0
    else:
        return (L - x) / (y - x)


# ----- Scoring Parameter 12: Sentiment Score (SES) ------
def sentiment_score(feedback):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(feedback)
    return sentiment['compound']


# ----- Scoring Parameter 13: Weighting parameters (WP) ------

w_LS = 0.1
w_SS = 0.3
w_RS = 0.2
w_NS = 0.1
w_IQS = 0.1
w_PSS = 0.1
w_CS = 0.05
w_CSS = 0.05


# ----- Scoring Parameter 14: Overall Feedback Score (OFS) ------

def overall_feedback_score(feedback, w_LS, w_SS, w_RS, w_NS, w_IQS, w_PSS, w_CS, w_CSS):
    LS = length_score(feedback)
    SS = sentiment_score(feedback)
    RS = relevance_score(feedback)
    NS = novelty_score(feedback)
    PSS = problem_solving_score(feedback)
    CS = collaboration_score(feedback)
    CSS = crowdsourcing_score(feedback)

    return w_LS * float(LS) + w_SS * float(SS) + w_RS * float(RS) + w_NS * float(NS) + w_PSS * float(
        PSS) + w_CS * float(CS) + w_CSS * float(CSS)


# ----- Feedback Generation -----

def generate_feedback(ns, ws, rs, ss, cs, ts, pss, cbs, css, ls, sns, ofs):
    feedback_text = ""

    if ns < 0.2:
        feedback_text += "Your idea is not very original. Consider exploring more unique and creative ideas. "

    if ws < 0.5:
        feedback_text += "Your idea needs more work to make it practical and feasible. Consider refining it further. "

    if rs < 0.5:
        feedback_text += "Your idea may not be relevant to the task or problem at hand. Consider re-evaluating your approach. "

    if ss < 0.5:
        feedback_text += "Your idea lacks specificity and detail. Consider adding more information to make it more concrete. "

    if cs < 0.5:
        feedback_text += "Your idea is not very clear or easy to understand. Consider simplifying your language and structure. "

    if ts < 0.5:
        feedback_text += "Your idea may not be timely or relevant to the current situation. Consider addressing more urgent needs. "

    if pss == 0:
        feedback_text += "Your idea may benefit from more problem-solving keywords. Consider using language related to solving problems, troubleshooting, or debugging. "

    if cbs < 0.5:
        feedback_text += "Your idea could benefit from more collaboration. Consider seeking out input and feedback from others to refine your approach. "

    if css < 0.5:
        feedback_text += "Your idea may benefit from more input from a larger crowd. Consider reaching out to more people for feedback and suggestions. "

    if ls == 0:
        feedback_text += "Your idea may be too short or too long. Consider refining your idea to fit within the recommended length. "

    if sns < 0:
        feedback_text += "Your idea may come across as negative or unenthusiastic. Consider using more positive and engaging language. "

    if ofs < 0.5:
        feedback_text += "Overall, your idea could benefit from more refinement and improvement. Consider incorporating the above feedback to take your idea to the next level. "
    elif ofs < 0.7:
        feedback_text += "Overall, your idea is decent but could benefit from some additional work. Consider incorporating some of the above feedback to make it stronger. "
    elif ofs < 0.9:
        feedback_text += "Overall, your idea is strong and well-developed. Consider incorporating some of the above feedback to take it to the next level. "
    else:
        feedback_text += "Overall, your idea is exceptional and well-developed. Great work! "
#
    # Generate more detailed feedback using BERT
#    inputs = BERTtokenizer.encode_plus(
#        feedback_text,
#        add_special_tokens=True,
#        max_length=128,
#        padding='max_length',
#        return_tensors='pt'
#    )
#    outputs = BERTmodel(**inputs)
#    logits = outputs[0]
#    probabilities = logits.softmax(dim=1)
#    sentiment = 'positive' if probabilities[0][1] > probabilities[0][0] else 'negative'
#    feedback_text += f' The sentiment of this feedback is {sentiment}.'
#
#    return feedback_text


if st.button("Analyze your feedback on innovation ideas"):
    # Calculate the feedback scores
    ns = novelty_score(feedback)
    ws = workability_score(idea)
    rs = relevance_score(feedback)
    ss = specificity_score(feedback)
    cs = clarity_score(feedback)
    ts = timeliness_score(submission_date, feedback_date)
    pss = problem_solving_score(feedback)
    cbs = collaboration_score(feedback)
    css = crowdsourcing_score(feedback)
    ls = length_score(feedback)
    sns = sentiment_score(feedback)
    ofs = overall_feedback_score(feedback, w_LS, w_SS, w_RS, w_NS, w_IQS, w_PSS, w_CS, w_CSS)

    # Display the feedback scores
    st.write("Novelty score: ", ns)
    st.write("Workability score: ", ws)
    st.write("Relevance score: ", rs)
    st.write("Specificity score: ", ss)
    st.write("Clarity score: ", cs)
    st.write("Timeliness score: ", ts)
    st.write("Problem solving score: ", pss)
    st.write("Collaboration score: ", cbs)
    st.write("Crowdsourcing score: ", css)
    st.write("Length score: ", ls)
    st.write("Sentiment score: ", sns)
    st.write("Overall Feedback score: ", ofs)
    st.write("Feedback suggestion: ", generate_feedback(ns, ws, rs, ss, cs, ts, pss, cbs, css, ls, sns, ofs))

    # ----- Plotting the results and making a jpg file of it -----

    # Define the scores as a dictionary
    scores = {"Novelty": ns, "Workability": ws, "Relevance": rs, "Specificity": ss, "Clarity": cs}

    # Create a bar chart of the scores
    plt.bar(scores.keys(), scores.values())

    # Set the title and labels for the chart
    plt.title("Feedback Analysis")
    plt.xlabel("Score Type")
    plt.ylabel("Score")

    # Save the chart to a file
    plt.savefig("feedback_analysis.png")

    # Read the file as bytes
    with open("feedback_analysis.png", "rb") as f:
        bytes_data = f.read()

    # Encode the bytes as base64
    b64_data = base64.b64encode(bytes_data).decode()

    # Create a download button
    st.download_button(
        label="Download Feedback Analysis",
        data=b64_data,
        file_name="feedback_analysis.png",
        mime="image/png"
    )





    def create_table(data):
        # Extract column names and data
        column_names = list(data[0].keys())
        cell_data = [[row[column] for column in column_names] for row in data]
    
        # Create a Matplotlib table
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=cell_data, colLabels=column_names, loc='center', cellLoc='center')
    
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 1.5)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("w")
            if key[0] == 0:  # Header cells
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor('#8B008B')  # Set header background color
            else:
                cell.set_facecolor('#F0F0F0')  # Set other cells background color
    
        # Save the table to a buffer in PNG format
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
    
        return buf
    
    def main():
        st.title("Matplotlib Table in Streamlit")
        
        
        data = [
            {"Name": "Novelty", "Score": ns},
            {"Name": "Workability", "Score": ws},
            {"Name": "Relevance", "Score": rs},
            {"Name": "Specificity", "Score": ss},
            {"Name": "Clarity", "Score": cs},
            {"Name": "Timeliness", "Score": ts},
            {"Name": "Problem Solving", "Score": pss},
            {"Name": "Collaboration", "Score": cbs},
            {"Name": "Crowdsourcing", "Score": css},
            {"Name": "Length", "Score": ls},
            {"Name": "Sentiment", "Score": sns},
            {"Name": "Overall Feedback", "Score": ofs}
            
        ]
    
    
    
        # Create the table and display it in Streamlit
        table_buf = create_table(data)
        
        # Add a download button to save the table
        table_data = table_buf.getvalue()
        st.download_button(
            label="Download table as PNG",
            data=table_data,
            file_name="table.png",
            mime="image/png"
        )
        
        st.image(table_buf, caption='Scores Table', use_column_width=True)
        
    
        
    
    if __name__ == "__main__":
        main()

    




#---------------------------------------------------#
#              SUB SECTION                          #
#---------------------------------------------------#


# ----- Further comments ------
# check weighting parameters and scoring system
# Sometimes the first time you click the button, nothing shows up, 
# but if you click the button again. Then it shows up straight away. 
#


