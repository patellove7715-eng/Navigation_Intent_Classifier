import streamlit as st
import pandas as pd
import joblib
import re
import html
import pathlib
from html.parser import HTMLParser
from pathlib import Path


#page configuration

st.set_page_config(
    page_title="Navigation Intent Classification",
    page_icon="🧭",
    layout="centered"
)

#file loctions

BASE_DIR = pathlib.Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / 'nav_intent_model.pkl'
TFIdT_DIR = BASE_DIR / 'tfidf_vectorizer.pkl'

#text preprocessing
# text preprocessing
import re
import html


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text while discarding HTML markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text):
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return " ".join(parser.parts)


def text_preprocessing(query):
    # Convert missing values and other data types to string
    query = str(query)

    # Replace certain special characters with their string equivalents
    query = query.replace('%', ' percent')
    query = query.replace('$', ' dollar ')
    query = query.replace('₹', ' rupee ')
    query = query.replace('€', ' euro ')
    query = query.replace('@', ' at ')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    query = query.replace(',000,000,000 ', 'b ')
    query = query.replace(',000,000 ', 'm ')
    query = query.replace(',000 ', 'k ')
    query = re.sub(r'([0-9]+)000000000', r'\1b', query)
    query = re.sub(r'([0-9]+)000000', r'\1m', query)
    query = re.sub(r'([0-9]+)000', r'\1k', query)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    query_decontracted = []

    for word in query.split():
        if word in contractions:
            word = contractions[word]

        query_decontracted.append(word)

    query = ' '.join(query_decontracted)
    query = query.replace("'ve", " have")
    query = query.replace("n't", " not")
    query = query.replace("'re", " are")
    query = query.replace("'ll", " will")

    # Removing HTML tags
    query = strip_html(query)

    # Convert HTML entities into normal characters
    query = html.unescape(str(query)).lower().strip()

    query = query.replace("’", "'")

    # Convert text to lowercase
    query = query.lower()

    # Remove email addresses
    query = re.sub(r"\S+@\S+\.\S+", " ", query)

    # Remove URLs
    query = re.sub(r"https?://\S+|www\.\S+", " ", query)

    # Remove special characters
    query = re.sub(r"[^a-z0-9\s]", " ", query)

    # Remove extra spaces
    query = re.sub(r"\s+", " ", query).strip()

    return query

#load model and vectorizer


@st.cache_resource
def load_files():
    model = joblib.load(MODEL_DIR)
    vectorizer = joblib.load(TFIdT_DIR)

    return model, vectorizer
try :
    model, vectorizer = load_files()

except FileNotFoundError:
    st.error("Model files were not found. Put the model and "
        "TF-IDF files in the same folder as app.py")
    st.stop()

except Exception as error:

    st.error(f"unable to load model: {error}")
    st.stop()

# application heading


st.title("Navigation Intent Classifier")


st.write(
    "Enter a health-application request and the model "
    "will predict the correct navigation category."
)

# ---------------------------------
# User input
# ---------------------------------
user_query = st.text_area(
    "Enter your query",
    placeholder=(
        "Example: Where can I see all my health "
        "information at a glance?"
    ),
    height=140
)



# ---------------------------------
# Prediction
# ---------------------------------
# ---------------------------------
if st.button(
    "Predict Intent",
    type="primary",
    use_container_width=True
):

    cleaned_query = text_preprocessing(user_query)

    if cleaned_query == "":

        st.warning("Please enter a valid query.")

    else:

        # Convert query into TF-IDF features
        query_tfidf = vectorizer.transform(
            [cleaned_query]
        )

        # Predict category
        prediction = model.predict(
            query_tfidf
        )[0]

        # Get probability scores
        probabilities = model.predict_proba(
            query_tfidf
        )[0]

        # Find confidence
        confidence = probabilities.max() * 100

        # Display result
        st.success(
            f"Predicted category: {prediction}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        # Create probability table
        probability_df = pd.DataFrame({
            "Category": model.classes_,
            "Confidence (%)": probabilities * 100
        })

        probability_df = probability_df.sort_values(
            by="Confidence (%)",
            ascending=False
        ).reset_index(drop=True)

        st.subheader("Top predictions")

        st.dataframe(
            probability_df.head(5),
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            probability_df.set_index("Category")[
                "Confidence (%)"
            ]
        )
        # ---------------------------------
        # Available categories
        # ---------------------------------
    # ---------------------------------
    # Available categories
    # ---------------------------------
    with st.expander("Available categories"):

        categories = [
            "AI_COACH",
            "AI_INSIGHTS",
            "DAILY_PROGRESS",
            "DASHBOARD",
            "DEVICES",
            "DIET_PLAN",
            "HEALTH_TIMELINE",
            "MANUAL_ENTRY",
            "MY_PLANS",
            "PROFILE",
            "WORKOUT_PLAN"
        ]

        for category in categories:
            st.write("•", category)
