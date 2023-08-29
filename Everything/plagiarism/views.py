from django.shortcuts import render
from .forms import CheckForm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize text into words
    words = nltk.word_tokenize(text)

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

# Create your views here.
def index(request):
    results = None
    if request.method == 'POST':
        form = CheckForm(request.POST)
        if form.is_valid():
            bio_1 = form.cleaned_data['bio_1']
            bio_2 = form.cleaned_data['bio_2']
            processed_text_1 = preprocess_text(bio_1)
            processed_text_2 = preprocess_text(bio_2)
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([processed_text_1, processed_text_2])
            cosine_sim = cosine_similarity(X[0], X[1])
            threshold = 0.5
            if cosine_sim[0][0] > threshold:
                results = 'plagiarised'
            else:
                results = 'non plagiarised'
    else:
        form = CheckForm()
    return render(request, 'index.html',{'form': form, 'results': results})