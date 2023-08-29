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


data = [("So much of modern-day life revolves around using opposable thumbs, from holding a hammer to build a home to ordering food delivery on our smartphones. But for our ancestors, the uses were much simpler. Strong and nimble thumbs meant that they could better create and wield tools, stones and bones for killing large animals for food", "A lot of life today involves using opposable thumbs, from using a hammer to build a house to ordering something on our smartphones. But for our predecessors, the uses were much more simple. Powerful and dexterous thumbs meant that they could better make and use tools, stones and bones for killing large animals to eat."),
        ("Ancient Sparta has been held up for the last two and a half millennia as the unmatched warrior city-state, where every male was raised from infancy to fight to the death. This view, as ingrained as it is alluring, is almost entirely false",
         "For the last 2,500 years, Ancient Sparta has been considered the unmatched warrior city-state in popular imagination. The idea that every male was raised from infancy to fight to the death, as ingrained as it is alluring, is actually not true"),
        ("For many Americans, the eagle feather headdress is a generic symbol of Native America indivisible from the narrative of the wild west and cowboys and Indians",
         "For many Americans, the headdress is a well-known symbol of indigenous America indistinguishable from the narrative of the wild west and cowboys and Indians"),
        ("Americans have always remembered the battle. What we often forget are the difficult decisions tribal leaders made afterward to ensure the safety of their people . Under skies darkened by smoke, gunfire and flying arrows, 210 men of the U.S. Army's 7th Cavalry Unit led by Lt. Colonel George Custer confronted thousands of Lakota Sioux and Northern Cheyenne warriors on June 25, 1876, near the Little Big Horn River in present-day Montana. The engagement was one in a series of battles and negotiations between Plains Indians and U.S. forces over control of Western territory, collectively known as the Sioux Wars",
         "On June 25, 1876, 210 men of the U.S. Army's 7th Cavalry Unit led by Lt. Colonel George Custer confronted thousands of Lakota Sioux and Northern Cheyenne warriors. Custer and his men were handily defeated, and Americans have always remembered the battle as “Custer’s Last Stand.” What is often forgotten is the difficult decisions tribal leaders made afterward to ensure the safety of their people")]


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


for d in data:
    processed_text_1 = preprocess_text(d[0])
    processed_text_2 = preprocess_text(d[1])
# print("processed_1 Text:", processed_text_1)
# print("Processed_2 Text:", processed_text_2)
# words = nltk.word_tokenize(processed_text)
# trigrams = list(nltk.trigrams(words))
# for trigram in trigrams:
#     print(trigram)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([processed_text_1, processed_text_2])
    cosine_sim = cosine_similarity(X[0], X[1])
    print(cosine_sim[0][0])
# create a threshold
    threshold = 0.5
    if cosine_sim[0][0] > threshold:
        print("Plagiarised")
    else:
        print("Not plagiarised")
