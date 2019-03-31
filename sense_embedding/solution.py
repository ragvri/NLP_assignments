from nltk.corpus import wordnet
import nltk
import re
from nltk.corpus import stopwords
import fastText
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_cosine_similiarity(a, b):
    len_a, len_b = 0, 0
    dot_prod = 0

    assert len(a) == len(b)
    for i in range(len(a)):
        dot_prod += (a[i]*b[i])
        len_a += (a[i]**2)
        len_b += (b[i]**2)
    return dot_prod/(math.sqrt(len_a)*math.sqrt(len_b))


def get_sense_dictionary(word):
    # remove the stop words to get the content words
    stop_words = set(stopwords.words('english'))

    syns_list = wordnet.synsets(word)

    sense_dictionary = {}
    for syns in syns_list:
        word_sense = syns.lemmas()[0].name()
        gloss = syns.definition()
        gloss = re.sub(r'[^\w\s]', '', gloss).split()
        gloss = [w for w in gloss if w not in stop_words]
        try:
            example_sentence = syns.examples()[0]
            example_sentence = re.sub(r'[^\w\s]', '', example_sentence).split()
            example_sentence = [
                w for w in example_sentence if w not in stop_words]
        except IndexError:
            continue

        hypernym_synsets = syns.hypernyms()
        hypernym_words = []
        for s in hypernym_synsets:
            word = s.lemmas()[0].name()
            if word not in stop_words:
                hypernym_words.append(word)

        hyponym_synsets = syns.hyponyms()
        hyponym_words = []
        for s in hyponym_synsets:
            word = s.lemmas()[0].name()
            if word not in stop_words:
                hyponym_words.append(word)
        sense_bag = []
        for l in [gloss, example_sentence, hypernym_words, hyponym_words]:
            sense_bag.extend(l)

        sense_dictionary[word_sense] = sense_bag
    return sense_dictionary


def get_sense_embedding(sense_dictionary, ft):

    wv_dim = ft.get_dimension()
    senses_embedding = {}
    for senses in sense_dictionary.keys():
        avg_wv = np.zeros(wv_dim)
        words_list = sense_dictionary[senses]

        for word in words_list:
            wv = ft.get_word_vector(word.lower())
            avg_wv = avg_wv + wv
        avg_wv = avg_wv/len(words_list)
        senses_embedding[senses] = avg_wv
    return senses_embedding


ft = fastText.load_model('D:\\softwares\\fasttext\\wiki.en.bin')


word = input("Enter the word to do")
sense_dictionary = get_sense_dictionary(word)
senses_embedding = get_sense_embedding(sense_dictionary, ft)
word_embedding = ft.get_word_vector(word.lower())

best_cosine = 0
for sense in senses_embedding:
    sense_embedding = senses_embedding[sense]
    cosine_score = get_cosine_similiarity(sense_embedding, word_embedding)
    if cosine_score > best_cosine:
        best_cosine = cosine_score
        best_sense = sense

print(f'Best sense of {word} is {best_sense} with cosine score: {best_cosine}')

pca = PCA(n_components=2)
X = []
labels = []
for sense in senses_embedding:
    labels.append(sense)
    X.append(senses_embedding[sense])
labels.append(word)
X.append(word_embedding)

X_new = pca.fit_transform(X)

x_separate = []
y_separate = []

for i in X_new:
    x_separate.append(i[0])

for i in X_new:
    y_separate.append(i[1])

plt.scatter(x_separate, y_separate)
for i, txt in enumerate(labels):
    plt.annotate(txt, (x_separate[i], y_separate[i]))
plt.show()
