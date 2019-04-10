import numpy as np
import fastText
from keras.utils import to_categorical
from keras.layers import *
from keras import Model, Sequential
from sklearn.model_selection import StratifiedKFold
from setup import Metrics
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report


def get_sentences(filename):
    sentences = []
    sentence = []
    label2Id = {}
    no_words = 0
    with open(filename) as f:
        for line in f:
            sentence = []
            word_list = line.split()
            for w in word_list:
                word = w.split('/')[0]
                label = w.split('/')[1]
                no_words += 1
                if label not in label2Id:
                    label2Id[label] = len(label2Id)
                sentence.append([word, label])
            sentences.append(sentence)

    return sentences, label2Id, no_words


def preprocessing(filename, context_size):
    # returns word, label pair
    sentences, label2Id, no_words = get_sentences(filename)
    print(f'Loading fasttext ..')
    ft = fastText.load_model('D:\\raghav\\cc.en.300.bin')
    dimension_size = ft.get_dimension()
    print('Done!')
    X = []
    y = []
    word_ft = {}
    for sentence in sentences:
        temp_sentence = [word for word, label in sentence]
        temp_sentence.insert(0, '0')
        temp_sentence.insert(0, '0')
        temp_sentence.insert(len(temp_sentence), '0')
        temp_sentence.insert(len(temp_sentence), '0')

        temp_features = []
        for word in temp_sentence:
            word = str(word).lower()
            if word not in word_ft:
                wv = ft.get_word_vector(word)
                word_ft[word] = wv
            else:
                wv = word_ft[word]
            temp_features.append(wv)
        for i, x in enumerate(sentence):
            X.append(temp_features[i:i+5])
            label = label2Id[x[1]]
            y.append(label)
    return np.array(X), np.array(y), label2Id, dimension_size


def create_model(no_labels, context_size, embedding_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(context_size, embedding_dim)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])
    model.summary()
    return model


def change_to_categorical(y, no_labels):
    temp = []
    for label in y:
        categorical_label = to_categorical(label, no_labels)
        temp.append(categorical_label)
    return np.array(temp)


if __name__ == "__main__":
    logs_file = './logs_pos.txt'
    weights_file = './nn_pos.h5'
    dataset = './Brown_train.txt'
    context_size = 5
    X, y, label2Id, embedding_dim = preprocessing(dataset, context_size)
    print(f'X size: {X.shape} y: {y.shape}')
    print(f'Label2ID:{label2Id}')
    kfold = StratifiedKFold(n_splits=3)
    f = open(logs_file, 'w')
    f.close()
    f = open(weights_file, 'w')
    f.close()
    metrics = Metrics(weights_file, True, logs_file)
    for train_index, test_index in kfold.split(X, y):
        model = create_model(len(label2Id), context_size, embedding_dim)
        X_train, y_train = X[train_index], y[train_index]
        y_train = change_to_categorical(y_train, len(label2Id))
        X_test, y_test = X[test_index], y[test_index]
        y_test = change_to_categorical(y_test, len(label2Id))

        print(
            f'X_train:{X_train.shape} X_test:{X_test.shape} y_train:{y_train.shape} y_test:{y_test.shape}')
        model.fit(X_train, y_train, epochs=20, verbose=1,
                  validation_data=(X_test, y_test), callbacks=[metrics])
        # y_pred = model.predict(X_test)
