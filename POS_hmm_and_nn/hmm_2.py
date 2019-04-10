from sklearn.model_selection import KFold
import numpy
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import math

# computing the intial probabilities for each state
def compute_start_probability():

    for state in states:
        start_p[state] = (start_p.get(state, 0) + 1) / (len(X_train) + 1)


#computing transition probabilites for given tuple of states
def compute_transition_probability(state, last_state, pre_last_state):

    trans_p_1[(state, last_state)] = trans_p_1.get((state, last_state), default_trans_const)
    trans_p_1[(last_state, pre_last_state)] = trans_p_1.get((last_state, pre_last_state), default_trans_const)
    trans_p_2[(state, last_state, pre_last_state)] = trans_p_2.get((state, last_state, pre_last_state), default_trans_const)

    n1 = tag_frequency[state]
    n2 = trans_p_1[(state, last_state)]
    n3 = trans_p_2[(state, last_state, pre_last_state)]
    c0 = len(states)
    c1 = tag_frequency[last_state]
    c2 = trans_p_1[(state, last_state)]
    k1 = n1
    k2 = (math.log(n2) + 1) / (math.log(n2) + 2)
    k3 = (math.log(n3) + 1) / (math.log(n3) + 2)
    return k3*n3/c2 + (1-k3)*k2*n2/c1 + (1-k3)*(1-k2)*k1/c0

#computing emission probabilities from each state to word
def compute_emission_probability():

    for word in words:
        for state in states:
            if (word, state) not in emit_p:
                emit_p[(word, state)] = default_emit_prob
            else:
                emit_p[(word, state)] = emit_p[(word, state)] / state_frequency[state]


def viterbi(query):

    dp_prob = {}
    dp_state = {}
    # calculating dp values for initial first word of the query
    for state in states:
        emit_p[(query[0], state)] = emit_p.get((query[0], state), default_emit_prob) 
        for prev_state in states:
            dp_prob[(state, prev_state)] = 1 / len(X_train)
            dp_state[(state, prev_state)] = [state]
        dp_prob[(state, state)] = start_p[state] * emit_p[(query[0], state)]
        dp_state[(state, state)] = [state]

    for i in range(1, len(query)):
        new_dp_prob = {}
        new_dp_state = {}
        # finding the probability of occurence for this current state
        for cur_state in states:
            emit_p[(query[i], cur_state)] =  emit_p.get((query[i], cur_state), default_emit_prob)
            # checking for all previous states, and finding the state for which the given probability will be maximum
            for last_state in states:
                max_probability = 0
                max_probability_state = cur_state
                for pre_last_state in states:
                    probability = dp_prob[(last_state, pre_last_state)] * compute_transition_probability(state, last_state, pre_last_state) 
                    if max_probability < probability:
                        max_probability = probability
                        max_probability_state = pre_last_state

                # assigning the maximum probability and sequence of states to the current state, last_state pair
                new_dp_prob[(cur_state, last_state)] = max_probability * emit_p[(query[i], cur_state)]
                new_dp_state[(cur_state, last_state)] = dp_state[(last_state, max_probability_state)] + [cur_state]

        dp_prob = new_dp_prob
        dp_state = new_dp_state

    max_probability = 0
    hidden_state = []
    for state in states:
        for last_state in states:
            if dp_prob[(state, last_state)] > max_probability:
                max_probability = dp_prob[(state, last_state)]
                hidden_state = dp_state[(state, last_state)]
    return hidden_state


if __name__ == '__main__':
    X = []
    default_emit_prob = 1e-8    # default emission probability
    default_trans_const = 1      # default transition constant
    
    # converting dataset into a list of sentences
    sentence = []
    with open("Brown_train.txt", 'r') as f:
        for line in f:
            sentence = []
            words = line.strip('\n').split(" ")
            for word in words:
                word_pos = word.split("/")
                if len(word_pos) == 1:
                    continue
                pos_tag = word_pos.pop()
                sentence.append(("/".join(word_pos), pos_tag))
            X.append(sentence)
    x = numpy.array(X)
    kf = KFold(n_splits = 3, random_state = None, shuffle = True)
    count = 0 # for producing the output into different files in the kfold validation
    # K-Fold validation
    for train_index, test_index in kf.split(x):
        filename = f'./hmm_predict_fold{count}.txt'
        count += 1
        print(count)

        X_train, X_test = x[train_index], x[test_index]
        trans_p_1 = {}
        trans_p_2 = {}
        emit_p = {}
        start_p = {}
        tags = set()
        tag_frequency = {}
        words = set()

        # computing the required frequency of words, tags(= hidden states) for computing required probabilities(i.e. emission, start, transition)
        for given_sentence in X_train:
            last_tag = ''
            pre_last_tag = ''
            is_start_index = 0
            for word_tag in given_sentence:
                if len(word_tag) == 1:
                    continue

                if is_start_index >= 2:
                    trans_p_2[(word_tag[1], last_tag, pre_last_tag)] = trans_p_1.get((word_tag[1], last_tag, pre_last_tag), 0) + 1
                
                if is_start_index >= 1:
                    trans_p_1[(word_tag[1], last_tag)] = trans_p_1.get((word_tag[1], last_tag), 0) + 1
                    is_start_index = 2

                if is_start_index == 0:
                    start_p[word_tag[1]] = start_p.get(word_tag[1], 0) + 1
                    is_start_index = 1

                pre_last_tag = last_tag
                last_tag = word_tag[1]

                tag_frequency[word_tag[1]] = tag_frequency.get(word_tag[1], 0) + 1
                emit_p[(word_tag[0], word_tag[1])] = emit_p.get((word_tag[0], word_tag[1]), 0) + 1
                words.add(word_tag[0])
                tags.add(word_tag[1])

        states = list(tags)
        words = list(words)
        state_frequency = tag_frequency
        # computing the required initial probabilities
        compute_start_probability()
        compute_emission_probability()

        # clearing the previous file contents
        with open(filename,'w') as f:
            pass

        expected_labels = []
        predicted_labels = []
        f1_total = 0
        acc_total = 0 
        # applying viterbi for the test samples
        with open(filename,'a') as f:
            for given_sentence in X_test:
                query = []
                expected_sequence = []
                for word_tag in given_sentence:
                    query.append(word_tag[0])
                    expected_sequence.append(word_tag[1])
                hidden_state_sequence = viterbi(query)

                # storing the test data and generated ouput in the output file
                to_write = ""
                for i, hidden_state in enumerate(hidden_state_sequence):
                    to_write += f'{query[i]}\t{hidden_state}\t{expected_sequence[i]}\n'
                to_write += f'\n'
                f.write(to_write)

                expected_labels.extend(expected_sequence)
                predicted_labels.extend(hidden_state_sequence)

        acc = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels, average='macro')
        print("Accuracy: ", acc,"\nF1 score: ", f1,"\n\n", sep=" ")
        f1_total += f1
        acc_total += acc

    acc_total /= 3
    f1_total /= 3
    print("Average Accuracy: ", acc_total)
    print("Average F1_Score: ", f1_total)

