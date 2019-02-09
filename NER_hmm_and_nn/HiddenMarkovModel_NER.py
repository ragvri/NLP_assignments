from sklearn.model_selection import KFold
import numpy

# computing the intial probabilities for each state/tag
def compute_start_probability():

    for state in states:
        start_p[state] = (sentence_start_with_tag.get(state, 0) + 1) / (len(X_train) + 1)


#computing transition probabilites of states
def compute_transition_probability():

    for next_state in states:
        for prev_state in states:
            trans_p[(next_state, prev_state)] = trans_p.get((next_state, prev_state), default_trans_prob) / tag_frequency[prev_state]


#computing emission probabilities from each state to word
def compute_emission_probability():

    for word in words:
        for state in states:
            if (word, state) not in emit_p:
                emit_p[(word, state)] = default_emit_prob
            else:
                emit_p[(word, state)] = emit_p[(word, state)] / tag_frequency[state]


def viterbi(start_p, emit_p, trans_p, query):

    dp_prob = {}
    dp_state = {}
    # calculating dp values for initial word, tag
    for state in states:
        emit_p[(query[0], state)] = emit_p.get((query[0], state), default_emit_prob) 
        dp_prob[state] = start_p[state] * emit_p[(query[0], state)]
        dp_state[state] = [state]

    for i in range(1, len(query)):
        new_dp_prob = {}
        new_dp_state = {}
        # finding the probability of occurence for this current state
        for cur_state in states:
            max_probability = 0
            max_probability_state = cur_state
            # checking for all previous states, and finding the state for which the given probability will be maximum
            for prev_state in states:
                trans_p[(cur_state, prev_state)] = trans_p.get((cur_state, prev_state), default_trans_prob)
                probability = dp_prob[prev_state] * trans_p[(cur_state, prev_state)]
                if max_probability < probability:
                    max_probability = probability
                    max_probability_state = prev_state

            emit_p[(query[i], cur_state)] =  emit_p.get((query[i], cur_state), default_emit_prob)
            # assigning the maximum probability and sequence of states to the current state
            new_dp_prob[cur_state] = max_probability * emit_p[(query[i], cur_state)]
            new_dp_state[cur_state] = dp_state[max_probability_state] + [cur_state]

        dp_prob = new_dp_prob
        dp_state = new_dp_state

    max_probability = 0
    hidden_state = []
    for state in states:
        if dp_prob[state] > max_probability:
            max_probability = dp_prob[state]
            hidden_state = dp_state[state]
    return hidden_state


if __name__ == '__main__':
    X = []
    default_emit_prob = 1e-8    # default emission probability
    default_trans_prob = 0      # default transition probability
    
    # converting dataset into a list of sentences
    sentence = []
    with open("CS563-NER-Dataset-10Types.txt", 'r') as f:
        for line in f:
            word_tag = line.strip('\n').split("\t")
            if len(word_tag) == 1:
                X.append(sentence)
                sentence = []
                continue
            sentence.append((word_tag[0], word_tag[1]))
        if len(sentence) != 0:
            X.append(sentence)

    x = numpy.array(X)
    kf = KFold(n_splits = 3, random_state = None, shuffle = True)
    count = 0 # for producing the output into different files in the kfold validation
    # K-Fold validation
    for train_index, test_index in kf.split(x):
        filename = f'./hmm_fine{count}.txt'
        count += 1
        print(count)

        X_train, X_test = x[train_index], x[test_index]
        trans_p = {}
        emit_p = {}
        states = set()
        sentence_start_with_tag = {}
        tag_frequency = {}
        words = set()

        # computing the required frequency of words, tags for computing required probabilities(i.e. emission, start, transition)
        for given_sentence in X_train:
            last_tag = ''
            is_start_index = 1
            for pair_of_words in given_sentence:
                if len(pair_of_words) == 1:
                    continue
                if is_start_index == 1:
                    sentence_start_with_tag[pair_of_words[1]] = sentence_start_with_tag.get(pair_of_words[1], 0) + 1
                    is_start_index = 0
                else:
                    trans_p[(pair_of_words[1], last_tag)] = trans_p.get((pair_of_words[1], last_tag), 0) + 1

                last_tag = pair_of_words[1]
                tag_frequency[pair_of_words[1]] = tag_frequency.get(pair_of_words[1], 0) + 1
                emit_p[(pair_of_words[0], pair_of_words[1])] = emit_p.get((pair_of_words[0], pair_of_words[1]), 0) + 1
                words.add(pair_of_words[0])
                states.add(pair_of_words[1])

        states = list(states)
        start_p = {}
        # computing the required initial probabilities
        compute_start_probability()
        compute_transition_probability()
        compute_emission_probability()

        # clearing the previous file contents
        with open(filename,'w') as f:
            pass

        # applying viterbi for the test samples
        for given_sentence in X_test:
            query = []
            expected_sequence = []
            for pair_of_words in given_sentence:
                query.append(pair_of_words[0])
                expected_sequence.append(pair_of_words[1])
            hidden_state_sequence = viterbi(start_p, emit_p, trans_p, query)

            # storing the test data and generated ouput in the output file
            for i, hidden_state in enumerate(hidden_state_sequence):
                with open(filename,'a') as f:
                    to_write = f'{query[i]}\t{hidden_state}\t{expected_sequence[i]}\n'
                    f.write(to_write)

            with open(filename,'a') as f:
                to_write = f'\n'
                f.write(to_write)
