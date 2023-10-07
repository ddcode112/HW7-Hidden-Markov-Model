import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, train_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(H):
    if H.ndim == 1:
        a = np.max(H)
        return np.log(np.sum(np.exp(H - a))) + a
    else:
        a = np.amax(H, axis=1)
        return np.log(np.sum(np.exp(H - np.reshape(a, (len(a), 1))), axis=1)) + a

def forwardbackward(seq, loginit, logtrans, logemit):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)
    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((M, L))
    for t in range(L):
        if t == 0:
            log_alpha[:, t] = loginit + logemit[:, words_to_indices.get(seq[t, 0])]
        else:
            log_alpha[:, t] = logemit[:, words_to_indices.get(seq[t, 0])] + logsumexp(log_alpha[:, t - 1] + logtrans.T)
    # Initialize log_beta and fill it in
    log_beta = np.zeros((M, L))
    for t in range(L - 1, -1, -1):
        if t == L - 1:
            log_beta[:, t] = np.log(1)
        else:
            log_beta[:, t] = logsumexp(logemit[:, words_to_indices.get(seq[t + 1, 0])] + log_beta[:, t + 1] + logtrans)
    # Compute the predicted tags for the sequence
    predicted_arg = np.argmax(log_alpha + log_beta, axis=0)
    predicted_tags = np.array([j for i in predicted_arg for j in tags_to_indices if tags_to_indices[j] == i])
    # Compute the log-probability of the sequence
    p_x = logsumexp(log_alpha[:, -1])
    # Return the predicted tags and the log-probability
    return predicted_tags, p_x
    

    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, train_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    # For each sequence, run forward_backward to get the pfredicted tags and
    # the log-probability of that sequence.
    # log_likelihood = np.zeros(len(validation_data))
    log_likelihood = np.zeros(len(validation_data))
    train_log_likelihood = np.zeros(len(train_data))
    right = 0
    total = 0
    validation_result = []
    for i in range(len(validation_data)):
        seq = np.array(validation_data[i])
        str_len = len(validation_data[i])
        predicted_tags, log_probability = forwardbackward(seq, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit))
        temp = np.array(validation_data[i])
        temp[:, 1] = predicted_tags
        temp = temp.tolist()
        validation_result.append(temp)
        acc = np.array([1 if seq[:, 1][i] == predicted_tags[i] else 0 for i in range(str_len)])
        right += np.sum(acc)
        total += acc.size
        log_likelihood[i] = log_probability

    for i in range(len(train_data)):
        train_seq = np.array(train_data[i])
        train_predicted_tags, train_log_probability = forwardbackward(train_seq, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit))
        train_log_likelihood[i] = train_log_probability
    # Compute the average log-likelihood and the accuracy. The average log-likelihood
    # is just the average of the log-likelihood over all sequences. The accuracy is
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    avg_log_likelihood = np.average(log_likelihood)
    avg_train_log_likelihood = np.average(train_log_likelihood)
    avg_accuracy = right/total

    with open(metric_file, "w") as m:
        m.write("Average Log-Likelihood: {}\n".format(avg_log_likelihood))
        m.write("Average Train Log-Likelihood: {}\n".format(avg_train_log_likelihood))
        m.write("Accuracy: {}".format(avg_accuracy))

    with open(predicted_file, "w") as p:
        for i in range(len(validation_result)):
            for j in range(len(validation_result[i])):
                p.write("{}\t{}\n".format(validation_result[i][j][0], validation_result[i][j][1]))
            p.write("\n")
