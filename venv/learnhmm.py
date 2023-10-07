import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

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
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()

    # Initialize the initial, emission, and transition matrices
    initial = np.zeros((len(tags_to_index), 1))
    emission = np.zeros((len(tags_to_index), len(words_to_index)))
    transition = np.zeros((len(tags_to_index), len(tags_to_index)))

    # Increment the matrices
    for i in range(10000):
        tag = None
        for j in train_data[i]:
            emission[tags_to_index.get(j[1]), words_to_index.get(j[0])] += 1
            if tag is not None:
                transition[tags_to_index.get(tag), tags_to_index.get(j[1])] += 1
            else:
                initial[tags_to_index.get(j[1]), 0] += 1
            tag = j[1]

    # Add a pseudocount
    initial = initial + 1
    emission = emission + 1
    transition = transition + 1

    initial = initial / np.sum(initial)
    emission = emission / np.sum(emission, axis=1)[:, None]
    transition = transition / np.sum(transition, axis=1)[:, None]
    # Save your matrices to the output files --- the reference solution uses
    # np.savetxt (specify delimiter="\t" for the matrices)
    np.savetxt(init_out, initial)
    np.savetxt(emit_out, emission)
    np.savetxt(trans_out, transition)
