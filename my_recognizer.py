import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for idx in range(test_set.num_items):
        # Get the item X lengths for hmmlearn
        X, lengths = test_set.get_item_Xlengths(idx)
        log_l = {}

        for word, model in models.items():
            try:
                log_l[word] = model.score(X, lengths)
            except:
                log_l[word] = float('-inf')

        probabilities.append(log_l)
        # Get the best guess
        best_guess = max(log_l, key=log_l.get)
        guesses.append(best_guess)

    return probabilities, guesses
