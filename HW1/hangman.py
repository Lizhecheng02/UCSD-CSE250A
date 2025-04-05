"""
CSE 150A/250A SP 25 HW 1

This file is meant to give you a template for implmenting hangman in question 1.6. You are not required to use the same implementation in your solution.

To run, call "python hangman.py".
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate


def compute_prior(word_counts):
    """
    TODO
    Computes the prior probability for every word in the corpus. In other words, computing P(W=w).

    Args:
        word_counts (pd.DataFrame): DataFrame containing words and their counts

    Returns:
        pd.Series: Prior probabilities for all words in the corpus
    """
    return


def get_prior(word, word_counts):
    """
    TODO
    Gets the prior probability for a given word from the dataframe.

    Args:
        word (str): Word to get prior probability for.
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.

    Returns:
        float: Prior probability for a given word.
    """
    return


def check_letter(l, w):
    """
    TODO
    Checks if the given letter is in the given word or P(Li = l for some i in {1,2,3,4,5} | W).

    Args:
        l (str): Letter of interest.
        w (str): Word to check letter in.
    Returns:
        bool: True if l is in w, False otherwise.
    """
    return


def check_evidence(evidence, w):
    """
    TODO
    Checks if it's possible that the evidence supports the given word. In other words, its trying to compute P(E|W=w).

    Evidence is a tuple containing two strings, first is the guessed word so far with correct letters 
    and the second being all incorrect letters.

    Args:
        evidence (tuple): A tuple containing two strings
        w (str): Word to be checked

    Returns:
        bool: True if its possible, False otherwise.
    """
    return


def compute_posterior_denominator(evidence, word_counts):
    """
    TODO
    Computes the probability of the evidence occurring or P(E). This is used in the calculation of the posterior probability.

    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.

    Returns:
        float: Probability of evidence.
    """
    return


def compute_posterior(evidence, word, word_counts, denominator):
    """
    TODO
    Computes the posterior probability or P(W=w|E).

    Should be computing the denominator separately for faster runtime.

    Args:
        evidence (tuple): A tuple containing two strings
        word (str): A given word to compute posterior for.
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        Denominator (float): Denominator of the posterior probability computed earlier

    Returns:
        float: Probability of evidence.
    """
    return


def predictive_probability(evidence, word_counts, denominator):
    """
    TODO
    Computes the probability for each letter being in the word given the evidence or P(Li = l for some i in {1,2,3,4,5} | E)

    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        denominator (float): Denominator of posterior probability.

    Returns:
        lst: A list of probabilities for each letter.
    """
    return


def predict_character(evidence, word_counts, denominator):
    """
    TODO
    Generates the prediction for the next best guess with the associated probability.

    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        denominator (float): Denominator of the posterior probability.

    Returns:
        tuple: Predicted character and associated probability.
    """
    return


if __name__ == "__main__":

    # TODO: Fill in correct file path
    file_path = ""

    empty_word = "-----"
    Evidence = [(empty_word, ""),
                (empty_word, "EA"),
                ("A---S", ""),
                ("A---S", "I"),
                ("--O--", "AEMNT"),
                (empty_word, "EO"),
                ("D--I-", ""),
                ("D--I-", "A"),
                ("-U---", "AEIOS")
                ]

    ### DO NOT MODIFY BELOW THIS LINE ###
    assert os.path.exists(file_path), f"File not found: {file_path}"

    word_counts = pd.read_csv(file_path, header=None, sep=' ')
    word_counts = word_counts.rename(columns={0: 'Word', 1: 'Count'})

    word_counts['Prior'] = compute_prior(word_counts)

    output = []
    pbar = tqdm(Evidence)
    for e in (Evidence):
        corr, incorr = e
        incorr = "{" + incorr + "}"
        pbar.set_description(f"Processing Evidence: '{e}'")
        char, prob = predict_character(e, word_counts, compute_posterior_denominator(e, word_counts))
        output += [(corr, incorr, char, prob)]
        pbar.update(1)

    output = pd.DataFrame(output, columns=["Correctly Guessed", "Incorrectly Guessed", "Character", "Probability"])
    print(print(tabulate(output, headers='keys', tablefmt='psql')))
