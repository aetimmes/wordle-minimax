#!/usr/bin/env python3.10
"""
Wordle solver.

Written by Andrew Timmes (andrew.timmes@gmail.com).
"""

import argparse
import asyncio
import functools
import logging
import multiprocessing
from collections import defaultdict
from multiprocessing.pool import ApplyResult
from typing import Any, Dict, List, Tuple

ANSWERS_FILENAME = "answers.txt"
GUESSES_FILENAME = "guesses.txt"
LOGFILE = "output.log"
logging.basicConfig(filename=LOGFILE)


def main() -> None:
    """
    Compute best guesses.

    Parse arguments, reads in legal guesses/answers, and then computes the best guesses
    to use given the list of legal guesses and pre-ordained guesses.
    """
    argparser = argparse.ArgumentParser(description="Minimax solver for Wordle")
    argparser.add_argument(
        "guesses",
        nargs="*",
        default=[],
        help="List of guesses to force solver to use before reverting to minimax, in \
            order",
    )
    argparser.add_argument(
        "-a",
        "--answers-file",
        action="store",
        default=ANSWERS_FILENAME,
        help="File containing legal answers",
    )
    argparser.add_argument(
        "-g",
        "--guesses-file",
        action="store",
        default=GUESSES_FILENAME,
        help="File containing legal guesses",
    )
    args = argparser.parse_args()
    with open(args.answers_file, mode="r", encoding="utf-8") as answer_file:
        legal_answers = [x.strip() for x in answer_file.readlines()]
    with open(args.guesses_file, mode="r", encoding="utf-8") as guess_file:
        legal_guesses = [x.strip() for x in guess_file.readlines()] + legal_answers

    while len(legal_answers) > 1:
        if args.guesses:
            guesses = [args.guesses.pop(0)]
        else:
            guesses = legal_guesses
        guess_candidates = get_candidates(guesses, legal_answers)
        gc_cardinalities = [
            (
                guess,
                get_cardinalities(candidates),
            )
            for guess, candidates in guess_candidates.items()
        ]
        gc_cardinalities.sort(key=lambda t: t[1])

        best_word = gc_cardinalities[0][0]
        hardest_response = gc_cardinalities[0][1][0][0]
        legal_answers = guess_candidates[best_word][hardest_response]

        print(best_word, hardest_response, len(legal_answers))


def get_candidates(
    legal_guesses: List[str], legal_answers: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Compute candidates from lists of legal guesses and answers.

    Args:
        legal_guesses (List[str]): list of legal guesses
        legal_answers (List[str]): list of legal answers

    Returns:
        Dict[str, Dict[str, List[str]]]: mapping of result to candidate list per-guess
    """
    results: Dict[str, Dict[str, List[str]]] = {}
    with multiprocessing.Pool() as pool:
        tasks: List[ApplyResult[Any]] = []
        for guess in legal_guesses:
            tasks.append(
                pool.apply_async(
                    get_hardest_response,
                    (
                        guess,
                        legal_answers,
                    ),
                )
            )
        for task in tasks:
            guess, candidates = task.get()
            results[guess] = candidates
    return results


def get_cardinalities(candidates: Dict[str, List[str]]) -> List[Tuple[str, int]]:
    """
    Compute the size of candidate sets per-guess per-result.

    Args:
        candidates (Dict[str, List[str]]): mapping of result to candidate list per-guess

    Returns:
        List[Tuple[str, int]]: pairings of guess outputs and number of words
    """
    result: List[Tuple[str, int]] = []
    for candidate in candidates:
        result.append(
            (
                candidate,
                len(candidates[candidate]),
            )
        )
    result.sort(key=lambda t: t[1], reverse=True)
    return result


def get_hardest_response(
    guess: str, answers: List[str]
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Determine the hardest response for a guess given a list of possible answers.

    This function wraps the _async function for use in multiprocessing.Pool above.

    Args:
        guess (str): guess
        answers (List[str]): list of legal answers

    Returns:
        Tuple[str, Dict[str, List[str]]]: mapping of guess to results and their
        candidates
    """
    return asyncio.run(_get_hardest_response(guess, answers))


async def _get_hardest_response(
    guess: str, answers: List[str]
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Determine the hardest response for a guess given a list of possible answers.

    Args:
        guess (str): guess
        answers (List[str]): list of legal answers

    Returns:
        Tuple[str, Dict[str, List[str]]]: mapping of guess to results and their
        candidates
    """
    candidates: defaultdict[str, List[str]] = defaultdict(list)
    for answer in answers:
        response = eval_guess(guess, answer)
        candidates[response].append(answer)
    return guess, candidates


def eval_guess(guess: str, answer: str) -> str:
    """
    Compute the result of a given guess and a given answer.

    Args:
        guess (str): guess
        answer (str): answer

    Raises:
        Exception: raised if the guess and answer don't have the same length.

    Returns:
        str: result of the guess: "g" indicates right letter and place, "y" indicates
        right letter, wrong place, and "r" indicates wrong letter.
    """
    if len(guess) != len(answer):
        raise Exception(f"{guess} and {answer} do not have matching length")
    result = ["r"] * 5
    count: defaultdict[str, int] = defaultdict(int)
    seen: defaultdict[str, int] = defaultdict(int)

    for c in answer:
        count[c] += 1

    for i, c in enumerate(guess):
        if c == answer[i]:
            result[i] = "g"
            seen[guess[i]] += 1

    for i, c in enumerate(guess):
        if result[i] != "g":
            if guess[i] in answer:
                if seen[guess[i]] < count[guess[i]]:
                    result[i] = "y"
                    seen[guess[i]] += 1

    return "".join(result)


def is_candidate(word: str, guesses: List[str], responses: List[str]) -> bool:
    """
    Compute whether or not a word is a candidate answer for given word and resposnes.

    Args:
        word (str): possible answer
        guesses (List[str]): list of previous guesses
        responses (List[str]): list of previous responses to guesses

    Returns:
        bool: [description]
    """
    for i, guess in enumerate(guesses):
        if not is_legal_guess(word, guess, responses[i]):
            return False
    return True


def is_legal_guess(word: str, guess: str, response: str) -> bool:
    """
    Compute whether or not a guess is legal.

    Args:
        word (str): candidate word
        guess (str): guess
        response (str): guess response

    Raises:
        Exception: raised if word/guess don't have the same length
        Exception: raised if guess/clues don't have the same length

    Returns:
        bool: whether or not a word is a candidate given the guess and response
    """
    if len(word) != len(guess):
        raise Exception(f"{word} and {guess} do not have matching length")
    if len(guess) != len(response):
        raise Exception(f"{guess} and {response} do not have matching length")
    count: defaultdict[str, int] = defaultdict(int)
    for c in word:
        count[c] += 1
    seen: defaultdict[str, int] = defaultdict(int)
    for i, c in enumerate(guess):
        match response[i]:
            case "r":
                if c in word:
                    return False
            case "g":
                if word[i] != c or seen[c] == count[guess[i]]:
                    return False
                seen[c] += 1
            case "y":
                if word[i] == c or seen[c] == count[guess[i]]:
                    return False
    return True


@functools.lru_cache()
def enumerate_responses(n: int = 5) -> List[str]:
    """
    Generate a list of all possible guess responses.

    Args:
        n (int, optional): size of guess/response. Defaults to 5.

    Returns:
        List[str]: list of valid responses
    """
    results = [""]
    possibilities = ["r", "y", "g"]
    for _ in range(n):
        results = [r + c for r in results for c in possibilities]
    return results


if __name__ == "__main__":
    main()
