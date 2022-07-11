"""
Use the Solution class to represent Leedcode problems - design
"""
from collections import defaultdict
from typing import List


class TrieNode:
    """description"""
    def __init__(self):
        self.dict = defaultdict(TrieNode)
        self.is_word = False


class StreamChecker:
    """ 1032. Stream of Characters (HARD) description"""
    """ Design an algorithm that accepts a stream of characters and checks if a suffix of these characters is a string of a given array of strings words.

    For example, if words = ["abc", "xyz"] and the stream added the four characters (one by one) 'a', 'x', 'y', and 'z', your algorithm should detect that the suffix "xyz" of the characters "axyz" matches "xyz" from words.

    Implement the StreamChecker class:

    - StreamChecker(String[] words) Initializes the object with the strings array words.
    - boolean query(char letter) Accepts a new character from the stream and returns true if any non-empty suffix from the stream forms a word that is in words.
    """

    def __init__(self, words: List[str]):
        """ Build a tire for each word in reversed order """
        # for user query record, init as empty string
        self.prefix = ''

        # for root node of tire, init as empty Trie
        self.tire = TrieNode()

        for word in words:
            # root
            cur_node = self.tire

            # make word in reverse order
            word = word[::-1]
            for char in word:
                cur_node = cur_node.dict[char]

                # make this tire path as a vaild word
                cur_node.is_word = True

    def query(self, letter: str) -> bool:
        """ Search user input in tire with reversed order docstring for query"""
        self.prefix += letter

        # root
        cur_node = self.tire

        for char in reversed(self.prefix):
            if char not in cur_node.dict:
                # current char not in Tire, impossible to match words
                break

            cur_node = cur_node.dict[char]
            if cur_node.is_word:
                # usr input match a word in Tire
                return True

        # No match
        return False
