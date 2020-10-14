"""
@file Easy937-ReorderDataInLogFiles.py
"""

example_input_logs = [
    "dig1 8 1 5 1",
    "let1 art can",
    "dig2 3 6",
    "let2 own kit dig",
    "let3 art zero"]
example_output = [
    "let1 art can",
    "let3 art zero",
    "let2 own kit dig",
    "dig1 8 1 5 1",
    "dig2 3 6"]

from functools import total_ordering
from operator import itemgetter

# 10/12/2020 22:58  Accepted    48 ms   14.3 MB python3
@total_ordering
class LetterLog:
    """
    @url https://portingguide.readthedocs.io/en/latest/comparisons.html
    """
    def __init__(self, letter_log):
        # Expected type is a space-delimited string.
        self.letter_log = letter_log
        # Expected to be at least of size 2 as given in problem.
        self.letter_log_list = letter_log.split()

    def __eq__(self, other):
        return (all([
            (log == other_log)
            for log, other_log
            in zip(self.letter_log_list, other.letter_log_list)]) and
                len(self.letter_log_list) == len(other.letter_log_list))

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if (self == other):
            return False

        letter_log_words = self.letter_log_list[1:]
        other_words = other.letter_log_list[1:]
        if letter_log_words == other_words:
            return self.letter_log_list[0] < other.letter_log_list[0]

        # zip will iterate over smallest list.
        for word, other_word in zip(letter_log_words, other_words):
            if (word != other_word):
                return word < other_word

        # Expect then for the words of each to have different lengths.
        return len(letter_log_words) < len(other_words)

    def __repr__(self):
        return self.letter_log


class Solution:
#    def reorderLogFiles(self, logs: List[str]) -> List[str]:
    def reorderLogFiles(self, logs):
        """
        letter-logs, each word consist only of lowercase letters
        digit-logs, each word consist only of digits
        
        Guaranteed that each log has at least 1 word after identifier.

        Want: letter-logs come before digit-logs.
        letter-logs ordered lexicographically, ignoring identifier - identifier
        used in case of ties.
        digit-logs put in their original order.

        Let N be the number of logs in the list.
        Let M max. length of a single log.

        """
        # Time (Zeit) O(N)
        # Space O(N)
        digit_logs = [log for log in logs if self.is_digit_log(log)]
        letter_logs = [log for log in logs if self.is_letter_log(log)]

        ordered_letter_logs = [LetterLog(log) for log in letter_logs]

        # Time O(N log N).
        # Comparison function invokved O(N log N) times.
        # O(M) times to compare the contents of the letter logs.
        sorted_letter_logs = sorted(ordered_letter_logs)

        sorted_letter_logs = [
            letter_log.letter_log for letter_log in sorted_letter_logs]

        # Time O(M N log N)
        # Space O(M log N)
        # log N for the quicksort algorithm space complexity.
        # M space to hold parsed logs, in this case, LetterLog.
        return sorted_letter_logs + digit_logs

    def is_letter_log(self, log):
        """
        Assume input log follows the given format:
        - guaranteed to have an alphanumeric identifier.

        Assume log is at least of size 2; we're given that a log has at least
        one word after its identifier.
        """
        # list of strings
        log_list = log.split()

        def is_word_lowercase(word):
            return word.islower()

        # Assume first word is identifier.
        return all(map(is_word_lowercase, log_list[1:]))

    def is_digit_log(self, log):
        log_list = log.split()

        def is_digit(word):
            return word.isdigit()

        return all(map(is_digit, log_list[1:]))

    def letter_log_compare(letter_log_1, letter_log_2):
        """
        Does a comparison for "less than" ("<").
        """



    def sort_same_letter_log(self, letter_logs):
        """
        Assume letter_logs consist of logs with the exact same words.
        """
        letter_logs_lists = [log.split() for log in letter_logs]

        sorted_letter_logs_lists = sorted(letter_logs_lists, key=itemgetter(0))

        return [" ".join(log) for log in sorted_letter_logs_lists]

    def sort_two_letter_logs(self, letter_logs):
        """
        Assume len(letter_logs) == 2
        """
        letter_logs_lists = [log.split() for log in letter_logs]

        if (letter_logs_lists[0][1:] == letter_logs_lists[1][1:]):
            return self.sort_same_letter_log(letter_logs)

        N_min = min([len(log[1:]) for log in letter_logs_lists])

        sorted_letter_logs_list = sorted(
            letter_logs_lists, key=itemgetter(*range(1, N_min + 1)))

        return [" ".join(log) for log in sorted_letter_logs_lists]

    def induction_sort_letter_logs(self, sorted_letter_logs, letter_log):
        """
        Assume sorted_letter_logs is already sorted. Now we're going to sort
        sorted_letter_logs and an additional element letter_log
        """
        letter_logs_lists = [log.split() for log in sorted_letter_logs]

        letter_log_words = letter_log.split()[1:]

        def check_if_same_words(log):
            return log[1:] == letter_log_words

        if (all(map(check_if_same_words, letter_logs_lists))):
            return sort_same_letter_log(sorted_letter_logs + [letter_log, ])

        N_min = min([len(log[1:]) for log in letter_logs_lists])
        N_min = min([N_min, len(letter_log_words)])

        sorted_letter_logs_list = sorted(
            letter_logs_lists + [letter_log_words,],
            key=itemgetter(*range(1, N_min + 1)))

        return [" ".join(log) for log in sorted_letter_logs_list]


    def order_letter_logs(self, letter_logs):
        """
        letter_logs - Python list of strings, each string or "log" is a
        space-delimited. 
        """
        # Consider the case of 
        sorted_letter_logs = letter_logs[0:1]

        N = len(letter_logs)
        for index in range(1, N):

            sorted_letter_logs = self.induction_sort_letter_logs(
                sorted_letter_logs,
                letter_logs[index])

        return sorted_letter_logs


if __name__ == "__main__":

    print("\nLeetCodePlayground\n")
    solution = Solution()

    print("\n Test your implementation (reorderLogFiles)")
    results = solution.reorderLogFiles(example_input_logs)
    print(results == example_output) # True