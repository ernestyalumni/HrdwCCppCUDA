"""
@file peak_finding.py

@ref https://youtu.be/HtSuA80QTyo
1. Algorithmic Thinking, Peak Finding, MIT OCW
MIT 6.006 Introduction to Algorithms, Fall 2011
"""

def straightforward_search_1d(a):
    """
    @fn straightforward_search_1d
    """
    if (a[0] >= a[1]):
        return 0

    N = a.length()

    if (a[N - 1] >= a[N - 2]):
        return N - 1;

    for i in range(1, N - 1):
        if (a[i] >= a[i - 1] and a[i] >= a[i + 1]):
            return i

    return -1
