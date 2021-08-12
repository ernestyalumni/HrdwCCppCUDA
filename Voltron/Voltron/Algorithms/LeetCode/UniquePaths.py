def _unique_paths(m: int, n: int, memo = {}):
    """
    @ref https://leetcode.com/problems/unique-paths/
    @ref https://youtu.be/oBt53YbR9Kk Dynamic Programming - Learn to Solve
    Algorithmic Problems & Coding Challenges. freeCodeCamp.com

    @details 62. Unique Paths.

    How many possible unique paths are there?

    08/12/2021 00:18    Accepted    32 ms   14.3 MB python3
    """
    if (m < 0 or n < 0):
        return 0

    if (m == 1 or n == 1):
        return 1

    if (m, n) in memo:
        return memo[(m, n)]

    memo[(m, n)] = _unique_paths(m - 1, n, memo) + _unique_paths(m, n - 1, memo)

    return memo[(m, n)]

def unique_paths(m: int, n: int):
    return _unique_paths(m, n)