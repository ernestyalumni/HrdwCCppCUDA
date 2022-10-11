"""
@ref https://youtu.be/oBt53YbR9Kk?t=1903 Dynamic Programming - Learn to Solve Algorithmic Problems & Coding Challenges, freeCodeCamp.org
"""

def fibonacci(n, memo = {}):
    """
    @details First n stack calls on first branch.
    O(n) space for memo. By induction, when unwinding one step "up" in call
    stack, the memo can provide turn value for next branch immediately. Thus,    
    O(n) time.
    """
    # Uncomment to see memo.
    #print(memo)

    if (n in memo):
        return memo[n]
    if (n < 2):
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]


def grid_traveler(m, n, memo = {}):
    """
    @input m - number of rows
    @input m - number of columns

    @ref https://youtu.be/oBt53YbR9Kk?t=2503
    @details How can we frame the problem where we decrease the problem size,
    usually by mutating my arguments to my function call.
    
    Given 3x3 example,
    if I first move downward, my only next playable area is 2x3.
    If I move to the right, my only next playable area is 3x2.

    O(m * n) time. O(m) or O(n) (max of m,n) space from stack frames. O(m * n)
    space for memo.
    """
    # Not a grid with any tiles at all.
    if (m < 1 or n < 1):
        return 0

    if m == 1 or n == 1:
        return 1

    if (m, n) in memo:
        return memo[(m, n)]

    # Add up the number of ways to travel a grid after a down move with number
    # of ways to travel a grid after a right move.

    memo[(m, n)] = grid_traveler(m - 1, n) + grid_traveler(m, n - 1)

    return memo[(m, n)]
