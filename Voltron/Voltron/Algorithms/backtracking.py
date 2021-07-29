# @ref https://medium.com/algorithms-and-leetcode/backtracking-e001561b9f28
#
#

def P_n_k(a, n, k, depth, used, current, answer):
    """
    @brief k-permutations of n elements computed.

    @details Demonstrates backtracking. P(n, k) = n (n-1) ... (n - k + 1)

    @param depth: starts from 0. Represents depth of the search.
    """
    # end condition
    if depth == k:
        # Use deepcopy because current is tracking all partial solutions, and
        # it eventually becomes []
        answer.append(current[::])
        return

    # O(n) in time complexity.
    for i in range(n):
        if not used[i]:
            # Generate the next solution from current
            current.append(a[i])
            used[i] = True
            print(current)

            # Move to the next solution.
            P_n_k(a, n, k, depth + 1, used, current, answer)

            # Backtrack to previous partial state.
            current.pop()
            print('backtrack: ', current)
            used[i] = False

    return

if __name__ == "__main__":

    a = [1, 2, 3]
    n = len(a)
    used = [False] * len(a)
    answer = []
    P_n_k(a, n, n, 0, used, [], answer)
    print(answer)