"""
@file practice.py
"""
from collections import deque

def solution(A):
    max_of_A = max(A)
    if (max(A) < 0):
        return 1
    if (min(A) > 1):
        return 1

    A_as_set = set(A) # O(N)
    reduced_A = list(A_as_set)
    reduced_A.sort()
    if (len(reduced_A) == max_of_A):
        return max_of_A + 1

    for i in range(len(reduced_A)):
        if reduced_A[i] > (i + 1):
            return (i + 1)

    # Should not exit this way
    return -1 

# https://app.codility.com/programmers/lessons/1-iterations/binary_gap/

def binary_gap_iteration(a):
    # assume A has 1s at both ends and 0s in between
    if len(set(a)) == 1:
        return True, 0
    # base case, only zeros in between ends
    if len(set(a[1:len(a) - 1])) == 1 and (0 in set(a[1:len(a) - 1])):
        return True, len(a) - 2

    next_index = a[1:].index(1)

    # Add one to the last one because we were "reindexing"
    return False, [next_index, a[next_index + 1:]]

def binary_gap_solution(N):
    x = bin(N)
    # take out 0b
    list_rep = list(map(int, x[2:]))
    """
    try:
        list_rep = list(map(int, "{0:b}".format(N)))
    except TypeError:
        x = bin(N)
        # take out 0b
        list_rep = list(map(int, x[2:]))
    """

    set_rep = set(list_rep)
    if len(set_rep) == 1: # all 0s or all 1s
        return 0

    # cut out each ends if they're filled with 0s
    l = list_rep.index(1)
    list_rep.reverse()
    r = list_rep.index(1)
    r = len(list_rep) - 1 - r

    list_rep.reverse()
    if (l == r) or (l + 1 == r): # only 1 1 or only 11
        return 0

    clean_list_rep = list_rep[l:r + 1]

    stack_of_gaps = []
    list_of_gap_lengths = []

    found, result = binary_gap_iteration(clean_list_rep)
    if found:
        return result

    list_of_gap_lengths.append(result[0])
    stack_of_gaps.append(result[1])

    while (stack_of_gaps != []):
        gap_to_test = stack_of_gaps.pop()        
        found, result = binary_gap_iteration(gap_to_test)
        if found:
            list_of_gap_lengths.append(result)
        else:
            list_of_gap_lengths.append(result[0])
            stack_of_gaps.append(result[1])

    return max(list_of_gap_lengths)

# https://app.codility.com/c/run/trainingKYNAWH-M8J/
# small frog

def small_frog_solution(X, Y, D):
    if ((Y - X) % D) == 0:
        return (Y - X) // D

    return (Y - X) // D + 1

# https://app.codility.com/c/run/trainingD9A84F-2SW/
def rotate_solution(A, K):
    for i in range(K):
        A = A[-1:] + A[:-1]
    return A

def triangle_sort_solution(A):
    # write your code in Python 3.6
    sorted_index_A = sorted(list(enumerate(A)), key=lambda pair : pair[1])
    for index in range(len(sorted_index_A) - 2):
        if ((sorted_index_A[index][1] + sorted_index_A[index + 1][1]) > \
            sorted_index_A[index + 2][1]):
            return 1
    return 0

# stacks and queues
# https://app.codility.com/c/run/trainingEMEQQX-UPB/

def stack_fish_initial_check(B):
    # return true if there's nothing to do
    B_as_set = set(B)
    if (len(B_as_set) == 1):
        return True, len(B)
    B_minus_last_set = set(B[:-1])
    if (len(B_minus_last_set) == 1) and (0 in B_minus_last_set) and \
        (B[-1] == 1):
        return True, len(B)
    return False, B

def stack_fish_meet_check(p, B):
    return (B[p] == 1 and B[p + 1] == 0)

def stack_fish_meet_check_all(B):
    # return true if there's nothing to do
    if len(B) == 1:
        return True
    for p in range(len(B) - 1):
        if stack_fish_meet_check(p, B):
            return False
    return True

def stack_fish_check(a, b):
    # Assume 1, 0
    da = deque(a)
    db = deque(b)
    fish_p = []
    fish_q = []
    fish_p.append(da.popleft())
    fish_q.append(da.popleft())
    fish_p.append(db.popleft())
    fish_q.append(db.popleft())

    if (fish_p[0] > fish_q[0]):
        da.appendleft(fish_p[0])
        db.appendleft(fish_p[1])

    if (fish_q[0] > fish_p[0]):
        da.appendleft(fish_q[0])
        db.appendleft(fish_q[1])
    return list(da), list(db)          

def stack_fish_iteration(a, b, p):
    temp_a = a[:p]        
    temp_b = b[:p]

    check_a = a[p:]
    check_b = b[p:]
    check_a, check_b = stack_fish_check(check_a, check_b)
    return temp_a + check_a, temp_b + check_b    

def stack_fish_solution_slow(A, B):

    while (True):

        found, result = stack_fish_initial_check(B)
        if found:
            return result
        found = stack_fish_meet_check_all(B)
        if found:
            return len(B)

        for p in range(len(A) - 1):
            if stack_fish_meet_check(p, B):
                A, B = stack_fish_iteration(A, B, p)


def stack_fish_solution(A, B):
    downstream_stack = []
    survivors = 0

    for q in range(len(A)):
        if B[q] == 1:
            # downstream fish, it's ready to eat other fish downstream
            downstream_stack.append(A[q])
        else:
            # upstream fish, stack contains previous fish headed downstream
            # upstream fish will surely encounter the top fish on stack
            while (len(downstream_stack) != 0):

                A_p = downstream_stack[-1]

                # p eats q, so upstream q died and downstream p lives
                if (A_p > A[q]):
                    break

                # q eats p, and so p is no longer in the stack
                downstream_stack.pop()
            # if upstream fish q eats all downstream fish, it survives.
            if len(downstream_stack) == 0:
                survivors += 1

    survivors += len(downstream_stack)

    return survivors


