"""
@file test_MIT_OCW_6_0001_lecture_code.py

@details

EXAMPLE USAGE:

pytest ./test_MIT_OCW_6_0001_lecture_code.py
or
pytest test_MIT_OCW_6_0001_lecture_code.py

"""
from Voltron.MIT_6_0001 import lecture_code
from Voltron.MIT_6_0001.lecture_code import (
    # In the order they appear on lecture, since that's how they were
    # introduced conceptually.
    is_palindrome,
    ReturnTuple,
    IterateOverTuples,
    SumElements,
    ListOperations,
    AliasExampleWithLists,
    CloningIsDeepCopying,
    MutateWhileIterating)

import pytest

"""
@file lec5_tuples_lists.py
MIT OCW 6.0001 Fall 2016
"""

@pytest.fixture
def example_lists():
    class ExampleLists:
        l1 = [2,4]
        l2 = [1,3,5,7,9]

    return ExampleLists()

#########################
## EXAMPLE: returning a tuple
#########################
def test_ReturnTuple_quotient_and_remainder():
    assert(ReturnTuple.quotient_and_remainder(5,3) == (1, 2))

#########################
## EXAMPLE: iterating over tuples
#########################
def test_IterateOverTuples():
    """
    @brief Demonstrates that we can iterate over a tuple.
    """
    test_on_data_1 = IterateOverTuples.get_data_test() == (1, 7, 2)
    test_on_data_2 = IterateOverTuples.get_data_tswift() == (2008, 2014, 5)

    assert(test_on_data_1 and test_on_data_2)

#########################
## EXAMPLE: sum of elements in a list
#########################
def test_SumElements():
    test1_worked = SumElements.test_method1() == 10
    test2_worked = SumElements.test_method2() == 10
    assert(test1_worked and test2_worked)

#########################
## EXAMPLE: various list operations
## put print(L) at different locations to see how it gets mutated
#########################

def test_ListOperations():
    l = ListOperations()
    L1_is_equal_to_its_value = (l.L1 == [2,1,3])
    summed_L1_and_L2_in_order = l.L3 == [2,1,3,4,5,6]    

    l.extend_L1()

    L1_now_extended = (l.L1 == [2,1,3,0,6])
    list_concatenation_tested = (L1_is_equal_to_its_value and
        summed_L1_and_L2_in_order and
        L1_now_extended)

    assert(list_concatenation_tested)

def test_list_append_mutates_original_list():
    l = [2,1,3]
    l.append(5)
    assert(l == [2,1,3,5])

def test_removing_list_elements():
    """
    @details Demonstartes also mutability of a list.
    """
    l = ListOperations()
    L = l.L
    L.remove(2)
    L_removes_value_2 = (L == [1,3,6,3,7,0])

    L.remove(3)
    L_removes_value_3 = (L == [1,6,3,7,0])

    del(L[1])
    L_deletes_at_index = (L==[1,3,7,0])

    x = L.pop()
    x_is_last_or_tail_elements = (x == 0)
    pop_L_removes_tail_element = (L == [1,3,7])

    lists_are_mutable = (l.L == [1,3,7])

    assert(L_removes_value_2 and L_removes_value_3 and
        L_deletes_at_index and x_is_last_or_tail_elements and
            pop_L_removes_tail_element and lists_are_mutable)

def test_string_list_transformations():
    s = "I<3 cs"
    sl = list(s)
    list_splits_up_each_character = sl == ['I','<','3',' ','c','s']
    splits = s.split('<')
    split_creates_new_list = splits == ['I','3 cs']
    L = ['a', 'b', 'c']
    joined_L = ''.join(L) == 'abc'
    joined_by_underscore = '_'.join(L) == 'a_b_c'

    assert(list_splits_up_each_character and split_creates_new_list and
        joined_L and joined_by_underscore)

def test_lists_sorted_and_reversed():
    L = [9,6,0,3]
    x = sorted(L)
    L_sorted = (x == [0,3,6,9])
    y = L.reverse()
    reverse_returns_None = (y == None)
    L_is_mutably_reversed = (L == [3,0,6,9])

    assert(L_sorted and reverse_returns_None and L_is_mutably_reversed)

#########################
## EXAMPLE: aliasing
## EY: Aliasing is all about aliases. In the global frame or frame in scope,
## the 2 or more aliases point to the same object (for example list object)
#########################

def test_AliasExampleWithLists():
    result = AliasExampleWithLists.run_example()
    both_got_mutated = result[0] == result[1]
    original_list_got_mutated = result[1] == ['red','yellow','orange','pink']
    assert(both_got_mutated and original_list_got_mutated)

def test_aliasing_assigning_variable_to_variable():
    a = 1
    b = a
    print(a)
    print(b)
    a_is_1 = (a == 1)
    b_is_1 = (b == 1)

    a = 3
    a_is_3 = (a == 3)
    b_is_1_again = (b == 1)
    assert(a_is_1 and b_is_1 and a_is_3 and b_is_1_again)

def test_aliasing_lists_are_mutable():
    warm = ['red', 'yellow', 'orange']
    hot = warm
    hot.append('pink')
    warm_equals_hot = (warm == hot)
    warm_changed = (warm == ['red', 'yellow', 'orange', 'pink'])

    assert(warm_equals_hot and warm_changed)

def test_concatenated_list_result_does_not_get_mutated(example_lists):
    l1 = example_lists.l1
    l2 = example_lists.l2

    l1_copied_correctly = l1 == [2, 4]
    l2_copied_correctly = l2 == [1, 3, 5, 7, 9]

    l3 = l1 + l2
    l3_concatenated = l3 == [2, 4, 1, 3, 5, 7, 9]
    x = l2.pop()
    x_is_last_element = x == 9
    l1.append(6)
    l1_mutated = l1 == [2,4,6]
    l2_mutated = l2 == [1,3,5,7]
    l3_not_mutated = l3 == [2,4,1,3,5,7,9]

    assert(l1_copied_correctly and l2_copied_correctly and x_is_last_element
        and l1_mutated and l2_mutated and l3_not_mutated)

def test_list_of_lists_are_mutated(example_lists):
    l1 = example_lists.l1
    l2 = example_lists.l2

    l1_copied_correctly = l1 == [2, 4]
    l2_copied_correctly = l2 == [1, 3, 5, 7, 9]

    ll3 = [l1, l2]
    ll3_is_list_of_lists = [[2, 4], [1,3,5,7,9]]
    del(l2[2])
    l1.append(6)
    l2_mutated = l2 == [1,3,7,9]
    ll3_mutated = ll3 == [[2,4,6], [1,3,7,9]]
    assert(l1_copied_correctly and l2_copied_correctly and l2_mutated and
        ll3_mutated)   

def test_tuples_of_aliases(example_lists):
    l1 = example_lists.l1
    l2 = example_lists.l2
    t1 = (l1, l2)
    del(l2[3])
    l1.append(7)
    tuple_of_aliases_mutated = t1 == ([2,4,7],[1,3,5,9])  
    assert(tuple_of_aliases_mutated)

#########################
## EXAMPLE: cloning
#########################
def test_CloningIsDeepCopying():
    te = CloningIsDeepCopying()
    te.mutate_chill_data_member()
    chill_mutated = (te.chill == ['blue','green','grey','black'])
    cool_not_mutated = (te.cool == ['blue','green','grey'])

    te.mutate_cl()
    ch_alias_of_cl = (te.cl == te.ch)
    ch_mutated = (te.ch == ['b','gr','gy','bl'])

    assert(chill_mutated and cool_not_mutated and ch_alias_of_cl and ch_mutated)


#########################
## EXAMPLE: sorting with/without mutation
#########################

def test_sort_mutates_sorted_does_not_mutate():
    warm = ['red', 'yellow', 'orange']
    sortedwarm = warm.sort()
    sortedwarm_is_None = (sortedwarm == None)
    warm_mutated = (warm == ['orange','red','yellow'])    

    cool = ['grey', 'green', 'blue']
    sortedcool = sorted(cool)    
    sortedcool_is_sorted = (sortedcool == ['blue','green','grey'])
    cool_is_not_mutated = (cool == ['grey','green','blue'])
    assert(sortedwarm_is_None and warm_mutated and sortedcool_is_sorted and
        cool_is_not_mutated)

###############################
## EXAMPLE: mutating a list while iterating over it
###############################

def test_MutateWhileIterating():
    L1 = [1, 2, 3, 4]
    L2 = [1, 2, 5, 6]

    MutateWhileIterating.remove_dups(L1, L2)
    # Python uses an internal counter to keep track of index it is in in the
    # loop.
    # Mutating changes lists length but PYthon doesn't update the counter.
    mutated_L1_skips_over_previous = (L1 == [2,3,4])

    L1 = [1, 2, 3, 4]
    MutateWhileIterating.remove_dups_new(L1, L2)
    cloned_L1_removes = (L1 == [3,4])

    L1 = [1, 2, 3, 4]
    MutateWhileIterating.remove_dups_v2(L1, L2)
    cloned_L1_removes_v2 = (L1 == [3,4])

    assert(mutated_L1_skips_over_previous and cloned_L1_removes and
        cloned_L1_removes_v2)

###############################
## EXERCISE: Test yourself by predicting what the output is and 
##           what gets mutated then check with the Python Tutor
###############################
def test_lec5_mutation_prediction():
    cool = ['blue','green']
    warm = ['red','yellow','orange']
    colors1 = [cool]
    colors1_is_list_of_list = colors1 == [['blue','green']]
    colors1.append(warm)
    colors1_is_still_list_of_list = colors1 == [
        ['blue','green'],
        ['red','yellow','orange']]

    colors2 = [['blue','green'],['red','yellow','orange']]
    colors2_equals_colors1 = colors1 == colors2

    warm.remove('red')

    colors1_mutated = colors1 == [['blue','green'],['yellow','orange']]
    colors2_unaffected = colors2 == [
        ['blue','green'],
        ['red','yellow','orange']]

    assert(colors1_is_list_of_list and colors1_is_still_list_of_list and
        colors2_equals_colors1 and colors1_mutated and colors2_unaffected)


def test_swapping_variables_and_tuples():
    # cf. https://youtu.be/RvRKT-jXvko?t=356
    # 5. Tuples, Lists, Aliasing, Mutability, and Cloning
    # MIT OCW 6.0001
    x = 3
    y = 5
    x_is_3 = (x == 3)
    y_is_5 = (y == 5)
    x = y
    x_is_5_now = (x == 5)
    y = x
    y_is_5_still = (y == 5)
    x = 3
    temp = x
    x = y
    y = temp
    swapped_x_is_5 = (x == 5)
    swapped_y_is_3 = (y == 3)

    # Swap in one line.
    (x, y) = (y, x)
    tuple_x_is_3 = (x == 3)
    tuple_y_is_5 = (y == 5)
    assert(x_is_3 and y_is_5 and x_is_5_now and y_is_5_still and swapped_x_is_5
        and swapped_y_is_3 and tuple_x_is_3 and tuple_y_is_5)


