# @name binary_search_main_driver.py
"""
@ref https://docs.python.org/3/library/collections.html#collections.deque
"""
import binary_search

test_list = [1, 3, 9, 11, 15, 19, 29]

test_val1 = 25
test_val2 = 15

print(binary_search.binary_search(test_list, test_val1))
print(binary_search.binary_search(test_list, test_val2))