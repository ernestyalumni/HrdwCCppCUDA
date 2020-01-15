"""
@file practice_main_driver.py
"""
from practice import *

A = [1,3,6,4,1,2]
A1 = [1,2,3]
A2 = [-1, -3]

# binary_gap_solution(9) # 2
# binary_gap_solution(529) # 4 (4, 3 but 4 >3)
# binary_gap(20) # 1
# binary_gap(15) # 0

bgi_test0 = [1, 1] # 0 
bgi_test1 = [1, 1, 1] # 0
bgi_test2 = [1, 0, 1] # 
bgi_test3 = [1, 0, 0, 1]
bgi_test4 = [1, 1, 0, 1]
bgi_test5 = [1, 0, 1, 1]
bgi_test6 = [1, 0, 1, 0, 1]

print(binary_gap_iteration(bgi_test0))
print(binary_gap_iteration(bgi_test1))
print(binary_gap_iteration(bgi_test2))
print(binary_gap_iteration(bgi_test3))
print(binary_gap_iteration(bgi_test4))
print(binary_gap_iteration(bgi_test5))
print(binary_gap_iteration(bgi_test6))

# Triangle sort
# https://app.codility.com/c/run/trainingNEM3UV-WGK/

ts_test0 = [10,50,5,1]


# Fish stack

fsB0 = [0, 0]
fsB1 = [1, 1]
fsB2 = [1, 0]
fsB3 = [0, 1]
fsB4 = [0, 0, 1]
fsB5 = [0, 1, 1]
fsB6 = [1, 0, 1]

fsAA0 = [4, 3, 2, 1, 5]
fsBB0 = [0, 1, 0, 0, 0]