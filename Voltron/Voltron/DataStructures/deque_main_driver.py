# @name deque_main_driver.py
"""
@ref https://docs.python.org/3/library/collections.html#collections.deque
"""
from collections import deque
from copy import copy, deepcopy

from queues import Queue

d = deque('ghi')            # make a new deque with 3 items
for elem in d:              # iterate over the deque's elements
    print(elem.upper())     # G H I

d.append('j')               # add a new entry to the right side
d.appendleft('f')           # add a new entry to the left side
print(d)                    # show the representation of the deque
# [f g h i j]

print(d.pop())  # 'j'       # return and remove the rightmost item
print(d.popleft()) # 'f'    # return and remove the leftmost item
print(list(d)) # ['g', 'h', 'i'] # list the contents of the deque

# append is the push to popleft
# appendleft is the push to pop

print(d[0]) # 'g'           # peek at leftmost item
print(d[-1]) # 'i'          # peek at rightmost item

list(reversed(d)) # ['i', 'h', 'g'] # list the contents of a deque in reverse

print('h' in d) # True      # search the deque

d.extend('jkl')             # add multiple elements at once
print(d) # ['g', h i j k l]

# Rotate the deque n steps to the right. If n is negative, rotate to the left
d.rotate(1) # right rotation
print(d) # [l g h i j k]

d.rotate(-1) # left rotation
print(d) # [g h i j k l]

print(deque(reversed(d))) # [l k j i h g] # make a new deque in reverse order

d.clear() # empty the deque

try:
  d.pop() # cannot pop from an empty deque
except IndexError as err:
  print(err)

d.extendleft('abc') # extendleft() reverses the input order
print(d) # [c b a]

test_sequence = range(1, 8)

d = deque(copy(test_sequence))

d.rotate(2)
print(d)

d.clear()
d.extend(test_sequence)

d.rotate(-3)
print(d)

# https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78875255560923
# Queue Practice

# Setup
q = Queue(1)
q.enqueue(2)
q.enqueue(3)

# Test peek
# Should be 1
print(q.peek())

# Test dequeue
# Should be 1
print(q.dequeue())

# Test enqueue
q.enqueue(4)
# Should be 2
print(q.dequeue())
# Should be 3
print(q.dequeue())
# Should be 4
print(q.dequeue())
q.enqueue(5)
# Should be 5
print(q.peek())