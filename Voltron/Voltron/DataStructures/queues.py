# @name queues.py
# @ref https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78875255560923

# @ref https://docs.python.org/3/library/collections.html
# deque - list-like container with fast appends and pops on either end
"""
@ref https://docs.python.org/3/library/collections.html#collections.deque

Returns new deque object initialized left-to-right( using append()) with data
from iterable. If iterable not specified, new deque empty

Deques support thread-safe, memory efficient appends, pops from either side of
deque, O(1) in either direction.

Deque vs. list. List optimized for fast fixed-length operations, incur O(N)
memory movement costs for pop(0) and insert(0, v) operations, which change both
size and position of underlying data representation.
"""
from collections import deque

# @ref https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78875255560923

class Queue:
    def __init__(self, head=None):
        self.storage = [head]

    def enqueue(self, new_element):

        self.storage.append(new_element)

        pass

    def peek(self):
        
        return self.storage[0]
        pass 

    def dequeue(self):

        return self.storage.pop(0)
      