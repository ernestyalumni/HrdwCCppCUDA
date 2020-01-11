# @name linked_list_main_driver.py
"""
@ref https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78875247320923
"""

from linked_list import (Element, LinkedList)


# Test cases
# Set up some Elements
e1 = Element(1)
e2 = Element(2)
e3 = Element(3)
e4 = Element(4)

# Start setting up a LinkedList
ll = LinkedList(e1)
ll.append(e2)
ll.append(e3)

# Test get_position
# Should print 3
print(ll.head.next.next.value)
# Should also print 3
print(ll.get_position(3).value)

# Test insert
ll.insert(e4,3)
# Should print 4 now
print(ll.get_position(3).value)

# Test delete
ll.delete(1)
# Should print 2 now
print(ll.get_position(1).value)
# Should print 4 now
print(ll.get_position(2).value)
# Should print 3 now
print(ll.get_position(3).value)

values1 = [5, -2, 23, 18, 8, 9, -5]

ll1 = LinkedList()
for ele in values1:
	ll1.append(Element(ele))
