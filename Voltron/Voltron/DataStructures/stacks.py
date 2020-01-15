# @name stack.py
# @ref https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78792965540923
#
# a Python list is a stack if treated as 
# a.append(5) # push
# a.pop() # 5 pop, LIFO

from linked_list import (Element, LinkedList)

class Stack(LinkedList):
	"""
	@details LIFO Last-In First-Out
	"""
	def __init__(self, head=None):
		super().__init__(head)

    def insert_first(self, new_element):
        """Insert new element as the head of the LinkedList"""

        new_element.next = self.head

        self.head = new_element

        pass

    def delete_first(self):
        """Delete the first (head) element in the LinkedList as return it"""
        if not self.head:
            return

        if not self.head.next:
            self.head = None
            return

        self.head = self.head.next
        pass


# Udacity solution
# https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78792965540923
class UdacityElement(object):
    def __init__(self, value):
        self.value = value
        self.next = None

class UdacityLinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def insert_first(self, new_element):
        new_element.next = self.head
        self.head = new_element

    def delete_first(self):
        if self.head:
            deleted_element = self.head
            temp = deleted_element.next
            self.head = temp
            return deleted_element
        else:
            return None

class UdacityStack(object):
    def __init__(self,top=None):
        self.ll = UdacityLinkedList(top)

    def push(self, new_element):
        self.ll.insert_first(new_element)

    def pop(self):
        return self.ll.delete_first()