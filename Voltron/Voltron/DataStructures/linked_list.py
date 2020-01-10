# @name linked_list.py
# @ref https://classroom.udacity.com/courses/ud513/lessons/7117335401/concepts/78875247320923

"""The LinkedList code from before is provided below.
Add three functions to the LinkedList.
"get_position" returns the element at a certain position.
The "insert" function will add an element to a particular
spot in the list.
"delete" will delete the first element with that
particular value.
Then, use "Test Run" and "Submit" to run the test cases
at the bottom."""

class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList(object):
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

    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""

        def get_next_and_count(j, current):

            if not current.next:
                return j, None

            return j + 1, current.next

        iteration_stopped = False

        index = 1
        current = self.head

        while (not iteration_stopped and index < position):
            index, current = get_next_and_count(index, current)

            if not current:
                iteration_stopped = True

        if (index == position):
            return position

        return None
    
    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""
        pass
    
    
    def delete(self, value):
        """Delete the first node with a given value."""
        pass
