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

    def _get_next_and_count(self, j, current):

        if not current.next:
            return j, current

        return j + 1, current.next

    def length(self):
        index = 1
        current = self.head

        while (True):
            new_index, new_current = self._get_next_and_count(index, current)
            if new_index == index:
                return index
            index = new_index
            current = new_current

    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""

        index = 1
        current = self.head

        for i in range(1, position + 1):
            if (index == position):
                break

            new_index, new_current = self._get_next_and_count(index, current)
            if (new_index == index):
                break
            index = new_index
            current = new_current

        if index == position:
            return current

        return None

    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""

        if position == 1:
            new_element.next = self.head
            self.head = new_element
            return

        prior_element = self.get_position(position - 1)
        after_element = self.get_position(position)

        new_element.next = after_element

        prior_element.next = new_element

        pass

    def search(self, value):

        index = 1
        current = self.head

        if current.value == value:
            return index, current

        while (True):
            new_index, new_current = self._get_next_and_count(index, current)
            # At the end or tail of linked list.
            if new_index == index:
                break
            index = new_index
            current = new_current

            # Found value, leave while loop.
            if new_current.value == value:
                break
        
        if (current.value == value):
            return index, current

        return None        

    def delete(self, value):
        """Delete the first node with a given value."""

        results = self.search(value)

        # Nothing to delete!
        if not results:
            return None

        position, node = results

        if position == 1:
            self.head = self.get_position(2)

        if position != 1:

            prior_element = self.get_position(position - 1)

            after_element = self.get_position(position + 1)

            prior_element.next = after_element

        pass

    def print(self):
        index = 1
        current = self.head
        print(current.value, ' ')

        while (True):
            new_index, new_current = self._get_next_and_count(index, current)
            if new_index == index:
                return
            index = new_index
            current = new_current
            print(current.value, ' ')

# Udacity's solution:

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

    def get_position(self, position):
        counter = 1
        current = self.head
        if position < 1:
            return None
        while current and counter <= position:
            if counter == position:
                return current
            current = current.next
            counter += 1
        return None

    def insert(self, new_element, position):
        counter = 1
        current = self.head
        if position > 1:
            while current and counter < position:
                if counter == position - 1:
                    new_element.next = current.next
                    current.next = new_element
                current = current.next
                counter += 1
        elif position == 1:
            new_element.next = self.head
            self.head = new_element

    def delete(self, value):
        current = self.head
        previous = None
        while current.value != value and current.next:
            previous = current
            current = current.next
        if current.value == value:
            if previous:
                previous.next = current.next
            else:
                self.head = current.next
