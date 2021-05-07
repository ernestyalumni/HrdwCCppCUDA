###############################################################################
# @ref https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Factory.html
# Cormen, Leiserson, Rivest, and Stein (2009), Introduction to Algorithms, 3rd.
# Ed., Ch. 10 Elementary Data Structures, 10.1 Stacks and queues
###############################################################################

class AbstractStack:
    """
    @class AbstractStack

    @details Element deleted from set is one most recently inserted: the stack
    implements LIFO, last-in, first-out
    """

    def push(self, x):
        """
        @ref Cormen, Leiserson, Rivest, and Stein (2009), pp. 232, 101 Stacks
        and queues, pp. 232, Stacks.
        """
        pass

    def pop(self):
        pass

    def top(self):
        pass

    def size(self):
        pass        


class StackAsPythonList(AbstractStack):
    def __init__(self, input = None):
        if not input:
            self.data = []
        else:
            # Python | Cloning or Copying a list using slicing technique.
            # @ref https://www.geeksforgeeks.org/python-cloning-copying-list/
            self.data = input[:]

    def push(self, x):
        self.data.append(x)

    def pop(self):
        # LIFO
        return self.data.pop()

    def size(self):
        return len(self.data)

    def top(self):
        return self.data[self.size() - 1]


