"""
@brief 
"""
class QueueAsTwoStacks(object):
    """
    @ref https://betterprogramming.pub/how-to-implement-a-queue-using-two-stacks-80772242b88c
    """

    def __init__(self):
        self._stack1 = []
        self._stack2 = []

    def enqueue(self, item):
        """
        @details O(1) Time.
        """
        self._stack1.append(item)

    def is_empty(self):
        return len(self._stack1) == 0 and len(self._stack2) == 0

    def dequeue(self):

        if (self.is_empty()):

            raise IndexError("Can't deque from empty queue!")

        elif (len(self._stack2) > 0):
            return self._stack2.pop()

        # While stack1 is not empty, move items from stack1 to stack2, thus
        # reversing the order.
        elif (len(self._stack1) > 0):

            while (len(self._stack1) > 0):

                last_stack1_item = self._stack1.pop()
                self._stack2.append(last_stack1_item)

            return self._stack2.pop()

