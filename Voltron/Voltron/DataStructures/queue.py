###############################################################################
# Cormen, Leiserson, Rivest, and Stein (2009), Introduction to Algorithms, 3rd.
# Ed., Ch. 10 Elementary Data Structures, 10.1 Stacks and queues
###############################################################################

class AbstractQueue:

    def enqueue(self, new_element):
        pass

    def dequeue(self):
        pass


class QueueAsPythonList(AbstractQueue):
    def __init__(self, another_list = None):
        if not another_list:
            self.data
        else:
            self.data = another_list[:]


    def enqueue(self, new_element):
        self.data.append(new_element)


    def dequeue(self):
        return self.data.pop(0)

    def peek(self):
        return self.data[0]

    def size(self):
        return len(self.data)


class QueueAsHeadTail(AbstractQueue):
    def __init__(self, array_implementation):
        self._implementation = array_implementation

    def enqueue(self, new_element):

        tail_index = self._implementation.tail
        self._implementation[tail_index] = new_element

        if (tail_index == (self._implementation.size() - 1)):
            # Wrap around.
            self._implementation.tail = 0
        else:
            self._implementation.tail = tail_index + 1


    def dequeue(self):
        head_index = self._implementation.head

        x = self._implementation[head_index]
        if (head_index == (self._implementation.size() - 1)):
            self._implementation.head = 0
        else:
            self._implementation.head = self._implementation.head + 1
        return x

    class PythonListWithHeadTail(list):

        self.head = 0
        self.tail = 0
        def size(self):
            return len(self)




