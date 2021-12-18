from Voltron.DataStructures.queue_as_stacks import QueueAsTwoStacks

import pytest

def test_works_as_queue():
    q = QueueAsTwoStacks()
    q.enqueue(4)
    q.enqueue(1)
    q.enqueue(3)
    result = q.dequeue()
    assert result == 4
    q.enqueue(8)
    result = q.dequeue()
    assert result == 1
    result = q.dequeue()
    assert result == 3
    result = q.dequeue()
    assert result == 8
