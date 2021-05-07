###############################################################################
# @details
# Example Usage:
#
# pytest DataStructures/test_stack.py
###############################################################################
from Voltron.DataStructures.stack import StackAsPythonList

def test_python_list_implementation_default_constructs():
    stack = StackAsPythonList()
    assert stack.size() == 0

def test_python_list_implementation_pops_and_pushes():
    """
    @ref Fig. 10.1, pp. 233, Sec. 10.1 Stacks and queues, Cormen, Leiserson,
    Rivest, and Stein (2009), Introduction to Algorithms, 3rd. Ed.
    """
    stack = StackAsPythonList()
    stack.push(15)    
    stack.push(6)    
    stack.push(2)    
    stack.push(9)

    top_value_1 = stack.top()
    size_1 = stack.size()

    stack.push(17)
    stack.push(3)
    top_value_2 = stack.top()
    size_2 = stack.size()    

    pop_value = stack.pop()
    top_value_3 = stack.top()
    size_3 = stack.size()

    assert top_value_1 == 9
    assert size_1 == 4
    assert top_value_2 == 3
    assert size_2 == 6
    assert pop_value == 3
    assert top_value_3 == 17
    assert size_3 == 5
