from Voltron.DataStructures.binary_search_tree import Node

import pytest

test_values = [5, 15, 2, 5, 13, 22, 1, 14]

@pytest.fixture
def sample_binary_search_tree_fixture():

    r = Node(10)

    r.insert_values(test_values)

    return r


def test_binary_sesarch_tree_node_default_constructs():

    r = Node()
    assert not r.value_ 
    assert not r.left_ 
    assert not r.right_ 

def test_binary_sesarch_tree_node_constructs():

    r = Node(42)
    assert r.value_ 
    assert r.value_ == 42
    assert not r.left_ 
    assert not r.right_ 

def test_binary_sesarch_tree_node_inserts_to_left():

    r = Node(42)

    r.insert(41)

    assert r.left_
    assert r.left_.value_ == 41

def test_binary_search_tree_node_inserts_to_right():

    r = Node(42)

    r.insert(43)

    assert not r.left_
    assert r.right_
    assert r.right_.value_ == 43

def test_binary_search_tree_node_inserts_more_than_once():

    r = Node(42)

    r.insert(41)

    r.insert(43)

    r.insert(42)

    r.insert(45)

    assert r.left_
    assert r.left_.value_ == 41

    assert r.right_
    assert r.right_.value_ == 43

    assert r.right_.right_
    assert r.right_.right_.value_ == 45

def test_insert_with_repeat_inserts_more_than_once():

    r = Node(10)

    r.insert_with_repeat(5)
    r.insert_with_repeat(15)
    r.insert_with_repeat(2)
    r.insert_with_repeat(5)
    r.insert_with_repeat(13)
    r.insert_with_repeat(22)
    r.insert_with_repeat(1)
    r.insert_with_repeat(14)

    assert r.value_ == 10
    assert r.left_.value_ == 5
    assert r.right_.value_ == 15
    assert r.left_.left_.value_ == 2
    assert r.left_.right_.value_ == 5
    assert r.right_.left_.value_ == 13
    assert r.right_.right_.value_ == 22
    assert r.left_.left_.left_.value_ == 1
    assert r.right_.left_.right_.value_ == 14


def test_insert_values_with_repeat():

    r = Node(10)

    r.insert_values(test_values)

    assert r.value_ == 10    
    assert r.left_.value_ == 5
    assert r.right_.value_ == 15
    assert r.left_.left_.value_ == 2
    assert r.left_.right_.value_ == 5
    assert r.right_.left_.value_ == 13
    assert r.right_.right_.value_ == 22
    assert r.left_.left_.left_.value_ == 1
    assert r.right_.left_.right_.value_ == 14


def test_preorder(sample_binary_search_tree_fixture):

    r = sample_binary_search_tree_fixture

    preorder_values = r.preorder()

    assert len(preorder_values) == 9

    assert preorder_values[0] == 10
    assert preorder_values[1] == 5
    assert preorder_values[2] == 2
    assert preorder_values[3] == 1
    assert preorder_values[4] == 5
    assert preorder_values[5] == 15
    assert preorder_values[6] == 13
    assert preorder_values[7] == 14
    assert preorder_values[8] == 22


def test_inorder(sample_binary_search_tree_fixture):

    r = sample_binary_search_tree_fixture

    inorder_values = r.inorder()

    assert len(inorder_values) == 9

    assert inorder_values[0] == 1
    assert inorder_values[1] == 2
    assert inorder_values[2] == 5
    assert inorder_values[3] == 5
    assert inorder_values[4] == 10
    assert inorder_values[5] == 13
    assert inorder_values[6] == 14
    assert inorder_values[7] == 15
    assert inorder_values[8] == 22


def test_postorder(sample_binary_search_tree_fixture):

    r = sample_binary_search_tree_fixture

    postorder_values = r.postorder()

    assert len(postorder_values) == 9

    assert postorder_values[0] == 1
    assert postorder_values[1] == 2
    assert postorder_values[2] == 5
    assert postorder_values[3] == 5
    assert postorder_values[4] == 14
    assert postorder_values[5] == 13
    assert postorder_values[6] == 22
    assert postorder_values[7] == 15
    assert postorder_values[8] == 10
