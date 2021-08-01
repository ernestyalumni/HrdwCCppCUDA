class Node:
    def __init__(self, value):
        self.value_ = value
        self.left_ = None
        self.right_ = None

def _insert_value_into_binary_tree(node, value):
    if node.value_ == None:
        node.value_ = value

    current_value = node.value_

    if current_value == value:
        if (node.left_ is not None and node.right_ is not None):
            # Recursively deal with problem.
            _insert_value_into_binary_tree(node.left_, value)
            return
        elif (node.left_ is not None):
            node.right_ = Node(value)
            return
        else:
            assert node.left_ is None
            node.left_ = Node(value)
            return

    assert current_value is not value

    if (value < current_value):
        if (node.left_ is not None):
            return _insert_value_into_binary_tree(node.left_, value)

        node.left_ = Node(value)
        return


    assert current_value < value
    if (node.right_ is not None):
        return _insert_value_into_binary_tree(node.right_, value)

    node.right_ = Node(value)


def insert_values_into_binary_tree(node, values):
    for element in values:
        _insert_value_into_binary_tree(node, element)


def _inorder_traversal_recursive(node, results):

    if node is not None:

        _inorder_traversal_recursive(node.left_, results)
        results.append(node.value_)
        _inorder_traversal_recursive(node.right_, results)


def dfs_inorder_traversal_recursive(node):
    """
    @details O(N) time. Because of recursive function.
    O(N) worst case space complexity.
    O(log N) average space complexity, N is number of nodes.
    """

    results = []
    _inorder_traversal_recursive(node, results)
    return results


def dfs_inorder_traversal_iterative(node):
    results = []
    stack = []

    current_node = node

    while (current_node is not None or len(stack) > 0):

        while current_node is not None:
            # Reach all the way to the left until a leaf, adding to stack along
            # the way.
            stack.append(current_node)
            current_node = current_node.left_

        node = stack.pop()
        results.append(node.value_)
        current_node = node.right_

    return results


def dfs_inorder_traversal_iterative_with_visited(node):
    """
    @ref https://leetcode.com/problems/binary-tree-inorder-traversal/discuss/713539/Python-3-All-Iterative-Traversals-InOrder-PreOrder-PostOrder-Similar-Solutions
    """
    results = []
    visited = set()
    stack = [node,]

    while (len(stack) > 0):
        # The last element
        current_node = stack.pop()

        if current_node is not None:

            if current_node in visited:

                results.append(current_node.value_)

            else:
                # Inorder: left -> root -> right
                stack.append(current_node.right_)

                stack.append(current_node)
                visited.add(current_node)

                stack.append(current_node.left_)

    return results                


def depth_first_search_recursive(v, visited=None):
    """
    @ref https://www.techiedelight.com/depth-first-search/
    @ref https://en.wikipedia.org/wiki/Depth-first_search
    @ref https://stackoverflow.com/questions/36827377/implementing-dfs-and-bfs-for-binary-tree
    """

    if visited == None:
        visited = set()

    # Label node or vertex v as discovered.

    visited.append(v)

    # For all directed edges from v to w that are in G.adjacentEdges(v) do
    #   if vertex w is not labeled as discovered then
    #     recursively called DFS(G, w)

    if v.left_ is not None and v.left_ not in visited:
        depth_first_search_recursive(v.left_, visited)

    if v.right_ is not None and v.right_ not in visited:
        depth_first_search_recursive(v.right_, visited)

    return visited


#def depth_first_search_iterative(v, visited=None, stack=None):

