class TreeNode:
    """
    @brief Breadth-first Search 
    """
    def __init__(self, name):
        self.children_ = []
        self.name_ = name

    def add_child(self, name):
        self.children_.append(TreeNode(name))
        return self

    def breadth_first_search(self, array):

        queue = [self,]

        while (len(queue) > 0):

            current_node = queue.pop()
            array.append(current_node.name_)
            # Children come first, even in next iteration.
            for child in current_node.children_:
                queue.insert(0, child)

        return array