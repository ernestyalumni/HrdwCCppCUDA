################################################################################
# @ref https://qvault.io/python/binary-search-tree-in-python/
################################################################################

class Node:
    def __init__(self, value = None):
        self.left_ = None
        self.right_ = None
        self.value_ = value

    def insert(self, value):

        # If the node doesn't yet have a value, we can just set the given value
        # and return.
        if not self.value_:
            self.value_ = value
            return

        if self.value_ == value:
            return

        # If given value is less than our node's value and we already have a
        # left child, then we recursively call insert on left child.
        if value < self.value_:
            if self.left_:
                self.left_.insert(value)
                return

            # If we don't have a left child yet, then we just make the given
            # value our new child.
            self.left_ = Node(value)

            return

        assert value > self.value_

        if self.right_:
            self.right_.insert(value)

            return
        self.right_ = Node(value)


    def insert_with_repeat(self, value):
        if not self.value_:
            self.value_ = value
            return

        if self.value_ == value:
            if self.left_ and self.right_:
                # Recursive deal with problem.
                self.left_.insert_with_repeat(value)
                return

            elif not self.left_:
                self.left_ = Node(value)
                return

            else:
                assert not self.right_
                self.right_ = Node(value)
                return

        if value < self.value_:
            if self.left_:
                self.left_.insert_with_repeat(value)
                return

            self.left_ = Node(value)

            return

        assert value > self.value_

        if self.right_:
            self.right_.insert_with_repeat(value)

            return
        self.right_ = Node(value)


    def insert_values(self, x, without_repeat=None):
        insert_function = self.insert

        if not without_repeat:
            insert_function = self.insert_with_repeat

        for element in x:

            insert_function(element)


    def get_min(self):
        """
        @details Simple recursive function that traverse edges of tree to find
        smallest value stored therein.
        """
        current = self

        while current.left_ is not None:

            current = current.left_

        return current.value_

    def get_max(self):
        """
        @details Simple recursive function that traverse right edges of tree to
        find largest value stored therein.
        """
        current = self

        while current.right_ is not None:
            current = current.right_

        return current.value_


    def delete(self, value):

        if self == None or self.value_ == None:
            return self

        if value < self.value_:
            if self.left_:
                self.left_.delete(value)

            return self

        if value > self.value_:
            if self.right_:
                self.right_.delete(value)

            return self

        assert value == self.value_

        # Just return other subtree, other side, if one side isn't there.

        if self.right_ == None:
            return self.left_

        if self.left_ == None:
            return self.right_

        # Since both left and right subtrees exist, get the node with the
        # smallest value on the right side. Connect that node to be the next
        # value that's larger than all values in the left subtree.

        min_larger_node = self.right_

        while min_larger_node.left_:
            min_larger_node = min_larger_node.left_

        self.value_ = min_larger_node.value_

        self.right_ = self.right_.delete(min_larger_node.value_)

        return self


    def exists(self, value):

        if value == self.value_:
            return True

        if value < self.value_:

            if self.left_ == None:
                return False

            return self.left_.exists(value)

        if self.right_ == None:
            return False

        return self.right_.exists(value)


    def preorder(self, values = None):
        """
        @details
        Reach down towards the left as far as possible, while recording each
        value traversed.
        Travel back up to next right and go right.
        Then reach down towards the left as far as possible and repeat.
        """

        if values == None:
            values = []

        if self.value_ is not None:
            values.append(self.value_)

        if self.left_ is not None:
            self.left_.preorder(values)

        if self.right_ is not None:
            self.right_.preorder(values)

        return values


    def inorder(self, values = None):
        """
        @details

        Reach down towards left as far as possible. Only then record first leaf
        reached.
        Travel back and record value. Try to traverse right side the same way.

        Inorder appears to print the elements in increasing or lexicographical 
        order.
        """

        if values == None:
            values = []

        if self.left_ is not None:
            self.left_.inorder(values)

        if self.value_ is not None:
            values.append(self.value_)

        if self.right_ is not None:
            self.right_.inorder(values)

        return values


    def postorder(self, values = None):
        if values == None:
            values = []

        if self.left_ is not None:
            self.left_.postorder(values)

        if self.right_ is not None:
            self.right_.postorder(values)

        if self.value_ is not None:
            values.append(self.value_)

        return values


