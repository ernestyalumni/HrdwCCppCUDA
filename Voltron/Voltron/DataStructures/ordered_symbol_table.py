################################################################################
## @ref https://introcs.cs.princeton.edu/python/44st/bst.py.html
################################################################################

class OrderedSymbolTable:
    """
    @brief An OrderedSymbolTable object is a collection of key value pairs that
    is kept in order by key. This implementation uses a binary search tree.
    """

    class _Node:
        """
        @brief A _Node object references a key, a value, and left and right
        children _Node objects. An OrderedSymTable object is composed of _Node
        objects.
        """

        def __init__(self, key, value):
            # Reference to key.
            self.key = key 
            # Reference to value
            self.value = value
            # Reference to left child _Node object
            self.left = None
            # Reference to right child _Node object
            self.right = None


    def __init__(self):
        """
        @brief Construct a new OrderedSymbolTable object.
        """

        self._root = None # Reference to root _Node object


    def _get(self, x, key):
        """
        @brief Search the subtree of self whose root is x for a _Node object
        with the given key. If found return that _Node object's value; otherwise
        raise a KeyError.
        """

        if x is None:
            raise KeyError

        if key < x.key:
            
            return self._get(x.left, key)

        elif x.key < key:

            return self._get(x.right, key)

        else:

            assert x.key == key

            return x.value


    def __getitem__(self, key):
        """
        @brief Return the value associated with key in self

        e.g.

        ordered_symbol_table['key']
        """

        return self._get(self._root, key)


    def _set(self, x, key, value):
        """
        @details x is the root of a subtree self. If a _Node object with the
        given key exists in that subtree, then set its value to value. Otherwise
        insert a new _Node object consisting of the given key and value into the
        subtree. Return the root of the resulting subtree.
        """

        if x is None:

            return OrderedSymbolTable._Node(key, value)

        if key < x.key:
            x.left = self._set(x.left, key, value)

        elif x.key < key:
            x.right = self._set(x.right, key, value)

        else:

          assert x.key == key

          x.value = value

        return x


    def __setitem__(self, key, value):
        """
        @brief Associate key with value in self.

        e.g.

        ordered_symbol_table['key'] = some_value
        """

        self._root = self._set(self._root, key, value)


    def _contains(self, x, key):
        """
        @brief Search the subtree of self whose root is x for a _Node object
        with the given key. If found, return True; otherwise return FAlse.
        """
        if x is None:

            return False

        if key < x.key:

            return self._contains(x.left, key)

        if x.key < key:

            return self._contains(x.right, key)

        assert x.key == key

        return True


    def __contains__(self, key):
        """
        @brief Return True if key is in self, and False otherwise.

        e.g. 'key' in ordered_symbol_table
        """

        return self._contains(self._root, key)


    def _inorder(self, x, a = []):
        """
        @brief Populate list a with all keys in the subtree of self whose root
        is x.
        """

        if x is None:
            return

        self._inorder(x.left, a)

        a += [x.key]

        self._inorder(x.right, a)

    def __iter__(self):
        """
        @brief Return an iterator for Symbolic Table object self.
        """
        a = []
        self._inorder(self._root, a)

        # iter() function creates an object which can be iterated one element at
        # a time.

        return iter(a)
