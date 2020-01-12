# @name hash_table.py
# @ref https://classroom.udacity.com/courses/ud513/lessons/7118294395/concepts/79034242780923

"""Write a HashTable class that stores strings
in a hash table, where keys are calculated
using the first two letters of the string."""

class HashTable(object):
    def __init__(self):
        self.table = [None]*10000

    def store(self, string):
        """Input a string that's stored in 
        the table."""

        hash_value = self.calculate_hash_value(string)

        if not self.table[hash_value]:
            self.table[hash_value] = [string,]
        else:
            self.table[hash_value].append(string)

        pass

    def lookup(self, string):
        """Return the hash value if the
        string is already in the table.
        Return -1 otherwise."""
        hash_value = self.calculate_hash_value(string)        
        try:
            if string in self.table[hash_value]:
                return hash_value
        except TypeError:
            return -1

        return -1

    def calculate_hash_value(self, string):
        """Helper function to calulate a
        hash value from a string."""
        if (len(string) > 1):

            ascii_value_1 = ord(string[0])
            ascii_value_2 = ord(string[1])
            return (ascii_value_1 * 100) + ascii_value_2

        return -1
