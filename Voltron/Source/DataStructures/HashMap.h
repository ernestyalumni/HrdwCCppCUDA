//------------------------------------------------------------------------------
/// \file HashMap.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Hash Table
/// \details Hash Map is 1 of the 2 different kinds of hash tables (other is
/// hash set.)
///
//------------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_HASH_TABLES_HASH_MAP_H
#define DATA_STRUCTURES_HASH_TABLES_HASH_MAP_H

#include <algorithm> // std::copy
#include <array>
#include <cstddef> // std::size_t
#include <iterator> // std::begin, std::end;
#include <list>
#include <memory>
#include <stdexcept> // std::runtime_error
#include <utility> // std::pair

namespace DataStructures
{

namespace HashTables
{

namespace HashMap
{

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1140/discuss/161483/C++-solution-using-vector-of-lists
//------------------------------------------------------------------------------

class HashMapListVector
{
  public:

    static constexpr std::size_t default_length_ {10000};

    // Initialize your data structure here.
    HashMapListVector();

    explicit HashMapListVector(const std::size_t bucket_number);

    // Value will always be non-negative.
    void put(int key, int value);

    int hash_function(int key);

    // Returns the value to which the specified key is mapped, or -1 if this map
    // contains no mapping for the key.
    int get(int key);

    // Removes the mapping of the specified value key if this map contains a
    // mapping for the key.
    void remove(int key);

  private:

    std::vector<std::list<std::pair<int, int>>> buckets_;
    std::size_t length_;
};

/*
template<typename Entry, typename Key, typename Value>
class HashMap
{
  public:

    static constexpr std::size_t default_buckets_{10000};

    // Initialize your data structure here.
    HashMap():
      buckets_(default_buckets_),
      capacity_{default_buckets_},
      length_{0}
    {}

    std::size_t length() const
    {
      return length_;
    }

    bool is_empty() const
    {
      return length() == 0;
    }

    // Value will always be non-negative.
    void put(Key key, Value value)
    {
      std::size_t bucket_index {get_access_index(key)};
      Entry<Key, Value> head {buckets.at(bucket_index)};

      // If we can find the entry, update it. If we cannot find it, perform the
      // insertion into the bucket.

      Entry<Key, Value>* bucket_entry_found_ptr {
        get_reference_if_key_exists(head, key)};

      // Not found.
      if (bucket_entry_found_ptr == nullptr)
      {
        Entry<Key, Value> new_entry {new Entry<Key, Value>(key, value)};

        new_entry.next_ = &head;

        buckets.at(bucketIndex) = new_entry;
        ++length_;
      }
      else
      {
        bucket_entry_found_ptr->value_ = value;
      }

      // Here, we would have provision for resizing according to load factor.
    }

    Value get(Key key)
    {
      std::size_t bucket_index {get_access_index(key)};


    }


      std::vector<Entry<Key, Value>> buckets (default_buckets),



    // Returns the value to which the specified key is mapped, or -1 if this map
    // contains no mapping for the key.
    Value get(Key key);

    // Removes the mapping of the specified value key if this map contains a
    // mapping for the key.
    void remove(Key key);


  private:


};
*/


class HashMap
{
  public:

    // Number of buckets
    static constexpr int default_M {10000};

    HashMap();

    void add(int key);

    void remove(int key);

    //--------------------------------------------------------------------------
    /// \brief Returns true if this set contains the specified element.
    //--------------------------------------------------------------------------
    bool contains(int key);

    int hash(int key);

  private:

    std::array<std::list<int>, default_M> chained_bucket_array_;
};

// cf. https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1139/discuss/179164/C++-97.97-without-a-massive-array-or-using-a-map-BST

template <typename T>
struct Node
{
  T value_;

  Node<T>* left_;
  Node<T>* right_;

  Node():
    value_{static_cast<T>(0)},
    left_{nullptr},
    right_{nullptr}
  {}

  // Constructor
  Node(T value):
    value_{value},
    left_{nullptr},
    right_{nullptr}
  {}

  Node(T value, Node<T>* left_node_ptr, Node<T>* right_node_ptr):
    value_{value},
    left_{left_node_ptr},
    right_{right_node_ptr}
  {}

  ~Node()
  {
    // Do this so that we can recursively delete, i.e. traverse through
    // destructors recursively.
    // cf. https://stackoverflow.com/questions/677653/does-delete-on-a-pointer-to-a-subclass-call-the-base-class-destructor
    // When call delete on a pointer allocated by new, dtor of the object
    // pointed to will be called.
    if (left_ != nullptr)
    {
      delete left_;
    }

    if (right_ != nullptr)
    {
      delete right_;
    }
  }
};

template <typename T>
struct NodeShared
{
  T value_;

  std::shared_ptr<NodeShared<T>> left_;
  std::shared_ptr<NodeShared<T>> right_;

  NodeShared(T value):
    value_{value},
    left_{nullptr},
    right_{nullptr}
  {}

  ~NodeShared()
  {}
};

template <typename T>
class Tree
{
  public:

    Tree():
      root_{nullptr}
    {}

    Tree(Node<T>* root):
      root_{root}
    {}

    ~Tree()
    {
      if (root_ != nullptr)
      {
        // Traverses through the destructors as the destructor of Node<T> does.
        delete root_;
      }
    }

    Node<T>* find(T x, Node<T>* parent)
    {
      Node<T>* current_ptr {parent};

      while (current_ptr != nullptr)
      {
        if (x < current_ptr->value_)
        {
          current_ptr = current_ptr->left_;
        }
        else if (current_ptr->value_ < x)
        {
          current_ptr = current_ptr->right_;
        }
        else
        {
          return current_ptr;
        }
      }

      // Could not find the value x despite traversing the binary search tree.
      // current_ptr must be nullptr, otherwise the while loop can't be exited
      // out.
      return current_ptr;
    }

    Node<T>* find(T x)
    {
      if (root_ != nullptr)
      {
        return find(x, root_);
      }

      return nullptr;
    }

    Node<T>* insert(T x, Node<T>* parent)
    {
      Node<T>* result {parent};

      // Base case for insertion.
      if (parent == nullptr)
      {
        result = new Node<T>(x);
      }
      // Put to the left of the parent. If there already is a left, do
      // recursion.
      else if (x < parent->value_)
      {
        // The next insert gets exited once a recursion branch gets to nullptr,
        // a leaf. Then the modified left subtree gets reattached to the left
        // for left ptr.
        parent->left_ = insert(x, parent->left_);
      }
      else if (x > parent->value_)
      {
        parent->right_ = insert(x, parent->right_);
      }
      else
      {
        return nullptr; // duplicate; insertion shouldn't comply.
      }

      return result;
    }

  bool insert(T x)
  {
    // Don't insert a value that's already in there.
    if (find(x) != nullptr)
    {
      return false;
    }

    root_ = insert(x, root_);

    return true;
  }

  // Finds the inorder successor to parent. In order words, also does inorder
  // traversal for one step.
  Node<T>* find_min(Node<T>* parent)
  {
    Node<T>* current_ptr {parent};
    Node<T>* previous_ptr {nullptr};

    while (current_ptr != nullptr)
    {
      previous_ptr = current_ptr;
      current_ptr = current_ptr->left_;
    }

    // Found a leaf, and so we go back to the previous ptr.
    return previous_ptr;
  }

  Node<T>* remove_min(Node<T>* parent)
  {
    // Base case. Nothing to remove. No elements at all.
    if (parent == nullptr)
    {
      return nullptr;
    }
    else if (parent->left_ != nullptr)
    {
      parent->left_ = remove_min(parent->left_);
      return parent;
    }
    // min node has a right subtree that we'll need to preserve.
    else
    {
      Node<T>* result {parent->right_};

      parent->right_ = nullptr;
      parent->left_ = nullptr;

      delete parent;

      return result;
    }
  }

  // https://www.youtube.com/watch?v=puyl7MBqPIg
  // Binary Search Tree | Set 2 (Delete) | GeeksforGeeks 
  Node<T>* remove(T x, Node<T>* parent)
  {
    Node<T>* current_ptr {parent};

    // Not found, if reached here.
    if (current_ptr == nullptr)
    {
      return current_ptr;
    }

    if (x < current_ptr->value_)
    {
      // Find x and remove it in the left subtree.
      current_ptr->left_ = remove(x, current_ptr->left_);
    }
    else if (x > current_ptr->value_)
    {
      current_ptr->right_ = remove(x, current_ptr->right_);
    }
    // x value is found. There are 3 different cases to consider:
    // 1. Node to be deleted is a leaf.
    // 2. Node to be deleted has only 1 child.
    // 3. Node to be deleted has 2 children.
    //
    // If it's a leaf, return nullptr.
    // If it has one child, return that subtree.
    else
    {
      // Node with only 1 child or no child.
      if (current_ptr->left_ == nullptr)
      {
        Node<T>* right {current_ptr->right_};
        delete current_ptr;
        // If right == nullptr, then return nullptr, since leaf was deleted.
        return right;
      }
      // Node with only 1 left child.
      else if (current_ptr->right_ == nullptr)
      {
        Node<T>* left {current_ptr->left_};
        delete current_ptr;
        return left;
      }
      // 3. Node to be deleted has 2 children.
      // e.g. root node with a left and right subtree.
      // First, find inorder successor of the node.
      // Now copy contents of inorder successor to node and delete inorder
      // successor.
      else
      {
        // Find the inorder successor of the current node.
        Node<T>* successor {find_min(current_ptr->right_)};

        // Now copy contents of inorder successor to the node and delete the
        // inorder successor.
        current_ptr->value_ = successor->value_;

        // Notice that successor is either a leaf or node with only right
        // subtree.
        current_ptr->right_ = remove(successor->value_, current_ptr->right_);
      }
    }

    return current_ptr;
  }

  bool remove(T x)
  {
    if (find(x) == nullptr)
    {
      return false;
    }

    root_ = remove(x, root_);
    return true;  
  }

  private:

    Node<T>* root_;
};

template <typename T>
class HashSetT
{
  public:

    // Initialize your data structure here.
    HashSetT()
    {}

    void add(T key)
    {
      tree_.insert(key);
    }

    void remove(T key)
    {
      tree_.remove(key);
    }

    // Returns true if this set contains the specified element.
    bool contains(T key)
    {
      return tree_.find(key) != nullptr;
    }

  private:

    Tree<T> tree_;

};

} // namespace HashMap
} // namespace HashTables
} // namespace DataStructures

#endif // DATA_STRUCTURES_HASH_TABLES_HASH_MAP_H