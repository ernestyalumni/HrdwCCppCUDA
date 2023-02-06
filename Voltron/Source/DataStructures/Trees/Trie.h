#ifndef DATA_STRUCTURES_TREES_TRIE_H
#define DATA_STRUCTURES_TREES_TRIE_H

#include <cstddef>
#include <string>

namespace DataStructures
{
namespace Trees
{
namespace Tries
{

namespace GeeksForGeeks
{

//------------------------------------------------------------------------------
/// cf. https://www.geeksforgeeks.org/trie-insert-and-search/
/// \details If we store keys (i.e. values) in a binary search tree, the time
/// needed is M log(N), where M is the maximum string length and N is number of
/// keys in the tree. This is because let N = 26 letters in the alphabet. For
/// each character in a word or string, there are M characters in a word. And
/// for each letter, you decide if it's one of 26 or N letters.
///
/// For trie, a word can be searched in O(M) time. Penalty is storage.
///
/// If you use TrieNode directly, keep in mind user will have to do the garbage
/// collection, destruction, manually.
//------------------------------------------------------------------------------
template <std::size_t ALPHABET_SIZE>
class TrieNode
{
  public:

    TrieNode():
      children_{nullptr},
      is_end_of_word_{false}
    {}

    TrieNode* children_[ALPHABET_SIZE];

    bool is_end_of_word_;

    template <std::size_t M>
    friend void insert(TrieNode<M>* root, const std::string& word);

    //--------------------------------------------------------------------------
    /// \details O(M) time complexity where M is the length of the largest word.
    //--------------------------------------------------------------------------
    template <std::size_t N>
    friend bool search(const TrieNode<N>* root, const std::string& word);

    //--------------------------------------------------------------------------
    /// \brief Do a post-order, depth-first search to properly delete nodes.
    /// It's up to the user to apply this function on the root.
    //--------------------------------------------------------------------------
    template <std::size_t P>
    friend void destroy(TrieNode<P>* node_ptr);
};

template <std::size_t M>
void insert(TrieNode<M>* root, const std::string& word)
{
  TrieNode<M>* current_ptr {root};

  for (std::size_t i {0}; i < word.length(); ++i)
  {
    const std::size_t index {static_cast<std::size_t>(word[i])};

    if (current_ptr->children_[index] == nullptr)
    {
      current_ptr->children_[index] = new TrieNode<M>{};
    }

    current_ptr = current_ptr->children_[index];
  }

  // Mark the last child to be the end of the word, as a leaf.
  current_ptr->is_end_of_word_ = true;
}

template <std::size_t N>
bool search(const TrieNode<N>* root_ptr, const std::string& word)
{
  const TrieNode<N>* current_ptr {root_ptr};
  for (std::size_t i {0}; i < word.length(); ++i)
  {
    const std::size_t index {static_cast<std::size_t>(word[i])};

    if (current_ptr->children_[index] == nullptr)
    {
      return false;
    }

    current_ptr = current_ptr->children_[index];
  }

  return current_ptr->is_end_of_word_;
}

template <std::size_t P>
void destroy(TrieNode<P>* node_ptr)
{
  if (node_ptr == nullptr)
  {
    return;
  }

  for (std::size_t i {0}; i < P; ++i)
  {
    if (node_ptr->children_[i] != nullptr)
    {
      destroy(node_ptr->children_[i]);
    }
  }

  delete node_ptr;
}

} // namespace GeeksForGeeks

//------------------------------------------------------------------------------
/// \details N is the alphabet size.
//------------------------------------------------------------------------------
template <std::size_t N>
class Trie
{
  public:

    class Node
    {
      public:

        Node():
          children_{nullptr},
          is_end_of_word_{false}
        {}

        Node* children_[N];
        bool is_end_of_word_;
    };

    Trie():
      root_ptr_{new Node({})}
    {}

    ~Trie()
    {
      destroy(root_ptr_);
    }

    Node* get_root_ptr() const
    {
      return root_ptr_;
    }

    void insert(const std::string& word)
    {
      Node* current_ptr {root_ptr_};

      for (char ch: word)
      {
        const std::size_t index {static_cast<std::size_t>(ch)};

        if (current_ptr->children_[index] == nullptr)
        {
          current_ptr->children_[index] = new Node({});
        }

        current_ptr = current_ptr->children_[index];
      }

      current_ptr->is_end_of_word_ = true;
    }

    bool search(const std::string& word)
    {
      const Node* current_ptr {root_ptr_};
      for (char ch: word)
      {
        const std::size_t index {static_cast<std::size_t>(ch)};

        if (current_ptr->children_[index] == nullptr)
        {
          return false;
        }

        current_ptr = current_ptr->children_[index];
      }

      return current_ptr->is_end_of_word_;
    }

    bool do_partial_search(const std::string& substring)
    {
      const Node* current_ptr {root_ptr_};
      for (char ch: substring)
      {
        const std::size_t index {static_cast<std::size_t>(ch)};

        if (current_ptr->children_[index] == nullptr)
        {
          return false;
        }

        current_ptr = current_ptr->children_[index];
      }

      return true;
    }

  private:

    void destroy(Node* node_ptr)
    {
      if (node_ptr == nullptr)
      {
        return;
      }

      for (std::size_t i {0}; i < N; ++i)
      {
        if (node_ptr->children_[i] != nullptr)
        {
          destroy(node_ptr->children_[i]);
        }
      }

      delete node_ptr;
    }

    Node* root_ptr_;
};

} // namespace Tries

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_TRIE_H