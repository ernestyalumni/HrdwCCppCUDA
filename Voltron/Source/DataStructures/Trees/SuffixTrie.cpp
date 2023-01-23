#include "SuffixTrie.h"

#include <cstddef>
#include <string>

using std::string;

namespace DataStructures
{
namespace Trees
{
namespace Tries
{

namespace SuffixTries
{

namespace ExpertIO
{

SuffixTrie::SuffixTrie(string str):
  root_{new TrieNode()},
  end_symbol_{'*'}
{
  populate_suffix_trie_from(str);
}

SuffixTrie::~SuffixTrie()
{
  destroy(root_);
}

void SuffixTrie::populate_suffix_trie_from(string str)
{
  for (std::size_t i {0}; i < str.length(); ++i)
  {
    TrieNode* current_ptr {root_};

    for (std::size_t j {i}; j < str.length(); ++j)
    {
      if (current_ptr->children_.find(str[j]) == current_ptr->children_.end())
      {
        current_ptr->children_.emplace(str[j], new TrieNode{});
      }
      
      current_ptr = current_ptr->children_[str[j]];
    }

    current_ptr->children_.emplace(end_symbol_, nullptr);
  }
}

bool SuffixTrie::contains(string str)
{
  TrieNode* current_ptr {root_};

  for (char& c: str)
  {
    if (current_ptr->children_.find(c) == current_ptr->children_.end())
    {
      return false;
    }
    else
    {
      current_ptr = current_ptr->children_[c];
    }
  }

  return current_ptr->children_.find(end_symbol_) !=
    current_ptr->children_.end();
}

void SuffixTrie::insert_substring_starting_at(const std::size_t i, string& str)
{
  TrieNode* node {root_};

  for (std::size_t j {i}; j < str.length(); ++j)
  {
    char letter {str[j]};

    if (node->children_.find(letter) == node->children_.end())
    {
      TrieNode* new_node {new TrieNode()};

      node->children_.insert({letter, new_node});
    }

    node = node->children_[letter];
  }

  node->children_.insert({end_symbol_, nullptr});
}

void SuffixTrie::destroy(TrieNode* node_ptr)
{
  if (node_ptr == nullptr)
  {
    return;
  }

  for (auto& child : node_ptr->children_)
  {
    if (child.first != '*')
    {
      destroy(child.second);
    }
  }

  delete node_ptr;
}

} // namespace ExpertIO

} // namespace SuffixTries

} // namespace Tries

} // namespace Trees
} // namespace DataStructures