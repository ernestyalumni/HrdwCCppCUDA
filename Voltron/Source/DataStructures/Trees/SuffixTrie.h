#ifndef DATA_STRUCTURES_TREES_SUFFIX_TRIE_H
#define DATA_STRUCTURES_TREES_SUFFIX_TRIE_H

#include <cstddef>
#include <string>
#include <unordered_map>

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

//------------------------------------------------------------------------------
/// \brief Suffix Trie Construction
//------------------------------------------------------------------------------
class TrieNode
{
  public:

    std::unordered_map<char, TrieNode*> children_;
};

class SuffixTrie
{
  public:

    SuffixTrie(std::string str);

    ~SuffixTrie();

    //--------------------------------------------------------------------------
    /// \details O(N^2) time complexity since we must iterate over every
    /// substring of a string, where N is the length of an input string.
    /// O(N^2) space for how we populating the trie.
    //--------------------------------------------------------------------------
    void populate_suffix_trie_from(std::string str);

    //--------------------------------------------------------------------------
    /// \details O(m) time | O(1) space.
    //--------------------------------------------------------------------------
    bool contains(std::string str);

    TrieNode* root_;
    char end_symbol_;

  protected:

    //--------------------------------------------------------------------------
    /// \brief Helper function if one wants to refactor
    /// populate_suffix_trie_from.
    //--------------------------------------------------------------------------
    void insert_substring_starting_at(const std::size_t i, std::string& str);

  private:

    //--------------------------------------------------------------------------
    /// \details Do post-order depth-first traversal to get the leaves and visit
    /// the children first.
    //--------------------------------------------------------------------------
    static void destroy(TrieNode* node_ptr);
};

} // namespace ExpertIO

} // namespace SuffixTries

} // namespace Tries

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_SUFFIX_TRIE_H