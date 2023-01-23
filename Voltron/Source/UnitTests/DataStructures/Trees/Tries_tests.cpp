#include "DataStructures/Trees/Trie.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <string>

using std::size_t;
using std::string;

using namespace DataStructures::Trees::Tries;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(Tries_tests)

BOOST_AUTO_TEST_SUITE(GeeksForGeeks_tests)

constexpr size_t alphabet_size {128};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CharValuesCanIndexArray)
{
  constexpr std::size_t alphabet_size {128};
  // cf. https://en.cppreference.com/w/c/language/array_initialization
  // Holds all 0s.
  int children[alphabet_size] {0};
  BOOST_TEST(children[0] == 0);
  BOOST_TEST(children[1] == 0);

  // cf. https://en.cppreference.com/w/cpp/language/ascii
  // Look up ASCII chart for values for chars.

  BOOST_TEST(static_cast<size_t>('-') == 45);
  BOOST_TEST(static_cast<size_t>('a') == 97);
  BOOST_TEST(static_cast<size_t>('b') == 98);
  BOOST_TEST(static_cast<size_t>('A') == 65);
  BOOST_TEST(static_cast<size_t>('B') == 66);

  const char ch45 {'-'};
  BOOST_TEST(children[static_cast<size_t>(ch45)] == 0);
  children[static_cast<size_t>(ch45)] = static_cast<int>(ch45);
  BOOST_TEST(children[static_cast<size_t>(ch45)] == 45);

  const char ch97 {'a'};
  BOOST_TEST(children[static_cast<size_t>(ch97)] == 0);
  children[static_cast<size_t>(ch97)] = static_cast<int>(ch97);
  BOOST_TEST(children[static_cast<size_t>(ch97)] == 97);

  const char ch99 {'c'};
  BOOST_TEST(children[static_cast<size_t>(ch99)] == 0);
  children[static_cast<size_t>(ch99)] = static_cast<int>(ch99);
  BOOST_TEST(children[static_cast<size_t>(ch99)] == 99);

  const char ch67 {'C'};
  BOOST_TEST(children[static_cast<size_t>(ch67)] == 0);
  children[static_cast<size_t>(ch67)] = static_cast<int>(ch67);
  BOOST_TEST(children[static_cast<size_t>(ch67)] == 67);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  GeeksForGeeks::TrieNode<128> root {};

  BOOST_TEST(true);  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertWordInsertsAWord)
{
  const string word1 {"and"};
  GeeksForGeeks::TrieNode<128>* root_ptr {new GeeksForGeeks::TrieNode<128>{}};
  GeeksForGeeks::insert<128>(root_ptr, word1);

  BOOST_TEST(root_ptr->children_[97] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')]->
      children_[static_cast<size_t>('d')] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')]->
      children_[static_cast<size_t>('d')]->is_end_of_word_);

  GeeksForGeeks::destroy(root_ptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertWordInsertWords)
{
  const string word1 {"and"};
  const string word2 {"ant"};

  GeeksForGeeks::TrieNode<128>* root_ptr {new GeeksForGeeks::TrieNode<128>{}};
  GeeksForGeeks::insert<128>(root_ptr, word1);
  GeeksForGeeks::insert<128>(root_ptr, word2);

  BOOST_TEST(root_ptr->children_[97] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')]->
      children_[static_cast<size_t>('d')] != nullptr);
  BOOST_TEST(
    root_ptr->children_[97]->children_[static_cast<size_t>('n')]->
      children_[static_cast<size_t>('t')]->is_end_of_word_);

  GeeksForGeeks::destroy(root_ptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SearchFindWords)
{
  const string word1 {"and"};
  const string word2 {"ant"};
  const string word3 {"dad"};
  const string word4 {"do"};

  GeeksForGeeks::TrieNode<alphabet_size>* root_ptr {
    new GeeksForGeeks::TrieNode<alphabet_size>{}};
  GeeksForGeeks::insert<alphabet_size>(root_ptr, word1);
  GeeksForGeeks::insert<alphabet_size>(root_ptr, word2);
  GeeksForGeeks::insert<alphabet_size>(root_ptr, word3);
  GeeksForGeeks::insert<alphabet_size>(root_ptr, word4);

  BOOST_TEST(GeeksForGeeks::search(root_ptr, word1));
  BOOST_TEST(GeeksForGeeks::search(root_ptr, word2));
  BOOST_TEST(GeeksForGeeks::search(root_ptr, word3));
  BOOST_TEST(GeeksForGeeks::search(root_ptr, word4));

  BOOST_TEST(!GeeksForGeeks::search(root_ptr, "baby"));
  BOOST_TEST(!GeeksForGeeks::search(root_ptr, "raw"));

  GeeksForGeeks::destroy(root_ptr);
}

BOOST_AUTO_TEST_SUITE_END() // GeeksForGeeks_tests

BOOST_AUTO_TEST_SUITE_END() // Tries_tests
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures
