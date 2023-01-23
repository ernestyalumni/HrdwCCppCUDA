#include "DataStructures/Trees/SuffixTrie.h"

#include <boost/test/unit_test.hpp>

using namespace DataStructures::Trees::Tries::SuffixTries;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(Tries)

BOOST_AUTO_TEST_SUITE(SuffixTrie_tests)

BOOST_AUTO_TEST_SUITE(ExpertIO_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SuffixTrieConstructsWithString)
{
  ExpertIO::SuffixTrie st {"babc"};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContainsFindsSubstringForSuffixTrie)
{
  {
    ExpertIO::SuffixTrie st {"babc"};
    BOOST_TEST(st.contains("abc"));
  }
  {
    ExpertIO::SuffixTrie st {"test"};
    BOOST_TEST(st.contains("t"));
    BOOST_TEST(st.contains("st"));
    BOOST_TEST(st.contains("est"));
    BOOST_TEST(st.contains("test"));
    BOOST_TEST(!st.contains("tes"));
  }
  {
    ExpertIO::SuffixTrie st {"invisible"};
    BOOST_TEST(st.contains("e"));
    BOOST_TEST(st.contains("le"));
    BOOST_TEST(st.contains("ble"));
    BOOST_TEST(st.contains("ible"));
    BOOST_TEST(st.contains("sible"));
    BOOST_TEST(st.contains("isible"));
    BOOST_TEST(st.contains("visible"));
    BOOST_TEST(st.contains("nvisible"));
    BOOST_TEST(st.contains("invisible"));
    BOOST_TEST(!st.contains("nvisibl"));
  }
  {
    ExpertIO::SuffixTrie st {"1234556789"};
    BOOST_TEST(st.contains("9"));
    BOOST_TEST(st.contains("89"));
    BOOST_TEST(st.contains("789"));
    BOOST_TEST(st.contains("6789"));
    BOOST_TEST(st.contains("56789"));
    BOOST_TEST(!st.contains("456789"));
    BOOST_TEST(!st.contains("3456789"));
    BOOST_TEST(!st.contains("23456789"));
    BOOST_TEST(!st.contains("123456789"));
    BOOST_TEST(!st.contains("45567"));
  }
  {
    ExpertIO::SuffixTrie st {"testtest"};
    BOOST_TEST(st.contains("t"));
    BOOST_TEST(st.contains("st"));
    BOOST_TEST(st.contains("est"));
    BOOST_TEST(st.contains("test"));
    BOOST_TEST(st.contains("ttest"));
    BOOST_TEST(st.contains("sttest"));
    BOOST_TEST(st.contains("esttest"));
    BOOST_TEST(st.contains("testtest"));
    BOOST_TEST(!st.contains("tt"));
  }
  {
    ExpertIO::SuffixTrie st {"ttttttttt"};
    BOOST_TEST(st.contains("t"));
    BOOST_TEST(st.contains("tt"));
    BOOST_TEST(st.contains("ttt"));
    BOOST_TEST(st.contains("tttt"));
    BOOST_TEST(st.contains("ttttt"));
    BOOST_TEST(st.contains("tttttt"));
    BOOST_TEST(st.contains("ttttttt"));
    BOOST_TEST(st.contains("tttttttt"));
    BOOST_TEST(st.contains("ttttttttt"));
    BOOST_TEST(!st.contains("vvv"));
  }
}

BOOST_AUTO_TEST_SUITE_END() // ExpertIO_tests

BOOST_AUTO_TEST_SUITE_END() // SuffixTrie_tests

BOOST_AUTO_TEST_SUITE_END() // Tries
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures
