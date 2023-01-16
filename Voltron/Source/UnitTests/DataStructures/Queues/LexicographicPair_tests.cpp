#include "DataStructures/Queues/LexicographicPair.h"

#include <boost/test/unit_test.hpp>
#include <cstddef>

using DataStructures::Queues::DWHarder::LexicographicPair;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(LexicographicPair_tests)

template <typename T>
class TestLexicographicPair : public LexicographicPair<T>
{
  public:

    using LexicographicPair<T>::LexicographicPair;
    using LexicographicPair<T>::count_priority_;
    using LexicographicPair<T>::count_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StaticCountStartsAtZero)
{
  BOOST_TEST(TestLexicographicPair<int>::count_ == 0);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  TestLexicographicPair<int> tlp {42};

  BOOST_TEST(tlp.priority_ == 42);
  BOOST_TEST(tlp.count_priority_ == 0);
  BOOST_TEST(tlp.count_ == 1);
  BOOST_TEST(TestLexicographicPair<int>::count_ == 1);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SecondConstructedObjectHasHigherCount)
{
  TestLexicographicPair<int> tlp1 {42};
  BOOST_TEST(tlp1.priority_ == 42);
  BOOST_TEST(tlp1.count_priority_ == 1);
  BOOST_TEST(tlp1.count_ == 2);
  BOOST_TEST(TestLexicographicPair<int>::count_ == 2);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);

  TestLexicographicPair<int> tlp2 {42};
  BOOST_TEST(tlp2.priority_ == 42);
  BOOST_TEST(tlp2.count_priority_ == 2);
  BOOST_TEST(tlp2.count_ == 3);
  BOOST_TEST(TestLexicographicPair<int>::count_ == 3);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);

  // Let's test each user-defined comparison operator.
  BOOST_TEST(!(tlp1 == tlp2));
  BOOST_TEST((tlp1 != tlp2));
  BOOST_TEST((tlp1 < tlp2));
  BOOST_TEST((tlp1 <= tlp2));
  BOOST_TEST(!(tlp1 > tlp2));
  BOOST_TEST(!(tlp1 >= tlp2));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComparisonChecksPriorityFirst)
{
  TestLexicographicPair<int> tlp1 {69};
  BOOST_TEST(tlp1.priority_ == 69);
  BOOST_TEST(tlp1.count_priority_ == 3);
  BOOST_TEST(tlp1.count_ == 4);
  BOOST_TEST(TestLexicographicPair<int>::count_ == 4);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);

  TestLexicographicPair<int> tlp2 {42};
  BOOST_TEST(tlp2.priority_ == 42);
  BOOST_TEST(tlp2.count_priority_ == 4);
  BOOST_TEST(tlp2.count_ == 5);
  BOOST_TEST(TestLexicographicPair<int>::count_ == 5);
  BOOST_TEST(TestLexicographicPair<std::size_t>::count_ == 0);

  // Let's test each user-defined comparison operator.
  BOOST_TEST((!(tlp1 == tlp2)));
  BOOST_TEST((tlp1 != tlp2));
  BOOST_TEST(!(tlp1 < tlp2));
  BOOST_TEST(!(tlp1 <= tlp2));
  BOOST_TEST((tlp1 > tlp2));
  BOOST_TEST((tlp1 >= tlp2));
}

BOOST_AUTO_TEST_SUITE_END() // LexicographicPair_tests
BOOST_AUTO_TEST_SUITE_END() // Queues
BOOST_AUTO_TEST_SUITE_END() // DataStructures