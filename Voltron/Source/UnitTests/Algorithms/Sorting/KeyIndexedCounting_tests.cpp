#include "Algorithms/Sorting/KeyIndexedCounting.h"
#include "DataStructures/Arrays/DynamicArray.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <tuple> // std::get
#include <utility> // std::move

using Algorithms::Sorting::key_indexed_counting_sort;

template <typename T>
using Array = DataStructures::Arrays::PrimitiveDynamicArray<T>;

using std::get;
using std::string;
using std::tuple;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Sorting)

//------------------------------------------------------------------------------
/// See Sec. 5.1, String Sorts, pp. 703, Algorithms, 4th Ed., Robert Sedgewick,
/// Kevin Wayne
//------------------------------------------------------------------------------

struct NameAndSection : public tuple<string, int>
{
  using tuple<string, int>::tuple;

  int key() const
  {
    return std::get<1>(*this);
  }
};

Array<NameAndSection> create_name_and_section_example()
{
  Array<NameAndSection> name_and_section {};
  name_and_section.initialize({
    {"Anderson", 2},
    {"Brown", 3},
    {"Davis", 3},
    {"Garcia", 4},
    {"Harris", 1},
    {"Jackson", 3},
    {"Johnson", 4},
    {"Jones", 3},
    {"Martin", 1},
    {"Martinez", 2},
    {"Miller", 2},
    {"Moore", 1},
    {"Robinson", 2},
    {"Smith", 4},
    {"Taylor", 3},
    {"Thomas", 4},
    {"Thompson", 4},
    {"White", 2},
    {"Williams", 3},
    {"Wilson", 4}});

  BOOST_TEST_REQUIRE(name_and_section.size() == 20);
  BOOST_TEST_REQUIRE(name_and_section.capacity() == 20);
  BOOST_TEST_REQUIRE(name_and_section[0].key() == 2);
  BOOST_TEST_REQUIRE(name_and_section[1].key() == 3);
  BOOST_TEST_REQUIRE(name_and_section[19].key() == 4);
  BOOST_TEST_REQUIRE(get<0>(name_and_section[0]) == "Anderson");
  BOOST_TEST_REQUIRE(get<0>(name_and_section[1]) == "Brown");
  BOOST_TEST_REQUIRE(get<0>(name_and_section[19]) == "Wilson");

  return std::move(name_and_section);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(KeyIndexedCountingSortsByKey)
{ 
  Array<NameAndSection> sample {create_name_and_section_example()};

  const Array<NameAndSection> result {key_indexed_counting_sort<4>(sample)};

  BOOST_TEST(get<0>(result[0]) == "Harris");
  BOOST_TEST(get<0>(result[1]) == "Martin");
  BOOST_TEST(get<0>(result[2]) == "Moore");
  BOOST_TEST(get<0>(result[3]) == "Anderson");
  BOOST_TEST(get<0>(result[4]) == "Martinez");
  BOOST_TEST(get<0>(result[5]) == "Miller");
  BOOST_TEST(get<0>(result[6]) == "Robinson");
  BOOST_TEST(get<0>(result[7]) == "White");
  BOOST_TEST(get<0>(result[8]) == "Brown");
  BOOST_TEST(get<0>(result[9]) == "Davis");
  BOOST_TEST(get<0>(result[10]) == "Jackson");
  BOOST_TEST(get<0>(result[11]) == "Jones");
  BOOST_TEST(get<0>(result[12]) == "Taylor");
  BOOST_TEST(get<0>(result[13]) == "Williams");
  BOOST_TEST(get<0>(result[14]) == "Garcia");
  BOOST_TEST(get<0>(result[15]) == "Johnson");
  BOOST_TEST(get<0>(result[16]) == "Smith");
  BOOST_TEST(get<0>(result[17]) == "Thomas");
  BOOST_TEST(get<0>(result[18]) == "Thompson");
  BOOST_TEST(get<0>(result[19]) == "Wilson");

  BOOST_TEST(result[0].key() == 1);
  BOOST_TEST(result[1].key() == 1);
  BOOST_TEST(result[2].key() == 1);
  BOOST_TEST(result[3].key() == 2);
  BOOST_TEST(result[4].key() == 2);
  BOOST_TEST(result[5].key() == 2);
  BOOST_TEST(result[6].key() == 2);
  BOOST_TEST(result[7].key() == 2);
  BOOST_TEST(result[8].key() == 3);
  BOOST_TEST(result[9].key() == 3);
  BOOST_TEST(result[10].key() == 3);
  BOOST_TEST(result[11].key() == 3);
  BOOST_TEST(result[12].key() == 3);
  BOOST_TEST(result[13].key() == 3);
  BOOST_TEST(result[14].key() == 4);
  BOOST_TEST(result[15].key() == 4);
  BOOST_TEST(result[16].key() == 4);
  BOOST_TEST(result[17].key() == 4);
  BOOST_TEST(result[18].key() == 4);
  BOOST_TEST(result[19].key() == 4);
}

BOOST_AUTO_TEST_SUITE_END() // Sorting
BOOST_AUTO_TEST_SUITE_END() // Algorithms