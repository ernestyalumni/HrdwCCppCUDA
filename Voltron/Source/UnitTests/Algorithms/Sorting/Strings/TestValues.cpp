#include "DataStructures/Arrays/DynamicArray.h"
#include "TestValues.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <utility>

template <typename T>
using Array = DataStructures::Arrays::PrimitiveDynamicArray<T>;

using std::string;

namespace UnitTests
{
namespace Algorithms
{
namespace Sorting
{
namespace Strings
{

Array<string> create_license_plates_example_input()
{
  Array<string> license_plates {};

  license_plates.initialize({
    "4PGC938",
    "2IYE230",
    "3CIO720",
    "1ICK750",
    "1OHV845",
    "4JZY524",
    "1ICK750",
    "3CIO720",
    "1OHV845",
    "1OHV845",
    "2RLA629",
    "2RLA629",
    "3ATW723"});

  BOOST_TEST_REQUIRE(license_plates.size() == 13);
  BOOST_TEST_REQUIRE(license_plates[0] == "4PGC938");
  BOOST_TEST_REQUIRE(license_plates[1] == "2IYE230");
  BOOST_TEST_REQUIRE(license_plates[12] == "3ATW723");

  return std::move(license_plates);
}

} // namespace Strings

} // namespace Sorting

} // namespace Algorithms
} // namespace UnitTests
