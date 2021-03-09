
#include "DataStructures/Arrays/FixedSizeArrays.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Arrays::DynamicFixedSizeArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(FixedSizeArrays_tests)

BOOST_AUTO_TEST_SUITE(DynamicFixedSizeArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerList)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};

	BOOST_TEST(a.size() == 5);

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(a[i] == 5 - i);
	}
}

BOOST_AUTO_TEST_SUITE_END() // DynamicFixedSizeArray_tests

BOOST_AUTO_TEST_SUITE_END() // FixedSizeArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures