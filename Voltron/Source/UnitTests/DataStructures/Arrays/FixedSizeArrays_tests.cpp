
#include "DataStructures/Arrays/FixedSizeArrays.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using DataStructures::Arrays::DynamicFixedSizeArray;
using std::move;

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructs)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};
	DynamicFixedSizeArray b {a};

	BOOST_TEST_REQUIRE(a.size() == 5);
	BOOST_TEST(b.size() == a.size());

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(a[i] == 5 - i);
		BOOST_TEST(b[i] == 5 - i);
	}

	BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyAssigns)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};
	DynamicFixedSizeArray b = a;

	BOOST_TEST_REQUIRE(a.size() == 5);
	BOOST_TEST(b.size() == a.size());

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(a[i] == 5 - i);
		BOOST_TEST(b[i] == 5 - i);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructs)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};
	DynamicFixedSizeArray b {move(a)};

	BOOST_TEST(a.size() == 0);
	BOOST_TEST(b.size() == 5);

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(b[i] == 5 - i);
	}

	BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssigns)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};
	DynamicFixedSizeArray b = move(a);

	BOOST_TEST(a.size() == 0);
	BOOST_TEST(b.size() == 5);

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(b[i] == 5 - i);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SizesAfterMoveConstructs)
{
	DynamicFixedSizeArray a {{5, 4, 3, 2, 1}};
	DynamicFixedSizeArray b {move(a)};

	BOOST_TEST(a.size() == 0);
	BOOST_TEST(b.size() == 5);

	for (int i {0}; i < 5; ++i)
	{
		BOOST_TEST(b[i] == 5 - i);
	}

	BOOST_TEST(true);
}


BOOST_AUTO_TEST_SUITE_END() // DynamicFixedSizeArray_tests

BOOST_AUTO_TEST_SUITE_END() // FixedSizeArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures