#include "DataStructures/Arrays/FixedSizeArrays.h"

#include <array>
#include <boost/test/unit_test.hpp>
#include <utility>
#include <vector>

using DataStructures::Arrays::DynamicFixedSizeArray;
using DataStructures::Arrays::FixedSizeArrayOnStack;
using std::array;
using std::move;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(FixedSizeArrays_tests)

BOOST_AUTO_TEST_SUITE(DynamicFixedSizeArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromDataFromStdVector)
{
	vector a_vec {420, 69, 42, 101};

	DynamicFixedSizeArray a {a_vec.data(), a_vec.size()};

	BOOST_TEST(a.size() == 4);
	BOOST_TEST(a[0] == 420);
	BOOST_TEST(a[1] == 69);
	BOOST_TEST(a[2] == 42);
	BOOST_TEST(a[3] == 101);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromDataFromStdArray)
{
	array<int, 4> a_array {420, 69, 42, 101};

	DynamicFixedSizeArray a {a_array.data(), a_array.size()};

	BOOST_TEST(a.size() == 4);
	BOOST_TEST(a[0] == 420);
	BOOST_TEST(a[1] == 69);
	BOOST_TEST(a[2] == 42);
	BOOST_TEST(a[3] == 101);
}

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

BOOST_AUTO_TEST_SUITE(FixedSizeArrayOnStack_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsWithBraceInitialization)
{
	FixedSizeArrayOnStack<int, 3> a {};

	BOOST_TEST(a.size() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerList)
{
	FixedSizeArrayOnStack<int, 4> a {{42, 69, 420, 101}};

	BOOST_TEST(a.size() == 4);
	BOOST_TEST(a[0] == 42);
	BOOST_TEST(a[1] == 69);
	BOOST_TEST(a[2] == 420);
	BOOST_TEST(a[3] == 101);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerListOfSmallerSize)
{
	FixedSizeArrayOnStack<int, 3> a {{42, 69}};

	BOOST_TEST(a.size() == 3);
	BOOST_TEST(a[0] == 42);
	BOOST_TEST(a[1] == 69);
}

BOOST_AUTO_TEST_SUITE_END() // FixedSizeArraysOnStack_tests

BOOST_AUTO_TEST_SUITE_END() // FixedSizeArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures