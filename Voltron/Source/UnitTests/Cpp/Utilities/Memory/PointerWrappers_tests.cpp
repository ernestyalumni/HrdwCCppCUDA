//------------------------------------------------------------------------------
/// \file PointerWrappers_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "Cpp/Utilities/Memory/PointerWrappers.h"

#include <boost/test/unit_test.hpp>
//#include <iostream>
#include <memory>
#include <tuple>

using Cpp::Utilities::Memory::WrappedUniquePtr;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Memory)
BOOST_AUTO_TEST_SUITE(PointerWrappers_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstruction)
{
	WrappedUniquePtr<int> u_ptr {};
	BOOST_TEST(u_ptr.get_object() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithConstructNullPtrTag)
{
	WrappedUniquePtr<double> u_ptr {WrappedUniquePtr<double>::ConstructNullPtr{}};
	BOOST_TEST(!u_ptr.owns_object());
}

// Corresponds to the ctor
// explicit WrapperUniquePtr(T&& input)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithIntAsRValue)
{
	WrappedUniquePtr u_ptr {5};
	BOOST_TEST(u_ptr.get_object() == 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OwnsObjectChecksIfThereIsAnAssociatedManagedObject)
{
	WrappedUniquePtr u_ptr {5};
	BOOST_TEST_REQUIRE(u_ptr.get_object() == 5);
	BOOST_TEST(u_ptr.owns_object());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReleaseReleasesOwnership)
{
	WrappedUniquePtr u_ptr {42};
	int* int_ptr {u_ptr.release()};
	BOOST_TEST(!u_ptr.owns_object());
	BOOST_TEST((*int_ptr == 42));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LinkNewResetsManagedObjectToTargetObject)
{
	WrappedUniquePtr u_ptr {42};
	int* int_ptr {u_ptr.release()};
	BOOST_TEST_REQUIRE(!u_ptr.owns_object());
	BOOST_TEST_REQUIRE((*int_ptr == 42));

	// Link the pointer to the released object to a new WrappedUniquePtr.
	WrappedUniquePtr<int> u_ptr2 {WrappedUniquePtr<int>::ConstructNullPtr{}};
	BOOST_TEST_REQUIRE(!u_ptr2.owns_object());
	u_ptr2.link_new(int_ptr);

	// Now the new instance owns an object through its unique pointer.
	BOOST_TEST(u_ptr2.owns_object());
	BOOST_TEST(u_ptr2.get_object() == 42);
}

BOOST_AUTO_TEST_SUITE_END() // PointerWrappers_tests
BOOST_AUTO_TEST_SUITE_END() // Memory
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp