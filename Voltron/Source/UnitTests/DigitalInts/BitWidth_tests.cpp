//------------------------------------------------------------------------------
// \file BitWidth_tests.cpp
//------------------------------------------------------------------------------
#include "DigitalInts/BitWidth.h"

#include <boost/test/unit_test.hpp>
#include <type_traits>

using DigitalInts::UInt32BitWidth;

BOOST_AUTO_TEST_SUITE(DigitalInts)
BOOST_AUTO_TEST_SUITE(BitWidth_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasDefaultConstructor)
{
	BOOST_TEST(std::is_default_constructible<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasCopyConstructor)
{
	BOOST_TEST(std::is_copy_constructible<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasMoveConstructor)
{
	BOOST_TEST(std::is_move_constructible<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasCopyAssignment)
{
	BOOST_TEST(std::is_copy_assignable<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasMoveAssignment)
{
	BOOST_TEST(std::is_move_assignable<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NoVirtualDestructor)
{
	BOOST_TEST(!std::has_virtual_destructor<UInt32BitWidth<8>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifdef FORCE_COMPILE_ERRORS // bit size errors
BOOST_AUTO_TEST_CASE(BitsGreaterThan32Assert)
{
	UInt32BitWidth<33>{};

	BOOST_TEST(true);
}
#endif // FORCE_COMPILE_ERRORS




BOOST_AUTO_TEST_SUITE_END() // BitWidth_tests
BOOST_AUTO_TEST_SUITE_END() // DigitalInts