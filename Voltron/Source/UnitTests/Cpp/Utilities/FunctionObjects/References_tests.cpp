//------------------------------------------------------------------------------
/// \file References_tests.cpp
/// \brief Unit tests demonstrating std::ref, std::cref, std::reference_wrapper
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/utility/functional/reference_wrapper
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <functional>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FunctionObjects)
BOOST_AUTO_TEST_SUITE(ReferenceWrapper_tests)

//------------------------------------------------------------------------------
/// \details 
/// template <class T>
/// class reference_wrapper;
///
/// std::reference_wrapper is a class template that wraps a reference in a
/// copyable, assignable object. It's frequently used as a mechanism to store
/// references inside standard containers (like std::vector) which can't
/// normally hold references.
///
/// Specifically, std::reference_wrapper is a CopyConstructible, CopyAssignable
/// wrapper around a reference to object or reference to function of type T.
/// Instances of 
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstruction)
{
  
}

BOOST_AUTO_TEST_SUITE_END() // ReferenceWrapper_tests
BOOST_AUTO_TEST_SUITE_END() // FunctionObjects
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp