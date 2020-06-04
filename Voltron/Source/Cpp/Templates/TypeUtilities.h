//------------------------------------------------------------------------------
/// \file TypeUtilities.h
/// \author 
/// \brief 
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-11/contained-type/type_utils.h
///-----------------------------------------------------------------------------
#ifndef CPP_TEMPLATES_TYPE_UTILITIES_H
#define CPP_TEMPLATES_TYPE_UTILITIES_H

#include <iterator> // std::begin;
#include <type_traits> // std::remove_cv_t
#include <utility> // std::declval
//#include <vector>

namespace Cpp
{
namespace Templates
{
namespace TypeUtilities
{

// cf. https://en.cppreference.com/w/cpp/types/remove_cv
// template <class T>
// struct remove_cv
// provides member typedef type which is same as T, except that its topmost
// cv-qualifiers are removed.
// remove_cv - removes topmost const, topmost volatile, or both, if present.
//
// cf. https://en.cppreference.com/w/cpp/utility/declval
// declval - converts any type T to a reference type, making it possible to use
// member functions in decltype expressions without need to go through ctors.
//
// cf. https://en.cppreference.com/w/cpp/types/remove_reference
// template <class T>
// struct remove_reference
// If type T is a reference type, provides member typedef type which is the type
// referred to by T. Otherwise type is T.

// Meta-function that returns the type of an element in an iterable collection.
template <typename T>
using ContainedType =
  std::remove_cv_t<
    std::remove_reference_t<decltype(*std::begin(std::declval<T>()))>
    >;

// Meta-function that returns a type with references stripped


} // TypeUtilities
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_TYPE_UTILITIES_H