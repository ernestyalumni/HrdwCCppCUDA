//------------------------------------------------------------------------------
/// \file TypeUtilities.h
/// \author 
/// \brief 
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-11/contained-type/type_utils.h
///-----------------------------------------------------------------------------

#ifndef _CPP_TEMPLATES_TYPE_UTILITIES_H_
#define _CPP_TEMPLATES_TYPE_UTILITIES_H_

namespace Cpp
{
namespace Templates
{
namespace TypeUtilities
{

// Meta-function that returns the type of an element in an iterable collection.
template <typename T>
using ContainedType =
  std::remove_cv_t<
    std::remove_reference_t<decltype(*std::begin(std::declval<T>()))>
    >

// Meta-function that returns a type with references stripped


} // TypeUtilities
} // namespace Templates
} // namespace Cpp

