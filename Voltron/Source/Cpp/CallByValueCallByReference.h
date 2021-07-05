//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifndef CPP_CALL_BY_VALUE_CALL_BY_REFERENCE_H
#define CPP_CALL_BY_VALUE_CALL_BY_REFERENCE_H

namespace Cpp
{

namespace CallByValue
{

void my_function(int x);

int call_by_value(int x);

} // namespace CallByValue

namespace CallByReference
{

void my_function(int& x);

int call_by_ref(int& x);

//------------------------------------------------------------------------------
/// \details error: binding reference of type ‘int&’ to ‘const int’ discards
/// qualifiers; this is obtained at compile-time if return type is int&.
/// Qualifier const is thrown out.
//------------------------------------------------------------------------------
const int& call_by_const_ref(const int& x);

//------------------------------------------------------------------------------
/// \details error: binding reference of type ‘int&’ to ‘const int’ discards
/// qualifiers; this is obtained at compile-time if return type is int&.
/// Qualifier const is thrown out.
//------------------------------------------------------------------------------
int const& call_by_ref_const(int const& x);

} // namespace CallByReference

} // namespace Cpp

#endif // CPP_CALL_BY_VALUE_CALL_BY_REFERENCE_H