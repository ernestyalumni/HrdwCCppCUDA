//------------------------------------------------------------------------------
/// \file List.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating list as an Abstract Data Type.
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LISTS_LIST_H
#define DATA_STRUCTURES_LISTS_LIST_H

namespace DataStructures
{
namespace Lists
{

//-----------------------------------------------------------------------------
/// \brief List ADT
//-----------------------------------------------------------------------------
template <typename E>
class List
{


  private:

    //--------------------------------------------------------------------------
    /// \brief Protect assignment.
    //--------------------------------------------------------------------------
    void operator=(const List&)
    {}

    //--------------------------------------------------------------------------
    /// \brief Protect copy ctor.
    //--------------------------------------------------------------------------
    List(const List&)
    {}

};



} // namespace Lists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LISTS_LISTS_H