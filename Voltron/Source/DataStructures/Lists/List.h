//------------------------------------------------------------------------------
/// \file List.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating list as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.1. The ADT for a list.
/// \ref https://github.com/OpenDSA/OpenDSA/blob/master/SourceCode/C%2B%2B_Templates/Lists/List.h
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
template <typename T>
class List
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default constructor.
    //--------------------------------------------------------------------------
    List()
    {}    

    //--------------------------------------------------------------------------
    /// \brief Base constructor.
    //--------------------------------------------------------------------------
    virtual ~List() = default;
    // {} // Previous implementation:

    //--------------------------------------------------------------------------
    /// \brief Clear contents from the list, to make it empty.
    //--------------------------------------------------------------------------
    virtual void clear() = 0;

    //--------------------------------------------------------------------------
    /// \brief Insert an element at the current location.
    /// item: The element to be inserted.
    //--------------------------------------------------------------------------
    virtual void insert(const T& item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Append an element at the end of the list.
    /// item: The element to be appended.
    //--------------------------------------------------------------------------
    virtual void append(const T& item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Remove and return the current element.
    /// \return: the element that was removed.
    //--------------------------------------------------------------------------
    virtual T remove() = 0;

    //--------------------------------------------------------------------------
    /// \brief Set the current position to the start of the list.
    //--------------------------------------------------------------------------
    virtual void move_to_start() = 0;

    //--------------------------------------------------------------------------
    /// \brief Set the current position to the end of the list.
    //--------------------------------------------------------------------------
    virtual void move_to_end() = 0;

    //--------------------------------------------------------------------------
    /// \brief Move the current position one step left. No change if already at
    /// beginning.
    //--------------------------------------------------------------------------
    virtual void previous() = 0;

    //--------------------------------------------------------------------------
    /// \brief Move the current position one step right. No change if already at
    /// end.
    //--------------------------------------------------------------------------
    virtual void next() = 0;

    //--------------------------------------------------------------------------
    /// \brief Return: The number of elements in the list.
    //--------------------------------------------------------------------------
    virtual int length() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Return: The position of the current element.
    //--------------------------------------------------------------------------
    virtual int current_position() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Set current position.
    /// pos: The position to make current.
    //--------------------------------------------------------------------------
    virtual void move_to_position(int position) = 0;

    //--------------------------------------------------------------------------
    /// \brief Return true if current position is at end of the list.
    //--------------------------------------------------------------------------
    //virtual bool is_at_end() = 0;

    //--------------------------------------------------------------------------
    /// \brief Return: The current element.
    //--------------------------------------------------------------------------
    virtual const T& get_value() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Return: The current element.
    //--------------------------------------------------------------------------
    //virtual T get_value() const = 0;

    //virtual bool is_empty() = 0;

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

#endif // DATA_STRUCTURES_LISTS_LIST_H