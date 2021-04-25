//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating array as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.1. The ADT for a list.
/// \ref https://github.com/OpenDSA/OpenDSA/blob/master/SourceCode/C%2B%2B_Templates/Lists/List.h
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_ARRAY_H
#define DATA_STRUCTURES_ARRAYS_ARRAY_H

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Arrays
{

//-----------------------------------------------------------------------------
/// \brief Array ADT
/// \details Intended to be a base class.
//-----------------------------------------------------------------------------
template <typename T>
class Array
{
  public:

  	virtual ~Array() = default;

  	//--------------------------------------------------------------------------
  	/// \brief Return: element at given index.
  	/// \details Do bound/range checking.
  	//--------------------------------------------------------------------------
  	virtual const T& get_value(const int index) const = 0;

  	//--------------------------------------------------------------------------
  	/// \brief Set element at given index.
  	/// \details Do bound/range checking.
  	//--------------------------------------------------------------------------
  	virtual void set_value(const int index, const T value) = 0;

  	//--------------------------------------------------------------------------
  	/// \brief Array subscript operator overload.
  	/// \details Do no bound/range checking.
  	/// \ref https://en.cppreference.com/w/cpp/language/operators
  	/// Array subscript operator.
  	//--------------------------------------------------------------------------
  	virtual T& operator[](const std::size_t index) = 0;
  	virtual const T& operator[](const std::size_t index) const = 0;

  	virtual std::size_t size() const = 0;

  	virtual std::size_t alignment_in_bytes() const = 0;

  protected:

  	Array() = default;
};

namespace CRTP
{

template <typename T, class Implementation>
class Array
{
  public:

    const T& get_value(const std::size_t index) const
    {
      return object()->get_value(index);
    }

    void set_value(const std::size_t index, const T value)
    {
      object()->set_value(index, value);
    }

    T& operator[](const std::size_t index)
    {
      return object()->operator[](index);
    }

    const T& operator[](const std::size_t index) const
    {
      return object()->operator[](index);
    }

    std::size_t size() const
    {
      return object()->size();
    }

  private:

    Implementation& object()
    {
      return static_cast<Implementation&>(*this);
    }
};

} // namespace CRTP

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_H