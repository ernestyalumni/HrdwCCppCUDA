//------------------------------------------------------------------------------
/// \file SpanView.h
/// \author Ernest Yeung
/// \brief Classes and functions substituting for std::span.
//-----------------------------------------------------------------------------
#ifndef CPP_STD_CONTAINERS_SPAN_VIEW_H
#define CPP_STD_CONTAINERS_SPAN_VIEW_H

#include <cstddef> // std::size_t
#include <limits>
#include <stdexcept>

namespace Cpp
{
namespace Std
{
namespace Containers
{

// The linker will consolidate all inline definitions into a single variable
// definition. This allows us to define variables in a header file and have
// them treated as if there was only 1 definition in a .cpp file somewhere.
// cf.
// https://www.learncpp.com/cpp-tutorial/global-constants-and-inline-variables/
inline constexpr std::size_t dynamic_extent {
  std::numeric_limits<std::size_t>::max()};

//-----------------------------------------------------------------------------
/// https://en.cppreference.com/w/cpp/container/span
//-----------------------------------------------------------------------------
//template <class T, std::size_t Extent = dynamic_extent>
template <class T>
class SpanView
{
  public:

    using PointerType = T*;

    //--------------------------------------------------------------------------
    /// cf. https://stackoverflow.com/posts/31377416/revisions
    /// \details constexpr ctors mean static initialization can be performed. 
    //--------------------------------------------------------------------------

    SpanView(PointerType ptr, const std::size_t size):
      ptr_to_array_{ptr},
      size_{size}
    {}

    // Returns an iterator to the beginning.
    constexpr T* begin()
    {
      return ptr_to_array_;
    }

    // Returns an iterator to the end.
    constexpr T* end()
    {
      return ptr_to_array_ + size_;
    }

    constexpr std::size_t size()
    {
      return size_;
    }

    T& operator[](const std::size_t index)
    {
      if (index >= size_)
      {
        throw std::runtime_error("span view index out of bound");
      }

      return ptr_to_array_[index];
    }

    //--------------------------------------------------------------------------
    /// \brief Obtains a span that's a view over the Count elements of this span
    /// starting at offset Offset.
    /// \details If Count is dynamic_extent, number of elements in subspan is
    /// size() - offset (i.e. it ends at end of *this)
    /// \ref https://en.cppreference.com/w/cpp/container/span/subspan
    //--------------------------------------------------------------------------
    constexpr SpanView<T> subspan(
      const std::size_t offset,
      const std::size_t count = dynamic_extent)
    {
      return (count == dynamic_extent) ?
        SpanView<T>{begin() + offset, size() - offset} :
          SpanView<T>{begin() + offset, count};
    }

  private:

    //--------------------------------------------------------------------------
    /// cf. https://stackoverflow.com/posts/61216722/revisions
    /// \details SpanView just points to the block of memory, knows how long the
    /// block of memory is, knows what data type, provides convenience accessor
    /// functions to work with elements in that contiguous memory.
    //--------------------------------------------------------------------------

    T* ptr_to_array_;
    std::size_t size_;
};

} // namespace Containers
} // namespace Std
} // namespace Cpp

#endif// CPP_STD_CONTAINERS_SPAN_VIEW_H