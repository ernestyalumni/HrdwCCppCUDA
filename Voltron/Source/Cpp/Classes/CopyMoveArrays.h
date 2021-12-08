#ifndef CPP_CLASSES_COPY_MOVE_ARRAYS_H
#define CPP_CLASSES_COPY_MOVE_ARRAYS_H

#include <algorithm>
#include <array>
#include <cstddef> // std::size_t
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Cpp
{
namespace Classes
{

template <typename T, std::size_t N>
class WithArray
{
  public:

    WithArray():
      data_{}
    {
      data_.fill(static_cast<T>(0));
    }

    explicit WithArray(const std::initializer_list<T> input):
      data_{}
    {
      std::copy(input.begin(), input.end(), data_.begin());
    }

    //--------------------------------------------------------------------------
    /// \brief Copy ctor.
    //--------------------------------------------------------------------------
    WithArray(const WithArray& rhs):
      data_{rhs.data_}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    WithArray& operator=(const WithArray& rhs)
    {
      data_ = rhs.data_;
      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Move ctor.
    //--------------------------------------------------------------------------
    WithArray(WithArray&& rhs):
      data_{}
    {
      data_.swap(rhs.data_);
    }

    //--------------------------------------------------------------------------
    /// \brief Move assignment.
    //--------------------------------------------------------------------------
    WithArray& operator=(WithArray&& rhs)
    {
      data_.swap(rhs.data_);
      return *this;
    }

    virtual ~WithArray() = default;

    T operator[](const std::size_t i)
    {
      return data_[i];
    }

    void set_value(const std::size_t i, const T value)
    {
      data_[i] = value;
    }

    template <typename U, std::size_t M>
    friend WithArray<U, M> operator+(
      const WithArray<U, M>& a, 
      const WithArray<U, M>& b);
    /*
    {
      WithArray<U, M> c;
      std::transform(
        a.data_.begin(),
        a.data_.end(),
        b.data_.begin(),
        c.data_.begin(),
        std::plus<T>{});
      return c; 
    }*/

  private:

    std::array<T, N> data_;
};

/* TODO: Need to make as friend.
template <typename T, std::size_t N>
inline WithArray<T, N> operator+(
  const WithArray<T, N>& a,
  const WithArray<T, N>& b)
{
  WithArray<T, N> c;
  std::transform(a.data_.begin(), a.data_.end(), b.data_.begin(), c.data_.begin(), std::plus<T>{});

  return c; 
}
*/

template <typename T, std::size_t N>
inline WithArray<T, N> operator+(
  const WithArray<T, N>& a,
  const WithArray<T, N>& b)
{
  WithArray<T, N> c;
  std::transform(
    a.data_.begin(),
    a.data_.end(),
    b.data_.begin(),
    c.data_.begin(),
    std::plus<T>{});

  return c; 
}


template <typename T, std::size_t N>
class DefaultWithArray
{
  public:

    DefaultWithArray():
      data_{}
    {
      data_.fill(static_cast<T>(0));
    }

    explicit DefaultWithArray(const std::initializer_list<T> input):
      data_{}
    {
      std::copy(input.begin(), input.end(), data_.begin());
    }

    //--------------------------------------------------------------------------
    /// \brief Copy ctor.
    //--------------------------------------------------------------------------
    DefaultWithArray(const DefaultWithArray& rhs) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    DefaultWithArray& operator=(const DefaultWithArray& rhs) = default;

    //--------------------------------------------------------------------------
    /// \brief Move ctor.
    //--------------------------------------------------------------------------
    DefaultWithArray(DefaultWithArray&& rhs) = default;

    //--------------------------------------------------------------------------
    /// \brief Move assignment.
    //--------------------------------------------------------------------------
    DefaultWithArray& operator=(DefaultWithArray&& rhs) = default;

    virtual ~DefaultWithArray() = default;

    T operator[](const std::size_t i)
    {
      return data_[i];
    }

    void set_value(const std::size_t i, const T value)
    {
      data_[i] = value;
    }

    template <typename U, std::size_t M>
    friend DefaultWithArray<U, M> operator+(
      const DefaultWithArray<U, M>& a, 
      const DefaultWithArray<U, M>& b);

  private:

    std::array<T, N> data_;
};

/* TODO: Need friend to access private data.
template <typename T, std::size_t N>
inline DefaultWithArray<T, N> operator+(
  const DefaultWithArray<T, N>& a,
  const DefaultWithArray<T, N>& b)
{
  DefaultWithArray<T, N> c;
  std::transform(a.data_.begin(), a.data_.end(), b.data_.begin(), c.data_.begin(), std::plus<T>{});

  return c; 
}
*/

template <typename T, std::size_t N>
inline DefaultWithArray<T, N> operator+(
  const DefaultWithArray<T, N>& a,
  const DefaultWithArray<T, N>& b)
{
  DefaultWithArray<T, N> c;
  std::transform(
    a.data_.begin(),
    a.data_.end(),
    b.data_.begin(),
    c.data_.begin(),
    std::plus<T>{});

  return c; 
}

template <typename T>
class WithVector
{
  public:

    virtual ~WithVector() = default;

  private:

    std::vector<T> data_;
};

template <typename T>
class WithRawPointer
{
  public:

    virtual ~WithRawPointer();

  private:
};

} // namespace Classes
} // namespace Cpp

#endif // CPP_CLASSES_COPY_MOVE_ARRAYS_H