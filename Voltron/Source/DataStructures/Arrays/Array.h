#ifndef DATA_STRUCTURES_ARRAYS_ARRAY_H
#define DATA_STRUCTURES_ARRAYS_ARRAY_H

#include <algorithm>
#include <cmath>
#include <cstddef> // std::size_t
#include <execution> // std::execution
#include <initializer_list>
#include <iostream>
#include <numeric> // std::reduce
#include <stdexcept>

namespace DataStructures
{
namespace Arrays
{

namespace DWHarder
{

template <typename T>
class Array
{
  public:

    using size_t = std::size_t;

    explicit Array(const size_t N);

    explicit Array(const std::initializer_list<T>& input);

    //--------------------------------------------------------------------------
    /// \brief Copy Constructor
    /// \ref Introductory Project, D.W. Harder, U. Waterloo.
    /// \details Consider what happens here:
    ///   void f(Array second)
    ///   { // do something }
    /// and you pass in an object, i.e.
    ///   Array first (5);
    ///   f(first);
    ///
    /// Calling f, a new Array, second, is allocated on the stack and all its
    /// member variables are copied over, but only a shallow copy is done. When
    /// f returns, dtor is called on instance second, leaving first dangling.
    ///
    /// shallow copy - when values of the member variables are all simply copied
    /// over.
    /// deep copy - allocate memory for a new array and copy over the values.
    //--------------------------------------------------------------------------
    Array(const Array&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    /// \ref https://en.cppreference.com/w/cpp/language/copy_assignment
    /// \details No copy-and-swap idiom.
    /// Copy assignment operator must correctly deal with an object that has
    /// already been constructed and may own resources.
    //--------------------------------------------------------------------------
    Array& operator=(const Array&);

    //--------------------------------------------------------------------------
    /// \brief Move constructor
    /// \ref 17.5.1 Copy Ch. 17 Construction, Cleanup, Copy, and Move;
    /// Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup,
    /// from the Matrix example.
    /// \details Move constructor simply takes representation from its source
    // and replace it with an empty Array.
    //--------------------------------------------------------------------------
    Array(Array&&);

    //--------------------------------------------------------------------------
    /// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/0/
    //--------------------------------------------------------------------------
    Array& operator=(Array&&);

    //--------------------------------------------------------------------------
    /// \brief Destructor.
    /// \details Deallocate the memory for the array.
    //--------------------------------------------------------------------------
    virtual ~Array()
    {
      delete [] internal_;
    }

    // Accessors.

    size_t size() const
    {
      return size_;
    }

    size_t capacity() const
    {
      return capacity_;
    }

    bool empty() const
    {
      return size() == 0;
    }

    bool full() const
    {
      return size() == capacity();
    }

    T operator[](const size_t n) const;

    T* begin()
    {
      return internal_;
    }

    // TODO: Could this be constexpr instead of const?
    const T* begin() const
    {
      return internal_;
    }

    T* end()
    {
      return internal_ + size_;
    }

    // TODO: Could this be constexpr instead of const?
    const T* end() const
    {
      return internal_ + size();
    }

    T* cbegin() const
    {
      return internal_;
    }

    T* cend() const
    {
      return internal_ + size();
    }

    // Mutators

    bool append(const T element)
    {
      if (size_ < capacity_)
      {
        internal_[size_] = element;
        ++size_;
        return true;
      }
      else
      {
        return false;
      }
    }

    //--------------------------------------------------------------------------
    /// \details It does not have to zero out the array - as new objects arrive,
    /// they will replace what currently exists in the array.
    //--------------------------------------------------------------------------
    void clear()
    {
      size_ = 0;
    }

    //--------------------------------------------------------------------------
    /// \details Swap each and all member variables.
    //--------------------------------------------------------------------------
    void swap(Array& other);

    // A friend to print the array
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const Array<U>&);

  private:

    size_t capacity_;
    T* internal_;
    size_t size_;
};

template <typename T>
Array<T>::Array(const size_t N):
  capacity_{std::max(static_cast<size_t>(1), N)},
  internal_{new T[capacity_]},
  size_{0}
{}

template <typename T>
Array<T>::Array(const std::initializer_list<T>& input):
  capacity_{std::max(static_cast<size_t>(1), input.size())},
  // TODO: Weigh doing this vs. having an empty initializer {}, i.e.
  // internal_{new T[capacity_]{}}
  internal_{new T[capacity_]},
  size_{input.size()}
{
  std::copy(input.begin(), input.end(), internal_);
}

template <typename T>
Array<T>::Array(const Array& other):
  capacity_{other.capacity()},
  internal_{new T[capacity_]},
  size_{other.size()}
{
  std::copy(other.begin(), other.end(), internal_);
}

template <typename T>
Array<T>& Array<T>::operator=(const Array& other)
{
  if (capacity_ < other.size())
  {
    delete [] internal_;
    internal_ = new T[other.capacity()];
    capacity_ = other.capacity();
  }

  size_ = other.size();

  std::copy(other.begin(), other.end(), internal_);

  return *this;
}

template <typename T>
Array<T>::Array(Array&& a):
  capacity_{a.capacity()},
  // Grab a's representation.
  internal_{a.internal_},
  size_{a.size()}
{
  a.capacity_ = 0;
  a.size_ = 0;
  // Clear a's representation.
  a.internal_ = nullptr;
}

template <typename T>
Array<T>& Array<T>::operator=(Array&& a)
{
  swap(a);
  return *this;
}

template <typename T>
T Array<T>::operator[](const size_t n) const
{
  return internal_[n];
}

template <typename T>
void Array<T>::swap(Array& other)
{
  std::swap(capacity_, other.capacity_);

  // This will swap the pointer of this object's internal_ for the rhs, which is
  // called other. Upon exiting operator=, the other object is destroyed. 
  std::swap(internal_, other.internal_);
  std::swap(size_, other.size_);
}

template <typename T>
T sum(const Array<T>& a)
{
  return std::accumulate(a.begin(), a.end(), static_cast<T>(0));
}

//------------------------------------------------------------------------------
/// \details The sample average is not defined if the size is zero.
//------------------------------------------------------------------------------
template <typename T>
T average(const Array<T>& a)
{
  if (a.size() == 0)
  {
    throw std::runtime_error("Cannot sample average if size is zero.");
  }

  return sum(a) / static_cast<T>(a.size());
}

template <typename T>
T variance(const Array<T>& a)
{
  const T sample_average {average(a)};

  if (a.size() == 1)
  {
    throw std::runtime_error("Cannot take variance if size is 1");
  }

  T summation {static_cast<T>(0)};

  for (const auto a_k : a)
  {
    summation += (a_k - sample_average) * (a_k - sample_average);
  }

  return summation / static_cast<T>(a.size() - 1);
}

template <typename T>
T std_dev(const Array<T>& a)
{
  T result {
    variance(a) * static_cast<T>(a.size() - 1) / static_cast<T>(a.size())};

  return std::sqrt(result);
}

} // namespace DWHarder
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_H