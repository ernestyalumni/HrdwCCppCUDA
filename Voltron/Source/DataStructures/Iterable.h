#ifndef DATA_STRUCTURES_ITERABLE_H
#define DATA_STRUCTURES_ITERABLE_H

#include <memory>

namespace DataStructures
{
namespace Iterables
{

template <typename T>
class Iterable
{
  public:

    Iterable() = default;

    virtual ~Iterable() = default;

    virtual T* begin() = 0;
    virtual T* end() = 0;
};

namespace CRTP
{

template <typename T, typename Implementation>
class Iterable
{
  public:

    T* begin()
    {
      return object()->begin();
    }

    T* end()
    {
      return object()->end();
    }

  private:

    Implementation& object()
    {
      return static_cast<Implementation&>(*this);
    }
};

} // namespace CRTP

namespace Pimpl
{

template <typename T>
class BaseIteratorImplementation
{
  public:

    BaseIteratorImplementation() = default;

    virtual ~BaseIteratorImplementation() = default;

    T* begin() = 0;
    T* end() = 0;
};

template <typename T>
class Iterator
{
  public:

    Iterator() = default;

    virtual ~Iterator() = default;

    T* begin()
    {
      return implementation_->begin();
    }

    T* end()
    {
      return implementation_->end();
    }

  private:

    std::unique_ptr<BaseIteratorImplementation<T>> implementation_;
};

} // Pimpl

} // namespace Iterables
} // namespace DataStructures

#endif // DATA_STRUCTURES_ITERABLE_H