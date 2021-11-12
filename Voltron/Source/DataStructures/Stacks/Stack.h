//------------------------------------------------------------------------------
/// \file Stack.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating stack.
/// \details
///
/// \ref https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
/// \ref https://www.techiedelight.com/stack-implementation-in-cpp/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_STACKS_STACK_H
#define DATA_STRUCTURES_STACKS_STACK_H

#include <cstddef> // std::size_t
#include <memory>

namespace DataStructures
{
namespace Stacks
{

namespace AsHierarchy
{

template <typename Item>
class Stack
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty stack.
    //--------------------------------------------------------------------------
    Stack() = default;

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    virtual void push(const Item item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Remove the most recently added item.
    //--------------------------------------------------------------------------
    virtual Item pop() = 0;

    //--------------------------------------------------------------------------
    /// \brief Is the queue empty?
    //--------------------------------------------------------------------------
    virtual bool is_empty() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the queue.
    //--------------------------------------------------------------------------
    virtual std::size_t size() const = 0;    
};

} // namespace AsHierarchy

namespace CRTP
{

template <typename Item, typename Implementation>
class Stack
{
  public:

    void push(const Item item)
    {
      object()->push(item);
    }

    Item pop()
    {
      return object()->pop();
    }

    bool is_empty() const
    {
      return object()->is_empty();
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

namespace Pimpl
{

template <typename Item>
class BaseStackImplementation
{
  public:

    BaseStackImplementation() = default;

    virtual ~BaseStackImplementation() = default;

    void push(const Item item) = 0;
    Item pop() = 0;
    bool is_empty() const = 0;
    std::size_t size() const = 0;
};

template <typename Item>
class Stack
{
  public:

    Stack() = default;

    virtual ~Stack() = default;

    void push(const Item item)
    {
      implementation_->push(item);
    }

    Item pop()
    {
      return implementation_->pop();
    }

    bool is_empty() const
    {
      return implementation_->is_empty();
    }

    std::size_t size() const
    {
      return implementation_->size();
    }

  private:

    std::unique_ptr<BaseStackImplementation<Item>> implementation_;
};

} // namespace Pimpl

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_STACK_H