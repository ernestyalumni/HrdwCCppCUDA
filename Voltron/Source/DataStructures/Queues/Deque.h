//------------------------------------------------------------------------------
/// \ref 3.04.Deques.pdf, D.W. Harder, 2011, ECE 250 Algorithms and Data
/// Structures
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_DEQUE_H
#define DATA_STRUCTURES_QUEUES_DEQUE_H

#include <cstddef> // std::size_t
#include <memory>

namespace DataStructures
{
namespace Deques
{

namespace AsHierarchy
{

template <typename T>
class Deque
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    Deque() = default;

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    virtual void push_front(const T& item) = 0;

    virtual void push_back(const T& item) = 0;

    virtual T pop_front() = 0;

    virtual T pop_back() = 0;

    virtual T front() const = 0;

    virtual T back() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Is the deque empty?
    //--------------------------------------------------------------------------
    virtual bool is_empty() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the deque.
    //--------------------------------------------------------------------------
    virtual std::size_t size() const = 0;
};

} // namespace AsHierarchy

namespace CRTP
{

} // namespace CRTP

namespace Pimpl
{

} // namespace Pimpl

} // namespace Deques
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_DEQUE_H