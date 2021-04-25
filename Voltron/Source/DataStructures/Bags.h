//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating bag as an Abstract Data Type.
/// \ref Sedgewick and Wayne, Algorithms, 4th. Ed., 2011, pp. 121, Sec. 1.3
/// Bags, Queues, and Stacks.
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_BAGS_H
#define DATA_STRUCTURES_BAGS_H

#include <cstddef> // std::size_t
#include <memory>

namespace DataStructures
{
namespace Bags
{

template <typename Item>
class Bag
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty bag.
    //--------------------------------------------------------------------------
    Bag() = default;

    virtual ~Bag() = default;

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    virtual void add(const Item item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Is the bag empty?
    //--------------------------------------------------------------------------
    virtual bool is_empty() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the bag.
    //--------------------------------------------------------------------------
    virtual std::size_t size() const = 0;
};

namespace CRTP
{

template <typename Item, typename Implementation>
class Bag
{
  public:

    void add(const Item item)
    {
      object()->add(item);
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
class BaseBagImplementation
{
  public:

    BaseBagImplementation() = default;

    virtual ~BaseBagImplementation() = default;

    void add(const Item item) = 0;
    bool is_empty() const = 0;
    std::size_t size() const = 0;
};

template <typename Item>
class Bag
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty bag.
    //--------------------------------------------------------------------------
    Bag() = default;

    virtual ~Bag() = default;

    void add(const Item item)
    {
      implementation_->add(item);
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

    //class BaseBagImplementation; // Forward declaration of implementation class
    std::unique_ptr<BaseBagImplementation<Item>> implementation_;
};


} // Pimpl

} // namespace Bags
} // namespace DataStructures

#endif // DATA_STRUCTURES_BAGS_H