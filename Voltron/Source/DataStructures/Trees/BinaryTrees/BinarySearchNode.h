#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_SEARCH_NODE_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_SEARCH_NODE_H

#include "BinaryNode.h"

#include <cassert>
#include <cstddef>

namespace DataStructures
{
namespace Trees
{
namespace BinaryTrees
{

namespace DWHarder
{

template <typename T>
class BinarySearchTree;

//------------------------------------------------------------------------------
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/Trees/Binary_search_trees/src/Binary_search_node.h
//------------------------------------------------------------------------------
template <typename T>
class BinarySearchNode : public BinaryNode<T>
{
  public:

    // Without this, you'll get compilation error: there are no arguments that
    // depend on a tempalte parameter, so a declaration must be available.
    using BinaryNode<T>::value;

    BinarySearchNode(const T& object):
      BinaryNode<T>{object},
      node_count_{1}
    {}

    BinarySearchNode* left() const
    {
      return reinterpret_cast<BinarySearchNode*>(BinaryNode<T>::left());
    }

    BinarySearchNode* right() const
    {
      return reinterpret_cast<BinarySearchNode*>(BinaryNode<T>::right());      
    }

    //--------------------------------------------------------------------------
    /// \brief Effectively gets the smallest value.
    /// \details Finds the minimum object.
    /// \ref 6.1.4.1 Finding the Minimum Object, Slide 21 of
    /// 6.0.1.Binary_search_trees.pptx
    //--------------------------------------------------------------------------
    T front() const
    {
      return (left() == nullptr) ? value() : left()->front();
    }

    //--------------------------------------------------------------------------
    /// \brief Effectively gets the largest value.
    /// \details Finds the maximum object.
    /// \ref 6.1.4.2 Finding the Maximum Object, Slide 22 of
    /// 6.0.1.Binary_search_trees.pptx
    //--------------------------------------------------------------------------
    T back() const
    {
      return (right() == nullptr) ? value() : right()->back();
    }

    //--------------------------------------------------------------------------
    /// \ref 6.1.4.3 Find, Slide 23 of 6.0.1.Binary_search_trees.pptx
    //--------------------------------------------------------------------------
    bool find(const T& object) const
    {
      if (value() == object)
      {
        return true;
      }
      else if (object < value())
      {
        return (left() == nullptr) ? false : left()->find(object);
      }
      else
      {
        assert(value() < object);
        return (right() == nullptr) ? false : right()->find(object);
      }
    }

    void clear()
    {
      if (left() != nullptr)
      {
        left()->clear();
      }

      if (right() != nullptr)
      {
        right()->clear();
      }

      delete this;
    }

    //--------------------------------------------------------------------------
    /// \brief Effectively gets the next largest value to input argument.
    //--------------------------------------------------------------------------
    T next(const T& object) const
    {
      if (value() == object)
      {
        return (right() == nullptr) ? object : right()->front();
      }
      else if (value() > object)
      {
        const T temp {(left() == nullptr) ? object : left()->next(object)};

        return (temp == object) ? value() : temp;
      }
      else
      {
        assert(value() < object);
        return (right() == nullptr) ? object : right()->next(object);
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Effectively gets the "next" smaller value to input argument.
    //--------------------------------------------------------------------------
    T previous(const T& object) const
    {
      if (value() == object)
      {
        return (left() == nullptr) ? object : left()->back();
      }
      else if (value() < object)
      {
        const T temp {right() == nullptr ? object : right()->previous(object)};
      }
      else
      {
        assert(value() > object);
        return (left() == nullptr) ? object : left()->previous(object);
      }
    }

    T at(const std::size_t k) const
    {
      if (k == 0 || left()->size() == k)
      {
        return value();
      }
      else if (left()->size() > k)
      {
        return left()->at(k);
      }
      else
      {
        return right()->at(k - left()->size() - 1);
      }
    }

    //--------------------------------------------------------------------------
    /// \details O(h) runtime, time complexity.
    /// \ref 6.1.4.4 Insert, Slide 30 of 6.0.1.Binary_search_trees.pptx
    //--------------------------------------------------------------------------
    bool insert(const T& object)
    {
      if (object < value())
      {
        if (left() == nullptr)
        {
          p_left_tree_ = new BinarySearchNode<T>(object);
          ++node_count_;
          return true;
        }
        else
        {
          if (left()->insert(object))
          {
            ++node_count_;
            return true;
          }
          else
          {
            return false;
          }
        }
      }
      else if (object > value())
      {
        if (right() == nullptr)
        {
          p_right_tree_ = new BinarySearchNode<T>(object);
          ++node_count_;
          return true;
        }
        else
        {
          if (right()->insert(object))
          {
            ++node_count_;
            return true;
          }
          else
          {
            return false;
          }
        }
      }
      else
      {
        return false;
      }
    }

    bool erase(const T&, BinarySearchNode*);

    friend class BinarySearchTree<T>;

  private:

    using BinaryNode<T>::node_value_;
    using BinaryNode<T>::p_left_tree_;
    using BinaryNode<T>::p_right_tree_;
    std::size_t node_count_;
};

} // namespace DWHarder

} // namespace BinaryTrees

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_SEARCH_NODE_H