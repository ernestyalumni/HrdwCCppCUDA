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

    T front() const
    {
      return (left() == nullptr) ? value() : left()->front();
    }

    T back() const
    {
      return (right() == nullptr) ? retrieve() : right()->back();
    }

    bool find(const T object) const
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

    /*
    bool insert(const T object)
    {
      if (object < value())
      {
        if (left() == nullptr)
        {

        }
      }
    }
    */

    bool erase(const T&, BinarySearchNode*);

    friend class BinarySearchTree<T>;

  private:

    std::size_t node_count_;
};

} // namespace DWHarder

} // namespace BinaryTrees

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_SEARCH_NODE_H