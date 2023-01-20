#ifndef DATA_STRUCTURES_TREES_PARENTAL_TREE_H
#define DATA_STRUCTURES_TREES_PARENTAL_TREE_H

#include "DataStructures/Arrays/DynamicArray.h"
#include "DataStructures/Trees/SimpleTreeNode.h"
#include "DataStructures/Trees/DynamicTreeNode.h"

#include <stdexcept> // std::runtime_error
#include <tuple>

namespace DataStructures
{
namespace Trees
{

//------------------------------------------------------------------------------
/// \ref 4.04.Parental_trees.pptx, Harder, 2011. ECE 250.
/// \details Memory requirements are only O(N), N = number of nodes.
//------------------------------------------------------------------------------
template <typename T>
class ParentalTree
{
  public:

    template <typename U>
    using Array = DataStructures::Arrays::PrimitiveDynamicArray<U>;

    template <typename V>
    using SimpleTreeNode = DataStructures::Trees::SimpleTreeNode<V>;

    template <typename W>
    using DynamicTreeNode = DataStructures::Trees::SimpleTreeNode<V>;

    explicit ParentalTree():
      parent_indices_{}
    {}

    explicit ParentalTree(const std::size_t number_of_nodes):
      parent_indices_{number_of_nodes}
    {}

    ~ParentalTree() = default;

    //--------------------------------------------------------------------------
    /// \details 0-based indexing.
    //--------------------------------------------------------------------------
    std::size_t parent_index_of_node(const std::size_t node_index) const
    {
      return parent_indices_[child_index];
    }

    void add_node(const std::size_t parent_index)
    {
      parent_indices_.append(parent_index);
    }

    std::size_t size() const
    {
      return parent_indices_.size();
    }

    // TODO: Clarify procedure to add the value for each node.
    /*
    std::tuple<SimpleTreeNode*, Array<SimpleTreeNode*>> to_simple_tree_nodes()
    {
      SimpleTreeNode<T>* root_node {nullptr};
      Array<SimpleTreeNode*> array {size(), nullptr};

      for (std::size_t i {0}; i < size(); ++i)
      {
        array[i] = new SimpleTreeNode
      }

    }
    */

  private:

    Array<std::size_t> parent_indices_;
};

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_PARENTAL_TREE_H