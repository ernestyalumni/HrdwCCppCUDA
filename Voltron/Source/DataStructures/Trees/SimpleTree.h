#ifndef DATA_STRUCTURES_TREES_SINPLE_TREE_H
#define DATA_STRUCTURES_TREES_SINPLE_TREE_H

#include "DataStructures/LinkedLists/SingleList.h"

namespace DataStructures
{
namespace Trees
{
namespace DWHarder
{

//------------------------------------------------------------------------------
/// \url https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/Trees/Simple_trees/src/Simple_tree.h
//------------------------------------------------------------------------------
template <typename T>
class SimpleTree
{
  public:

    template <typename U>
    using SingleList = DataStructures::LinkedLists::DWHarder::SingleList<U>;

    SimpleTree(const T&, SimpleTree* = nullptr);

    ~SimpleTree() = default;

    T retrieve() const;
    size_t degree() const;
    bool is_root() const;
    bool is_leaf() const;
    SimpleTree* parent() const;
    SimpleTree* child(const std::size_t n) const;

    std::size_t size() const;
    void height() const;

    // void depth_first_traversal() const;

    void insert(const T&);
    void insert(SimpleTree*);
    void detach();

    //--------------------------------------------------------------------------
    /// \details For each node within a tree, there's a unique path from root
    /// node to that node. Length of this path is referred to as depth of the
    /// node. Depth of the root node is 0.
    /// \ref 4.2c. Write a recursive member function that finds the depth of a
    /// node. 4.02.Abstract_trees.Questions.pdf, 2013 D.W. Harder. ECE 250.
    //--------------------------------------------------------------------------
    std::size_t depth() const
    {
      if (is_root())
      {
        return 0;
      }

      return 1 + parent()->depth();
    }

    //--------------------------------------------------------------------------
    /// \ref 4.2c. Write an iterative member function that finds the depth of a
    /// node. 4.02.Abstract_trees.Questions.pdf, 2013 D.W. Harder. ECE 250.
    //--------------------------------------------------------------------------
    std::size_t iterative_depth() const
    {
      if (is_root())
      {
        return 0;
      }

      std::size_t depth {0};
      for (
        SimpleTree* current_ptr {parent()};
        current_ptr != nullptr;
        current_ptr = current_ptr->parent())
      {
        depth += 1;
      }

      return depth;
    }

    SimpleTree* root()
    {
      if (is_root())
      {
        return this;
      }

      return parent_node_->root();
    }

  private:

    T element_;
    SimpleTree* parent_node_;
    SingleList<SimpleTree*> children_;
};

template <typename T>
SimpleTree<T>::SimpleTree(const T& obj, SimpleTree* p):
  element_{obj},
  parent_node_{p},
  children_{}
{}

template <typename T>
T SimpleTree<T>::retrieve() const
{
  return element_;
}

template <typename T>
std::size_t SimpleTree<T>::degree() const
{
  return children_.size();
}

template <typename T>
bool SimpleTree<T>::is_root() const
{
  return parent_node_ == nullptr;
}

template <typename T>
bool SimpleTree<T>::is_leaf() const
{
  return degree() == 0;
}

template <typename T>
SimpleTree<T>* SimpleTree<T>::parent() const
{
  return parent_node_;
}

template <typename T>
SimpleTree<T>* SimpleTree<T>::child(const std::size_t n) const
{
  if (n >= degree())
  {
    return nullptr;
  }

  auto* ptr = children_.head();

  for (std::size_t i {1}; i < n; ++i)
  {
    ptr = ptr->next();
  }

  return ptr->retrieve();
}

template <typename T>
std::size_t SimpleTree<T>::size() const
{
  std::size_t s {1};

  for (auto* ptr = children_.head(); ptr != nullptr; ptr = ptr->next())
  {
    s += ptr->retrieve()->size();
  }

  return s;
}

template <typename T>
void SimpleTree<T>::height() const
{
  std::size_t h {0};

  for (auto* ptr = children_.head(); ptr != 0; ptr = ptr->next())
  {
    h = std::max(h, 1 + ptr->retrieve()->height());
  }
}

template <typename T>
void SimpleTree<T>::insert(const T& obj)
{
  children_.push_back(new SimpleTree(obj, this));
}

template <typename T>
void SimpleTree<T>::detach()
{
  if (parent_node_ == nullptr)
  {
    return;
  }

  parent_node_->children_.erase(this);
}

} // namespace DWHarder

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_SINPLE_TREE_H