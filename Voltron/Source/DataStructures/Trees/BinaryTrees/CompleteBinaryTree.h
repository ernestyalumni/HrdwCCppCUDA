#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_COMPLETE_BINARY_TREE_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_COMPLETE_BINARY_TREE_H

#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace DataStructures
{
namespace Trees
{
namespace BinaryTrees
{

//-----------------------------------------------------------------------------
/// \ref 5.03.Complete_binary_trees.Questions.pdf 4.6g Consider the following
/// class. 2013 D.W. Harder. ECE 250. University of Waterloo.
/// 5.03.Complete_binary_trees.pptx.
/// 5.3.3 Array storage
/// \details Height of a complete binary tree with N nodes is h = floor(lg(N))
/// See 5.3.2 Logarithmic Height, 2011 D.W. Harder, ECE 250.
//-----------------------------------------------------------------------------
template <typename T, std::size_t N>
class CompleteBinaryTree
{
  public:

    CompleteBinaryTree():
      array_{},
      size_{0},
      tail_index_{0}
    {}

    std::size_t size() const
    {
      return size_;
    }

    bool is_full() const
    {
      return size_ == N;
    }

    bool is_empty() const
    {
      return size_ == 0;
    }

    //--------------------------------------------------------------------------
    /// \ref 5.3.3 Array storage of 5.03.Complete_binary_trees.pptx. Having left
    /// the first entry blank yields a bonus.
    //--------------------------------------------------------------------------
    std::tuple<std::size_t, std::size_t> children_index(const std::size_t k)
    {
      if (k < 1 || k > size())
      {
        throw std::runtime_error(
          "Failed children_index for CompleteBinaryTree - input out of bounds");
      }

      return std::make_tuple(k << 1, (k << 1) | 1);
    }

    std::size_t parent_index(const std::size_t k)
    {
      if (k < 1 || k > size())
      {
        throw std::runtime_error(
          "Failed children_index for CompleteBinaryTree - input out of bounds");
      }

      return k >> 1;
    }

    //--------------------------------------------------------------------------
    /// \brief Append value to a node but not checking if argument is already
    /// there.
    //--------------------------------------------------------------------------
    void push_back_no_check(const T& argument)
    {
      if (is_full())
      {
        return;
      }

      ++size_;
      array_[size_] = argument;
    }

    //--------------------------------------------------------------------------
    /// \brief Remove value from most "bottom right" node but not checking if
    /// argument is already there.
    //--------------------------------------------------------------------------
    void pop_back_no_check(const T& argument)
    {
      if (is_empty())
      {
        throw std::runtime_error(
          "Failed pop_back_no_check for CompleteBinaryTree - already empty");
      }

      --size_;
    }

  protected:

    T& operator[](const std::size_t i)
    {
      return array_[i];
    }

    //--------------------------------------------------------------------------
    /// \return 0 if argument was not found.
    //--------------------------------------------------------------------------
    std::size_t find(const T& argument) const
    {
      for (std::size_t i {1}; i <= size(); ++i)
      {
        if (array_[i] == argument)
        {
          return i;
        }
      }

      return 0;
    }

  private:

    //--------------------------------------------------------------------------
    /// \details Allow for an extra, "empty" slot or element in the beginning,
    /// at index 0, so that we can use bitshift'ing.
    //--------------------------------------------------------------------------
    T array_[N + 1];

    std::size_t tail_index_;
    std::size_t size_;
};

} // namespace BinaryTrees

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_COMPLETE_BINARY_TREE_H