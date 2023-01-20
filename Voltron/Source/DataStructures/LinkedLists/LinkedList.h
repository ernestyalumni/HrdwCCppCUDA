#ifndef DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H

#include "Node.h"

#include <cstddef>
#include <stdexcept>

namespace DataStructures
{
namespace LinkedLists
{
namespace DWHarder
{

//------------------------------------------------------------------------------
/// \ref 3.05.Linked_lists.pptx, D.W. Harder.
//------------------------------------------------------------------------------
template <typename T>
class LinkedList
{
  public:

    using Node = DataStructures::LinkedLists::Nodes::Node<T>;

    LinkedList():
      head_{nullptr}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    LinkedList(const LinkedList& list):
      head_{nullptr}
    {
      if (list.is_empty())
      {
        return;
      }

      // Copy the first node-we no longer modify head_.
      push_front(list.front());

      // We modify the next pointer of the node pointed to by copy.
      for (
        Node* original {list.begin()->next_}, *copy {begin()};
        original != list.end();
        // Then we move each pointer forward.
        original = original->next_, copy = copy->next_)
      {
        copy->next_ = new Node(original->value_, nullptr);
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Move constructor.
    /// \ref Slide 119. 3.05.Linked_lists.pptx, D.W. Harder
    /// Move ctor called when an rvalue is being assigned - as an rvalue, it'll
    /// be deleted anyway. The instance ls is being deleted as soon as it's
    /// copied.
    /// Instead, if a move ctor is defined, it'll be called instead of a copy
    /// ctor.
    //--------------------------------------------------------------------------
    LinkedList(LinkedList&& list):
      head_{list.head_}
    {
      list.head_ = nullptr;
    }

    //--------------------------------------------------------------------------
    /// \brief Assignment
    /// \ref Slide 120 3.05.Linked_lists.pptx, D.W. Harder
    /// \details The RHS is passed by value (a copy is made). Return value is
    /// passed by reference.
    /// Note that rhs is a copy of the list, it's not a pointer to a list.
    ///
    /// If this isn't commented out, it'll lead to error, ambiguous overload.
    //--------------------------------------------------------------------------
    /*
    LinkedList& operator=(LinkedList rhs)
    {
      // We will swap all the values of the member variables between the left-
      // and right-hand sides.
      // rhs is already a copy, so we swap all member variables of it and *this
      std::swap(*this, rhs);

      // Memory for rhs was allocated on the stack and the dtor will delete it.

      return *this;
    }
    */

    //--------------------------------------------------------------------------
    /// \brief Assignment
    /// \ref Slide 131 3.05.Linked_lists.pptx, D.W. Harder
    /// \details First, pass by reference-no copying.
    //--------------------------------------------------------------------------
    LinkedList& operator=(const LinkedList& rhs)
    {
      // Ensure we're no assigning something to itself.
      if (this == &rhs)
      {
        return *this;
      }

      // If the right-hand side is empty, it's straightforward: just empty this
      // list.
      if (rhs.is_empty())
      {
        while (!is_empty())
        {
          pop_front();
        }

        return *this;
      }

      if (is_empty())
      {
        head_ = new Node{rhs.front()};
      }      
      else
      {
        begin()->value_ = rhs.front();
      }

      //------------------------------------------------------------------------
      /// Step through the right-hand side list and for each node,
      /// * if there's a corresponding node in this, copy over the value, else
      /// * There's no corresponding node, create a new node and append it.
      //------------------------------------------------------------------------

      Node* this_node {head_};
      Node* rhs_node {rhs.begin()->next_};

      while (rhs_node != nullptr)
      {
        // There's no corresponding node; create a new node and append it.
        if (this_node->next_ == nullptr)
        {
          this_node->next_ = new Node(rhs_node->value_);
          this_node = this_node->next_;
        }
        // If there's a corresponding node in this, copy over the value.
        else
        {
          this_node = this_node->next_;
          this_node->value_ = rhs_node->value_;
        }

        rhs_node = rhs_node->next_;
      }

      // If there are any nodes remaining in this, delete them.
      while (this_node->next_ != nullptr)
      {
        Node* tmp {this_node->next_};
        this_node->next_ = this_node->next_->next_;
        delete tmp;
      }

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Move Assignment
    /// \ref Slide 134 3.05.Linked_lists.pptx, D.W. Harder
    //--------------------------------------------------------------------------
    LinkedList& operator=(LinkedList&& rhs)
    {
      while (!is_empty())
      {
        pop_front();
      }

      head_ = rhs.begin();
      rhs.head_ = nullptr;

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \details Runs in O(N) time, where N is number of objects in the linked
    /// list.
    //--------------------------------------------------------------------------   
    virtual ~LinkedList()
    {
      while (!is_empty())
      {
        pop_front();
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Adding the value at the front of the linked list.
    /// \details insert, front O(1)
    //--------------------------------------------------------------------------
    void push_front(const T value)
    {
      // When list is empty, head_ == 0, i.e. head_ == nullptr;
      head_ = new Node(value, head_);      
    }    

    //--------------------------------------------------------------------------
    /// \brief Retrieving the value at the front of the linked list.
    /// \details access, front O(1)
    //--------------------------------------------------------------------------
    T front() const
    {
      if (is_empty())
      {
        throw std::runtime_error("Attempted front() on empty LinkedList");
      }

      return head_->value_;      
    }

    //--------------------------------------------------------------------------
    /// \brief Removing the value at the front of the linked list.
    /// \details Erase, front O(1).
    //--------------------------------------------------------------------------
    T pop_front()
    {
      if (is_empty())
      {
        throw std::runtime_error("Attempted pop_front() on empty LinkedList");
      }

      const T result {front()};

      // Assign a temporary pointer to point to the node being deleted.
      // Because we could have been accessing a node which we have just deleted:
      // e.g. delete begin(); head_ = begin()->next_;

      Node* head_ptr {head_};

      head_ = head_->next_;

      // Given that delete is approximately 100x slower than most other instructions
      // (it does call the OS). cf. Slide 76, Destructor. Waterloo Engineering.
      delete head_ptr;

      return result;
    }    

    //--------------------------------------------------------------------------
    /// \brief Access the head of the linked list.
    //--------------------------------------------------------------------------
    Node* begin() const
    {
      // This will always work: if list is empty, it'll return nullptr.
      return head_;      
    }

    //--------------------------------------------------------------------------
    /// \details The member function Node* end() const equals whatever the last
    /// node in the linked list points to - in this case, nullptr.
    /// \ref 3.05.Linked_lists.pptx, D.W. Harder 2001 Waterloo Engineering.
    //--------------------------------------------------------------------------
    Node* end() const
    {
      return nullptr;
    }

    //--------------------------------------------------------------------------
    /// \brief Find the number of instances of an integer in the list.
    /// \details O(N) time complexity.
    //--------------------------------------------------------------------------
    std::size_t count(const T value) const;

    //--------------------------------------------------------------------------
    /// \brief Remove all instances of a value from the list.
    /// \details erase, front O(1), arbitrary O(N), back O(N)
    //--------------------------------------------------------------------------
    std::size_t erase(const T value);

    //--------------------------------------------------------------------------
    /// \brief Is the linked list empty?
    //--------------------------------------------------------------------------
    bool is_empty() const
    {
      return (head_ == nullptr);
    }

    //--------------------------------------------------------------------------
    /// \brief How many objecst are in the list?
    /// \details The list is empty when the head pointer is set to nullptr.
    /// O(N) time.
    //--------------------------------------------------------------------------
    std::size_t size() const
    {
      std::size_t counter {0};

      for (Node* ptr {begin()}; ptr != end(); ptr = ptr->next_)
      {
        ++counter;
      }

      return counter;
    }

    //--------------------------------------------------------------------------
    /// \ref https://www.geeksforgeeks.org/reverse-a-linked-list/
    //--------------------------------------------------------------------------
    Node* reverse_list()
    {
      // Initialize current, previous, and next pointers.
      Node* current {head_};
      Node* previous {nullptr};
      Node* next {nullptr};

      while (current != nullptr)
      {
        // Before changing next of current, store next.
        next = current->next_;

        // Reverse current node's pointer.
        current->next_ = previous;

        // Move pointers one position ahead.
        previous = current;
        current = next;
      }

      head_ = previous;

      return previous;
    }

    //--------------------------------------------------------------------------
    /// \details Assume 0-based indexing.
    //--------------------------------------------------------------------------
    void list_delete(const std::size_t i)
    {
      if (i >= size())
      {
        throw std::runtime_error("Node to delete is not in the list");
      }

      if (i == 0)
      {
        Node* current {head_};
        head_ = head_->next_;
        delete current;
        return;
      }

      Node* previous {head_};
      Node* current {head_->next_};
      for (std::size_t j {1}; j < i; ++j)
      {
        previous = previous->next_;
        current = current->next_;
      }

      previous->next_ = current->next_;
      delete current;
    }    

  private:

    //--------------------------------------------------------------------------
    /// \ref 3.05.Linked_lists.pptx,
    /// \details Because each node in a linked lists refers to the next, linked
    /// list class need only link to first node in list.
    //--------------------------------------------------------------------------
    Node* head_;
};

template <typename T>
std::size_t LinkedList<T>::count(const T value) const
{
  std::size_t result {0};

  for (Node* ptr {begin()}; ptr != end(); ptr = ptr->next_)
  {
    if (ptr->value_ == value)
    {
      ++result;
    }
  }

  return result;
}

template <typename T>
std::size_t LinkedList<T>::erase(const T value)
{
  std::size_t counter {0};

  if (is_empty())
  {
    return counter;
  }

  Node* previous_ptr {nullptr};

  Node* ptr {begin()};

  while (ptr != end())
  {
    // We're at the front.
    if (ptr->value_ == value && previous_ptr == nullptr)
    {
      pop_front();

      ptr = head_;

      ++counter;
    }
    else if (ptr->value_ == value)
    {
      Node* ptr_to_delete {ptr};

      previous_ptr->next_ = ptr->next_;

      ptr = ptr->next_;

      delete ptr_to_delete;

      ++counter;
    }
    else
    {
      previous_ptr = ptr;
      ptr = ptr->next_;
    }
  }

  return counter;
}

} // namespace DWHarder
} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H