#ifndef DATA_STRUCTURES_LINKED_LISTS_REVERSE_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_REVERSE_LIST_H

#include "Node.h"

namespace DataStructures
{
namespace LinkedLists
{

//------------------------------------------------------------------------------
/// \ref https://www.geeksforgeeks.org/reverse-a-linked-list/
//------------------------------------------------------------------------------
template <typename T>
Nodes::Node<T>* reverse_list(Nodes::Node<T>* input_head)
{
  using Node = Nodes::Node<T>;

  // Initialize 3 pointers.
  Node* previous {nullptr};
  Node* current {input_head};
  Node* next {nullptr};

  while (current != nullptr)
  {
    // Before changing next of current, store next node.
    next = current->next_;
    // Now change next of current. This is where the actual reversing happens.
    current->next_ = previous;
    // Move previous and current one step forward.
    previous = current;
    current = next;
  }

  return previous;
}

} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_NODES_NODE_H
