#ifndef DATA_STRUCTURES_LINKED_LISTS_NODES_DOUBLE_NODE_H
#define DATA_STRUCTURES_LINKED_LISTS_NODES_DOUBLE_NODE_H

namespace DataStructures
{
namespace LinkedLists
{

template <typename U>
class DoublyLinkedList;

namespace Nodes
{

//-----------------------------------------------------------------------------
/// \details No dynamic memory allocation; it's "static" memory.
//-----------------------------------------------------------------------------
template <typename T>
class DoubleNode
{
	public:

		DoubleNode(
			const T& element = T{},
			DoubleNode* next = nullptr,
			DoubleNode* previous = nullptr
			):
			value_{element},
			next_{next},
			previous_{previous}
		{}

		DoubleNode(const DoubleNode&) = default;
    DoubleNode& operator=(const DoubleNode&) = default;
    DoubleNode(DoubleNode&&) = default;
    DoubleNode& operator=(DoubleNode&&) = default;

		T retrieve() const
		{
			return value_;
		}

    T& get_value()
    {
      return value_;
    }

		DoubleNode* next() const
		{
			return next_;
		}

		DoubleNode* previous() const
		{
			return previous_;
		}

		friend class DoublyLinkedList<T>;

		// If ptr is a pointer to a DoubleNode<T> object,
		// in one of the friendly classes, you should:
		// 		use ptr->next_node 	to modify it
		// 		use ptr->next() 		to access it

		// These data members are made public because there are other applications,
		// in particular functions, that'll need to modify these values directly.

		T value_;
		DoubleNode* next_;
		DoubleNode* previous_;
}; // class DoubleNode

} // namespace Nodes
} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_NODES_DOUBLE_NODE_H