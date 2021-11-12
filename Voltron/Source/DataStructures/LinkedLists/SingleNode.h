//------------------------------------------------------------------------------
/// \file SingleNode.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating LinkedLists.
/// @ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/1/Single_node/src/Single_node.h
///-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LISTS_NODES_SINGLE_NODE_H
#define DATA_STRUCTURES_LISTS_NODES_SINGLE_NODE_H

namespace DataStructures
{
namespace Lists
{
namespace Nodes
{

//-----------------------------------------------------------------------------
/// \brief SingleNode with raw pointer for next.
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/1/Single_node/src/
/// \details No dynamic memory allocation; it's "static" memory.
//-----------------------------------------------------------------------------
template <typename T>
class SingleNode
{
	public:

		SingleNode(const T& element = T{}, SingleNode* next_node = nullptr);

		SingleNode(const SingleNode&) = default;

		T retrieve() const;
		SingleNode* next() const;

		// friend class SingleList<T>;
		// friend class CycleList<T>;
		// friend class SentinelList<T>;
		// friend class CycleSentinelList<T>;

		// If ptr is a pointer to a SingleNode<T> object,
		// in one of the friendly classes, you shoud:
		// 		use ptr->next_node 	to modify it
		// 		use ptr->next() 		to access it

	private:

		T element_;
		SingleNode* next_node_;
}; // class SingleNode

template <typename T>
SingleNode<T>::SingleNode(const T& element, SingleNode<T>* next_node):
	element_{element},
	next_node_{next_node}
{
	// empty constructor
}

template <typename T>
T SingleNode<T>::retrieve() const
{
	// Enter your implementation here
	return element_;
}

template <typename T>
SingleNode<T>* SingleNode<T>::next() const
{
	// Enter your implementation here
	return next_node_;
}

} // namespace Nodes
} // namespace Lists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LISTS_NODES_SINGLE_NODE_H