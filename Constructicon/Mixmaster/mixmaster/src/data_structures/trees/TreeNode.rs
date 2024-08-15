// https://doc.rust-lang.org/std/cell/struct.RefCell.html
// Mutable memory location with dynamically checked borrow rules.
use std::cell::RefCell;
use std::collections::VecDeque;
// https://doc.rust-lang.org/std/rc/struct.Rc.html
// https://doc.rust-lang.org/std/rc/
// A single-threaded reference-counting pointer. 'Rc' stands for 'Reference
// Counted.'
// Rc<T> Provides shared ownership of a value of type T, allocated in the heap.
// Invoking clone on Rc produces a new pointer to same allocation in the heap.
// Rc uses non-atomic reference counting. If you need multi-threaded, atomic
// reference counting, use sync::Arc.
use std::rc::Rc;

type NodeRef<T> = Rc<RefCell<TreeNode<T>>>;

#[derive(Debug)]
struct TreeNode<T>
{
	value: T,
  parent: Option<NodeRef<T>>,
  children: Vec<NodeRef<T>>,
}

impl<T: std::fmt::Debug> TreeNode<T>
{
  pub fn new(value: T) -> NodeRef<T>
  {
    Rc::new(RefCell::new(TreeNode
    {
      value,
      parent: None,
      children: Vec::new(),
    }))
  }

  //----------------------------------------------------------------------------
  /// \param parent: &NodeRef<T>
  /// Using & indicates function takes a reference, rather than taking ownership
  /// of it.
  //----------------------------------------------------------------------------
  pub fn add_child(parent: &NodeRef<T>, value: T) -> NodeRef<T>
  {
    let child = TreeNode::new(value);
    child.borrow_mut().parent = Some(Rc::clone(parent));
    parent.borrow_mut().children.push(Rc::clone(&child));
    child
  }

  pub fn is_root(&self) -> bool
  {
    self.parent.is_none()
  }

  pub fn is_leaf(&self) -> bool
  {
    self.children.is_empty()
  }

  //---------------------------------------------------------------------------- 
  /// \return The degree (number of children) of the node.
  //---------------------------------------------------------------------------- 
  pub fn degree(&self) -> usize
  {
    self.children.len()
  }

  pub fn size(node: &NodeRef<T>) -> usize
  {
    1 + node.borrow().children.iter().map(
      |child| TreeNode::size(child)).sum::<usize>()
  }

  pub fn depth_first_traversal(node: &NodeRef<T>) -> Vec<T>
  where
    T: Clone,
  {
    let mut results = Vec::new();
    // Iterative approach with stack.
    let mut stack = VecDeque::new();
    stack.push_front(Rc::clone(node));

    while let Some(current) = stack.pop_front()
    {
      let current_ref = current.borrow();
      results.push(current_ref.value.clone());

      for child in &current_ref.children
      {
        stack.push_front(Rc::clone(child));
      }
    }

    results
  }

  pub fn breadth_first_traversal(root: &NodeRef<T>) -> Vec<T>
  where
    T: Clone,
  {
    let mut results = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(Rc::clone(root));

    while let Some(node) = queue.pop_front()
    {
      let node_ref = node.borrow();
      results.push(node_ref.value.clone());

      for child in &node_ref.children
      {
        queue.push_back(Rc::clone(child));
      }
    }

    results
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn test_tree_creation()
  {
    let root = TreeNode::new(42);

    // Otherwise, root.is_root() results in error: no method found for
    // Rc<RefCell<TreeNode<{integer}>>>
    assert!(root.borrow().is_root());
    assert!(root.borrow().is_leaf());
    assert_eq!(root.borrow().degree(), 0);
    assert_eq!(root.borrow().value, 42);
    assert_eq!(TreeNode::size(&root), 1);
  }

  #[test]
  fn test_size()
  {
    let root = TreeNode::new(1);
    let child1 = TreeNode::add_child(&root, 2);
    let child2 = TreeNode::add_child(&root, 3);
    TreeNode::add_child(&child1, 4);
    TreeNode::add_child(&child1, 5);
    TreeNode::add_child(&child2, 6);

    assert!(root.borrow().is_root());
    assert!(!child1.borrow().is_root());
    assert!(!child2.borrow().is_root());
    assert!(!root.borrow().is_leaf());
    assert!(!child1.borrow().is_leaf());
    assert!(!child2.borrow().is_leaf());

    assert_eq!(root.borrow().degree(), 2);
    assert_eq!(child1.borrow().degree(), 2);
    assert_eq!(child2.borrow().degree(), 1);

    assert_eq!(root.borrow().value, 1);
    assert_eq!(child1.borrow().value, 2);
    assert_eq!(child2.borrow().value, 3);

    assert_eq!(TreeNode::size(&root), 6);
    assert_eq!(TreeNode::size(&child1), 3);
    assert_eq!(TreeNode::size(&child2), 2);
  }

  #[test]
  fn test_breadth_first_traversal()
  {
    let root = TreeNode::new(1);
    let child1 = TreeNode::add_child(&root, 2);
    let child2 = TreeNode::add_child(&root, 3);
    TreeNode::add_child(&child1, 4);
    TreeNode::add_child(&child1, 5);
    TreeNode::add_child(&child2, 6);

    let bfs_result = TreeNode::breadth_first_traversal(&root);
    assert_eq!(bfs_result, vec![1, 2, 3, 4, 5, 6]);

    let bfs_result = TreeNode::breadth_first_traversal(&child1);
    assert_eq!(bfs_result, vec![2, 4, 5]);

    let bfs_result = TreeNode::breadth_first_traversal(&child2);
    assert_eq!(bfs_result, vec![3, 6]);
  }

  #[test]
  fn test_depth_first_traversal()
  {
    let root = TreeNode::new(1);
    let child1 = TreeNode::add_child(&root, 2);
    let child2 = TreeNode::add_child(&root, 3);
    TreeNode::add_child(&child1, 4);
    TreeNode::add_child(&child1, 5);
    TreeNode::add_child(&child2, 6);

    let dfs_result = TreeNode::depth_first_traversal(&root);
    assert_eq!(dfs_result, vec![1, 3, 6, 2, 5, 4]);

    let dfs_result = TreeNode::depth_first_traversal(&child1);
    assert_eq!(dfs_result, vec![2, 5, 4]);

    let dfs_result = TreeNode::depth_first_traversal(&child2);
    assert_eq!(dfs_result, vec![3, 6]);
  }
}