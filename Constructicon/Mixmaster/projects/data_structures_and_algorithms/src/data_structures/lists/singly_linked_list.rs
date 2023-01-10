//-------------------------------------------------------------------------------------------------
/// cf. https://github.com/tranzystorek-io/cons-list/blob/master/src/list.rs
//-------------------------------------------------------------------------------------------------

struct Node<T>
{
  value: T,
  next: Link<T>
}

type Link<T> = Option<Box<Node<T>>>;

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn node_constructs()
  {
    let x: Node<i32> = Node::<i32> { value: 42, next: None };

    assert_eq!(x.value, 42);
    match x.next
    {
      Some(_) => assert!(false),
      None => assert!(true),
    }
  }
}