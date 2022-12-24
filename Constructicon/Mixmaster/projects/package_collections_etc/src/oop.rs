// Debug annotation needed for assert_eq! macro.
#[derive(Debug)]
pub struct AveragedCollection {
  list_: Vec<i32>,
  average_: f64,
}

impl PartialEq for AveragedCollection
{
  fn eq(&self, other: &Self) -> bool
  {
    self.list_ == other.list_ && self.average_ == other.average_
  }
}

impl AveragedCollection
{
  pub fn add(&mut self, value: i32)
  {
    self.list_.push(value);
    self.update_average();
  }

  pub fn remove(&mut self) -> Option<i32>
  {
    let result = self.list_.pop();
    match result
    {
      Some(value) =>
      {
        self.update_average();
        Some(value)
      }
      None => None,
    }
  }

  pub fn average(&self) -> f64
  {
    self.average_
  }

  fn update_average(&mut self)
  {
    let total: i32 = self.list_.iter().sum();
    self.average_ = total as f64 / self.list_.len() as f64;
  }
}

pub mod gui;

#[cfg(test)]
mod tests
{
  // The tests module is a regular module that follows the usual visibility rules in Ch. 7,
  // "Paths for Referring to an Item in the Module Tree" of Rust book.
  // Because tests module is an inner module, we need to bring the code under test in the outer
  // module into scope of inner module.
  use super::*;

  //-----------------------------------------------------------------------------------------------
  /// \url https://doc.rust-lang.org/std/vec/struct.Vec.html
  //-----------------------------------------------------------------------------------------------
  #[test]
  fn construct_from_macro()
  {
    let a = AveragedCollection {list_: vec![1, 2, 3], average_: 42.69};
    let b = AveragedCollection {list_: Vec::from([1, 2, 3]), average_: 42.69};
    assert_eq!(a, b);
  }

 #[test]
  fn construct_initializing_every_element()
  {
    let a = AveragedCollection {list_: vec![0; 5], average_: 69.42};

  }
}