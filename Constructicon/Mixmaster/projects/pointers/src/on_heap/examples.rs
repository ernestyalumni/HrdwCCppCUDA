#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn box_stores_i32_value_on_heap()
  {
    let b = Box::new(5);
    //println!("b = {}", b);

    assert_eq!(*b, 5);
    assert_eq!(*b as i32, 5);

  }
}