#[derive(Debug, PartialEq, Copy, Clone)]
enum ShirtColor
{
  Red,
  Blue,
}

struct Inventory
{
  shirts: Vec<ShirtColor>,
}

impl Inventory
{
  fn giveaway(&self, user_preference: Option<ShirtColor>) -> ShirtColor
  {
    // The unwrap_or_else method on Option<T> takes 1 argument: a *closure* without any arguments
    // that returns a value T (same type stored in the Some variant of the Option<T>)
    user_preference.unwrap_or_else(|| self.most_stocked())
  }

  fn most_stocked(&self) -> ShirtColor
  {
    let mut num_red = 0;
    let mut num_blue = 0;

    for color in &self.shirts
    {
      match color
      {
        ShirtColor::Red => num_red += 1,
        ShirtColor::Blue => num_blue +=1,
      }
    }
    if num_red > num_blue
    {
      ShirtColor::Red
    }
    else
    {
      ShirtColor::Blue
    }
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  // cf. https://stackoverflow.com/questions/46378637/how-to-make-a-variable-with-a-scope-lifecycle-for-all-test-functions-in-a-rust-t

  struct Setup
  {
    store: Inventory,
  }

  impl Setup
  {
    fn new() -> Self
    {
      Self
      {
        store: Inventory {
          shirts: vec![ShirtColor::Blue, ShirtColor::Red, ShirtColor::Blue],
        }
      }
    }
  }

  #[test]
  fn user_with_preference_gets_preference()
  {
    let setup = Setup::new();

    // cf. https://doc.rust-lang.org/std/option/
    // Type Option represents an optional value: every Option is either Some and contains a value,
    // or None, and doesn't.
    let user_pref1 = Some(ShirtColor::Red);
    let giveaway1 = setup.store.giveaway(user_pref1);

    assert_eq!(giveaway1, ShirtColor::Red);
  }

  #[test]
  fn user_with_no_preference_gets_most_stocked()
  {
    let setup = Setup::new();

    let user_pref2 = None;
    let giveaway2 = setup.store.giveaway(user_pref2);

    assert_eq!(giveaway2, ShirtColor::Blue);
  }
}