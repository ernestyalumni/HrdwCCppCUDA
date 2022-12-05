#[derive(Debug)]
pub enum IpAddrKind
{
  V4,
  V6,
}

//-------------------------------------------------------------------------------------------------
/// \ref https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html
/// Rather than an enum inside a struct, we can put data directly into each enum variant.
/// This new definition of enum says variants will have associated values.
/// You can put any kind of data inside an enum variant.
//-------------------------------------------------------------------------------------------------
#[derive(Debug)]
pub enum IpAddr
{
  V4(String),
  V6(String),
}

pub enum Message
{
  // variant has no data associated with it at all.
  Quit,
  // Has named fields like a struct does.
  Move { x: i32, y: i32 },
  // Includes a single String.
  Write(String),
  // Includes 3 i32 values.
  ChangeColor(i32, i32, i32),
}

impl Message
{
  pub fn call (&mut self)
  {
    //self = &mut Message::Quit;
  }
}

// So we can inspect the state in a minute.
#[derive(Debug)]
pub enum UsState
{
  Alabama,
  Alaska,
  California,
}

pub enum Coin
{
  Penny,
  Nickel,
  Dime,
  // Quarter,
  // Changed enum variant to hold data inside. Include a UsState value inside it.
  Quarter(UsState),
}

pub fn value_in_cents(coin: Coin) -> u8
{
  match coin
  {
    Coin::Penny => {
      println!("Lucky penny!");
      1
    }
    Coin::Nickel => 5,
    Coin::Dime => 10,
    Coin::Quarter(state) => {
      println!("State quarter from {:?}!", state);
      25
    }
  }
}

//-------------------------------------------------------------------------------------------------
// Matching with Option<T>
//-------------------------------------------------------------------------------------------------

pub fn plus_one(x: Option<i32>) -> Option<i32>
{
  match x
  {
    // Matches are exhaustive.
    None => None,
    // This gets caught.
    Some(42) =>
    {
      println!("Caught a 42 value");
      Some(42)
    }
    // Some(5) match Some(i) - i binds to value contained in Some, so i takes value 5.
    Some(i) => Some(i + 1),
    // This doesn't get caught.
    Some(0) =>
    {
      println!("Caught a 0 value");
      Some(0)
    }
  }
}