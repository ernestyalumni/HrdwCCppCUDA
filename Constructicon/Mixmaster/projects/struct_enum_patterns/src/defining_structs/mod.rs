//-------------------------------------------------------------------------------------------------
/// \url https://doc.rust-lang.org/book/ch05-01-defining-structs.html
/// To define a struct, enter keyword struct, and name entire struct.
/// Define names and types of pieces of data, which we call fields.
//-------------------------------------------------------------------------------------------------
pub struct User
{
  pub active: bool,
  pub username: String,
  pub email: String,
  pub sign_in_count: u64,
}

pub fn build_user(email: String, username: String) -> User {
  User {
    email: email,
    username: username,
    active: true,
    sign_in_count: 1,
  }
}

// Using the Field Init Shorthand
pub fn build_user2(email: String, username: String) -> User {
  User {
    // Use the field init shorthand syntax
    email,
    username,
    active: true,
    sign_in_count: 1,
  }
}

pub struct Color(pub i32, pub i32, pub i32);

pub struct Point(pub i32, pub i32, pub i32);

// outer attribute, Debug trait, enables us to print out struct in a way that's useful for
// developers
#[derive(Debug)]
pub struct Rectangle {
  pub width: u32,
  pub height: u32,
}

//-------------------------------------------------------------------------------------------------
/// Method Syntax
/// Methods are defined within context of struct (or enum or trait object), and their first
/// parameter is always self, which represents the instance of struct the method is being called
/// on.
/// To define function within the context of Rectangle, start an impl (implementation) block for
/// Rectangle. Everything within this impl block will be associated with the Rectangle type.
///
/// All functions defined within impl block are called associated functions because they're
/// associated with type named after impl.
//-------------------------------------------------------------------------------------------------
impl Rectangle
{
  // &self is actually short for self: &Self, type Self is an alias for the type that the impl
  // block is for.
  pub fn area(&self) -> u32
  {
    self.width * self.height
  }

  pub fn can_hold(&self, other: &Rectangle) -> bool
  {
    self.width > other.width && self.height > other.height
  }  
}

// Each struct is allowed to have multiple impl blocks.
impl Rectangle
{
  pub fn square(size: u32) -> Self {
    Self
    {
      width: size,
      height: size,
    }
  }
}