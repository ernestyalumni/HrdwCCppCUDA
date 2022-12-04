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


#[derive(Debug)]
pub struct Rectangle {
  pub width: u32,
  pub height: u32,
}