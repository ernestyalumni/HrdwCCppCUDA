use std::fmt::Display;

pub fn largest_i32(list: &[i32]) -> &i32
{
  let mut largest = &list[0];

  for item in list
  {
    if item > largest
    {
      largest = item;
    }
  }

  largest
}

pub fn largest_char(list: &[char]) -> &char
{
  let mut largest = &list[0];

  for item in list
  {
    if item > largest
    {
      largest = item;
    }
  }

  largest
}

// Restricting types valid for T to only those that implement PartialOrd.
pub fn largest<T: std::cmp::PartialOrd>(list: &[T]) -> &T
{
  let mut largest = &list[0];

  for item in list
  {
    if item > largest
    {
      largest = item;
    }
  }

  largest
}

//-------------------------------------------------------------------------------------------------
// Traits: Defining Shared Behavior
// Defining a Trait
// We declare method signatures that describe the behaviors of the types that implement this
// trait.
// After the method signature, instead of providing an implementation within curly brackets, we use
// a semicolon. Each type implementing this trait must provide its own custom behavior for the body
// of the method.
//-------------------------------------------------------------------------------------------------
pub trait Summary
{
  fn summarize(&self) -> String;
}

// Implementing a Trait on a Type

pub struct NewsArticle
{
  pub headline: String,
  pub location: String,
  pub author: String,
  pub content: String,
}

impl Summary for NewsArticle
{
  fn summarize(&self) -> String
  {
    format!("{}, by {} ({})", self.headline, self.author, self.location)
  }
}

pub struct Tweet
{
  pub username: String,
  pub content: String,
  pub reply: bool,
  pub retweet: bool,
}

impl Summary for Tweet
{
  fn summarize(&self) -> String
  {
    format!("{}: {}", self.username, self.content)
  }
}

//-------------------------------------------------------------------------------------------------
/// \url https://doc.rust-lang.org/book/ch10-02-traits.html
/// Using Trait Bounds o Conditionally Implement Methods
/// \details By using a trait bound with an impl block that uses generic type parameters, we can
/// implement methods conditionally for types that implement the specified traits.
//-------------------------------------------------------------------------------------------------

struct Pair<T>
{
  x: T,
  y: T,
}

impl<T> Pair<T>
{
  fn new(x: T, y: T) -> Self
  {
    Self { x, y }
  }
}

impl<T: Display + PartialOrd> Pair<T>
{
  fn cmp_display(&self)
  {
    if self.x >= self.y
    {
      println!("The largest member is x = {}", self.x);
    }
    else
    {
      println!("The largest member is y = {}", self.y);
    }
  }
}

//-------------------------------------------------------------------------------------------------
/// cf. Lifetime Annotation Syntax, Lifetime Annotations in Function Signatures
/// \url https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
/// \details Lifetime annotations don't change how long any of the references live. Rather, they
/// describe the relationships of the lifetimes of multiple referneces to each other without
/// affecting lifetimes.
/// Just as a generic parameter can accept any type, function can accept references with any
/// lifetime by specifying a generic lifetime parameter.
///
/// Names of lifetime parameters must start with an apostrophe (') and are usually lowercase and
/// very short. Place lifetime parameter annotations after & of a reference, using space to
/// separate annotation from reference's type.
///
/// Declare generic lifetime parameters inside angle brackets between function name and parameter
/// list.
/// The signature expresses the following constraint: the returned reference will be valid as long
/// as both parameters are valid.
/// Lifetime of the reference returned by longest function is the same as the smaller of the
/// lifetimes of the values referred to by the function arguments.
///
//-------------------------------------------------------------------------------------------------
pub fn longest<'a>(x: &'a str, y: &'a str) -> &'a str
{
  if x.len() > y.len()
  {
    x
  }
  else
  {
    y
  }
}

//-------------------------------------------------------------------------------------------------
/// \ref https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
/// \brief Lifetime Annotations in Struct Definitions
/// \details So far, structs we'ved defined all hold owned types. We can define structs to hold
/// references, but in that case, we'd need to add lifetime annotation on every reference in the
/// struct's definition.
//-------------------------------------------------------------------------------------------------
pub struct ImportantExcerpt<'a>
{
  pub part_: &'a str,
}

//-------------------------------------------------------------------------------------------------
/// \ref https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
/// Lifetime elision rules - patterns programmed into Rust's analysis of referencess; the'yre a set
/// of particular cases that the compiler will consider, and if your code fits these cases, you
/// don't need to write the lifetimes explicitly.
//-------------------------------------------------------------------------------------------------
pub fn first_word(s: &str) -> &str
{
  let bytes = s.as_bytes();

  for (i, &item) in bytes.iter().enumerate()
  {
    if item == b' '
    {
      return &s[0..i];
    }
  }

  &s[..]
}