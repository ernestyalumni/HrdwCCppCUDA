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