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