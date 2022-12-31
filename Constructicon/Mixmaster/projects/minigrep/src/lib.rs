use std::error::Error;

pub struct Configuration
{
  pub query_: String,
  pub file_path_: String,
  pub ignore_case_: bool,
}

impl Configuration
{
  // Creating a Constructor for Configuration
  // cf. https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#creating-a-constructor-for-config
  fn new(args: &[String]) -> Result<Configuration, &'static str>
  {
    if args.len() < 3
    {
      panic!("not enough arguments");
    }

    let query = args[1].clone();
    let file_path = args[2].clone();

    Ok(Configuration { query_: query, file_path_: file_path, ignore_case_: false })
  }

  //-----------------------------------------------------------------------------------------------
  /// cf. https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#returning-a-result-instead-of-calling-panic
  /// panic! more appropriate for programming problem than a usage problem.
  /// Programmers expect new to never fail.
  /// \return Result with a Configuration instance in the success case and a &'static str in the
  /// error case. Our error values will always be string literals that have the 'static lifetime.
  //-----------------------------------------------------------------------------------------------
  pub fn build(args: &[String]) -> Result<Configuration, &'static str>
  {
    if args.len() < 3
    {
      return Err("not enough arguments");
    }

    let query = args[1].clone();
    let file_path = args[2].clone();

    // This allows us to do something like this, where "to" is what we're searching for:
    // IGNORE_CASE=1 cargo run -- to poem.txt
    let ignore_case = std::env::var("IGNORE_CASE").is_ok();

    Ok(Configuration { query_: query, file_path_: file_path, ignore_case_: ignore_case })
  }
}

//-------------------------------------------------------------------------------------------------
/// Array literal [...]
/// cf. https://doc.rust-lang.org/book/appendix-02-operators.html
//-------------------------------------------------------------------------------------------------
/*
pub fn parse_config(args: &[String]) -> (&str, &str)
{
  let query = &args[1];
  let file_path = &args[2];

  (query, file_path)
}
*/

fn parse_configuration(args: &[String]) -> Configuration
{
  // Clone makes a full copy of the data for the Configuration instance to own, which takes more
  // time and memory than storing a reference to the string data.
  let query = args[1].clone();
  let file_path = args[2].clone();

  Configuration { query_: query, file_path_: file_path, ignore_case_: false }
}

pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str>
{
  let mut results = Vec::new();

  for line in contents.lines()
  {
    if line.contains(query)
    {
      results.push(line);
    }
  }

  results
}

pub fn search_case_insensitive<'a>(query: &str, contents: &'a str,) -> Vec<&'a str>
{
  let query = query.to_lowercase();
  let mut results = Vec::new();

  for line in contents.lines()
  {
    if line.to_lowercase().contains(&query)
    {
      results.push(line);
    }
  }

  results
}

//-------------------------------------------------------------------------------------------------
/// cf. https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#extracting-logic-from-main
/// Hold all logic that isn't involved with setting up configuration or handling errors.
/// cf. https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#returning-errors-from-the-run-function
/// \return Result<T, E>, Result<(), Box<dyn Error>>
/// Returns unit type (); for error type, use trait object Box<dyn Error>, Box<dyn Error> means
/// function will return a type that implements the Error trait. dyn is short for dynamic.
//-------------------------------------------------------------------------------------------------
pub fn run(configuration: Configuration) -> Result<(), Box<dyn Error>>
{
  // Remove call to expect in favor of ? operator, in Ch. 9 of Rust book. Rather than panic! on an
  // error, ? will return error value from the current function for the caller to handle.
  let contents = std::fs::read_to_string(configuration.file_path_)?;
    // .expect("Should have been able to read the file")

  let results = if configuration.ignore_case_
  {
    search_case_insensitive(&configuration.query_, &contents)
  }
  else
  {
    search(&configuration.query_, &contents)
  };

  //for line in search(&configuration.query_, &contents)
  for line in results
  {
    println!("{line}");
  }
  
  // println!("With text:\n{contents}");

  Ok(())
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn one_result()
  {
    let query = "duct";
    let contents = "\
Rust:
safe, fast, productive.
Pick three.";

    assert_eq!(vec!["safe, fast, productive."], search(query, contents));
  }

  #[test]
  fn case_sensitive()
  {
    let query = "duct";
    let contents = "\
Rust:
safe, fast, productive.
Pick three.
Duct tape.";

    assert_eq!(vec!["safe, fast, productive."], search(query, contents));
  }

  #[test]
  fn case_insensitive()
  {
    let query = "rUsT";
    let contents = "\
Rust:
safe, fast, productive.
Pick three.
Trust me.";

    assert_eq!(
      vec!["Rust:", "Trust me."],
      search_case_insensitive(query, contents)
    );    
  }
}