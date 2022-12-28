//-------------------------------------------------------------------------------------------------
/// cf. https://doc.rust-lang.org/book/ch12-01-accepting-command-line-arguments.html
/// The goal is to be able to run our program with cargo run, 2 hyphens to indicate the following
/// arguments are for our program rather than for `cargo`:
/// cargo run -- searchstring example-filename.txt
/// e.g.
/// $ cargo run --bin minigrep -- to Data/poem.txt
//-------------------------------------------------------------------------------------------------

use std::env;
use std::process;

use minigrep::Configuration;

fn main() {
  let args: Vec<String> = env::args().collect();
  let configuration = Configuration::build(&args).unwrap_or_else(|err|
    {
      //-------------------------------------------------------------------------------------------
      // \brief Writing Error Messsages to Standard Error Instead of Standard Output
      // \url https://doc.rust-lang.org/book/ch12-06-writing-to-stderr-instead-of-stdout.html
      // \details In most terminals, there are 2 kinds of output standard output ( stdout ) for
      // general information and standard error ( stderr ) for error messages. 
      // The distinction enables users to choose to direct the successful output of a program to a
      // file but still print error messages to the screen.
      //-------------------------------------------------------------------------------------------
     
      eprintln!("Problem parsing arguments: {err}");
      process::exit(1);
    });

  /*
  let query = &args[1];
  let file_path = &args[2];
  */

  println!("Searching for {}", configuration.query_);
  println!("In file {}", configuration.file_path_);

  //dbg!(args);

  if let Err(e) = minigrep::run(configuration)
  {
    eprintln!("Application Error occurred during run: {e}");
    process::exit(1);
  }
}
