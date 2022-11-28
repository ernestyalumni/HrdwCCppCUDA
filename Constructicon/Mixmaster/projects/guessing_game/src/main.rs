//-------------------------------------------------------------------------------------------------
/// \ref https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html
//-------------------------------------------------------------------------------------------------
// Rng trait defines methods that random number generators implement, and this trait must be in
// scope for us to use those methods.
use rand::Rng;
// Ordering type is another enum and has variants Less, Greater, Equal. There are the 3 outcomes
// that are possible when you compare 2 values.
use std::cmp::Ordering;
use std::io;

fn main()
{
  // Storing Values with Variables.
  // Create a *variable* to store user input.
  // = sign tells Rust we want to bind something to the variable now.
  // String::new is a function that returns a new instance of a String, String is a string type
  // provided by the standard library that's a growable, UTF-8 encoded bit of text.
  let mut guess = String::new();

  // In Rust, variables are immutable by default.
  let apples = 5; // immutable
  let mut bananas = 5; // mutable
  bananas = 6;

  println!("apples {} and bananas {}", apples, bananas);

  println!("Guess the number!");

  // rand::thread_rng function gives us the particular random number generator that we're going to
  // use: 1 that's local to current thread of execution and seeded by the OS.
  // gen_range method takes a range expression and generates a random number in the range.
  let secret_number = rand::thread_rng().gen_range(1..=100);

  println!("The secret number is: {secret_number}");

  loop
  {
    println!("Please input your guess.");

    // Receiving User Input. stdin function returns an instance of std::io::Stdin, which is a type
    // that represents a handle to the standard input for your terminal.
    // Next, line .read_line(&mut guess) calls read_line method on standard input handle to get
    // input from the user. 
    io::stdin()
      // & indicates this argument is a reference, which gives you a way to let multiple parts of
      // your code access 1 piece of data without needing to copy that data into memory multiple
      // times. Like variables, references are immutable by default.
      // Hence, you need &mut guess to make it a mutable reference.
      .read_line(&mut guess)
      // 
      .expect("Failed to read line");

    // Create a variable named guess. Rust allows us to *shadow* previous value of guess with a new
    // one. Shadowing lets us reuse the guess variable name.
    // Bind this new variable to expression guess.trim().parse().
    // parse method on strings converts string to another type.
    // The colon (:) after guess tells Rust we'll annotate the variable's type.
    //let guess: u32 = guess.trim().parse().expect("Please type a number:");

    // Handling Invalid Input
    let guess: u32 = match guess.trim().parse() {
      Ok(num) => num,
      Err(_) => continue,
    };

    println!("You guessed: {guess}");

    match guess.cmp(&secret_number)
    {
      Ordering::Less => println!("Too small!"),
      Ordering::Greater => println!("Too big!"),
      // cf. https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html
      // Quitting After a Correct Guess.
      Ordering::Equal => {
        println!("You win!");
        // Adding break line after You win! makes program exit loop.
        break;
      }
    }
  }
}
