mod constants;

fn main() {
    let x = 5;
    println!("The value of x is: {x}");

    let mut y = 6;
    println!("The value of y is: {y}");

    // This should yield a compiler error.
    // x = 6;
    println!("The value of x is: {x}");

    y = 7;
    println!("The value of y is: {y}");

    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
    println!("const value: {THREE_HOURS_IN_SECONDS}");

    println!("constant value : {}", constants::SECONDS_IN_YEAR);

    // Shadowing
    // By using let, we can perform a few transformations on a value, but have the variable be
    // immutable after those transformations have been completed.
    let xx = 8;
    // We effectively create a new variable.
    let xx = xx + 1;
    {
        let xx = xx * 2;
        println!("The value of xx in the inner scope is: {xx}");
    }

    println!("The value of xx after inner scope is: {xx}");

    // Shadowing; because we're effectively creating a new variable, we can change the type of the
    // value.
    let spaces = "    ";
    let spaces = spaces.len();
    println!("number of spaces: {spaces}");
}