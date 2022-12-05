// https://doc.rust-lang.org/book/ch08-01-vectors.html#using-an-enum-to-store-multiple-types
// Define an enum whose variants will hold different value types, and all enum variants will be
// considered the same type: that of the enum. Then we can create a vector to hold that enum.
pub enum SpreadsheetCell
{
  Int(i32),
  Float(f64),
  Text(String),
}

//-------------------------------------------------------------------------------------------------
// \url https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
// Propagating Errors
//-------------------------------------------------------------------------------------------------


pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
