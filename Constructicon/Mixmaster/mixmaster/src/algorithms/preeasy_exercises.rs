pub mod array_indexing
{
    use std::fmt::Display;

    pub fn iterate_over_slice<T: Display + Clone>(
        array: &[T],
        is_print: bool) -> Vec<T>
    {
        let mut result = Vec::new();
        for element in array
        {
            if is_print
            {
                println!("{}", element);
            }
            result.push(element.clone());
        }
        if is_print
        {
            println!();
        }
        result
    }

    pub fn iterate_over_array<I>(array: I, is_print: bool) -> Vec<I::Item>
    where
        I: IntoIterator,
        I::Item: Display,
    {
        let mut result = Vec::new();
        for element in array
        {
            if is_print
            {
                print!("{} ", element);
            }
            result.push(element);
        }
        if is_print
        {
            println!();
        }
        result
    }
}

//----------------------------------------------------------------------------
/// Unit Tests
/// EXAMPLE USAGE:
/// cargo test preeasy_exercises
/// cargo test preeasy_exercises -- --nocapture
//----------------------------------------------------------------------------
#[cfg(test)]
mod tests
{
    use super::array_indexing;

    #[test]
    fn test_iterate_over_slice_iterates_over_an_array()
    {
        let array = vec![1, 2, 3, 4, 5];
        let result = array_indexing::iterate_over_slice(&array, false);
        assert_eq!(result, array);
    }

    #[test]
    fn test_iterate_over_slice_with_printing()
    {
        let array = vec![1, 2, 3, 4, 5];
        // Test that it doesn't panic when printing
        let result = array_indexing::iterate_over_slice(&array, true);
        assert_eq!(result, array);
    }

    #[test]
    fn test_iterate_over_slice_with_empty_array()
    {
        let array: Vec<i32> = vec![];
        let result = array_indexing::iterate_over_slice(&array, false);
        assert_eq!(result, array);
    }

    #[test]
    fn test_iterate_over_slice_with_single_element()
    {
        let array = vec![42];
        let result = array_indexing::iterate_over_slice(&array, false);
        assert_eq!(result, array);
    }

    #[test]
    fn test_iterate_over_array_with_vec()
    {
        let array = vec![1, 2, 3, 4, 5];
        let result = array_indexing::iterate_over_array(array.clone(), false);
        assert_eq!(result, array);
    }

    #[test]
    fn test_iterate_over_array_with_printing()
    {
        let array = vec![1, 2, 3, 4, 5];
        // Test that it doesn't panic when printing
        let result = array_indexing::iterate_over_array(array.clone(), true);
        assert_eq!(result, array);
    }
}