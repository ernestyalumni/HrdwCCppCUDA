//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// EXAMPLE USAGE:
/// cargo test -- --nocapture
//------------------------------------------------------------------------------

/// 1. Array Indexing
/// Understanding how to navigate arrays is essential. Here are ten exercises,
//  sorted in increasing difficulty, that build upon each other:
//------------------------------------------------------------------------------

pub mod array_indexing {
    use std::fmt::Display;

    /// 1. Iterate Over an Array
    /// Write a function that prints each element in an array in order from the
    /// first to the last.
    ///
    /// # Generic Type Parameters
    /// - `I`: Any type that can be converted into an iterator (`IntoIterator`)
    /// - `C`: The output collection type that can be built from an iterator
    ///   (`FromIterator`)
    /// - `T`: The element type, must implement `Display` for printing
    /// 
    /// Trait Bounds specified by 'where'
    ///
    /// This function takes owned values and returns owned values (and both are
    /// not references).
    pub fn iterate_over_array<I, C, T>(
        array: I,
        is_print: bool,
    ) -> C
    where
        I: IntoIterator<Item = T>,
        C: FromIterator<T>,
        T: Display + Clone,
    {
        // Convert the input into an iterator and collect into Vec<T>
        // This consumes the input (takes ownership)
        let vec: Vec<T> = array.into_iter().collect();

        if is_print {
            for element in &vec {
                print!("{} ", element);
            }
            println!();
        }
        
        vec.into_iter().collect()
    }

    /// # Key Differences from Forward Iteration
    ///
    /// 1. **Bidirectional Iterator Requirement**: We need `DoubleEndedIterator`
    ///    - This allows iterating both forward and backward
    ///    - Not all iterators support this (e.g., some streaming iterators)
    ///
    /// 2. **`.rev()` Method**: Reverses the iterator direction
    ///    - Similar to C++'s `std::ranges::rbegin()` and `std::ranges::rend()`
    ///    - Works on any `DoubleEndedIterator`
    ///
    /// # Generic Type Parameters
    /// - `I`: Input that can be converted to a bidirectional iterator
    /// - `C`: Output collection type
    /// - `T`: Element type
    ///
    /// # Trait Bounds
    /// - `I: IntoIterator<Item = T>` - Must be iterable
    /// - `I::IntoIter: DoubleEndedIterator` - Iterator must support reverse
    ///   iteration
    /// - `C: FromIterator<T>` - Output must be buildable from iterator
    /// - `T: Display + Clone` - Elements must be printable and clonable
    pub fn iterate_over_array_in_reverse<I, C, T>(
        array: I,
        is_print: bool,
    ) -> C
    where
        I: IntoIterator<Item = T>,
        <I as IntoIterator>::IntoIter: std::iter::DoubleEndedIterator,
        C: FromIterator<T>,
        T: Display + Clone,
    {
        // Convert to iterator and reverse it
        // .rev() creates a Rev<I::IntoIter> that iterates backwards
        let iter = array.into_iter().rev();
        
        // Collect into Vec for printing (if needed) and final collection
        let vec: Vec<T> = iter.collect();

        if is_print {
            // Print in reverse order (which is now the order in vec)
            for element in &vec {
                print!("{} ", element);
            }
            println!();
        }
        
        // Convert to requested output type
        vec.into_iter().collect()
    }

    //--------------------------------------------------------------------------
    /// 3. Fetch Every Second Element
    /// 
    /// Write a function that accesses every other element in the array,
    //  starting from the first element. 
    //--------------------------------------------------------------------------
    pub fn fetch_second_element<I, C, T>(
        array: I,
        is_print: bool,
    ) -> C
    where
        I: IntoIterator<Item = T>,
        C: FromIterator<T>,
        T: Display + Clone,
    {
        // step_by(2) takes every 2nd element: indices 0, 2, 4, 6, ...
        let vec: Vec<T> = array.into_iter().step_by(2).collect();

        if is_print {
            for element in &vec {
                print!("{} ", element);
            }
            println!();
        }

        vec.into_iter().collect()
    }

    //--------------------------------------------------------------------------
    /// 4. Find the Index of a Target Element
    /// Write a function that searches for a specific element in an array and
    /// returns its index. If the element is not found, return -1.
    //--------------------------------------------------------------------------
    pub fn find_index_of_target_element<I, T>(
        array: I,
        target: T,
    ) -> i32
    where
        I: IntoIterator<Item = T>,
        T: PartialEq,
    {
        // .position() returns Option<usize< with index of first matching
        // element.
        // match handles Some, None cases.
        // |element| element == target is a closure. Closures are anonymous
        // functions that can capture variables from their environment.
        match array.into_iter().position(|element| element == target) {
            Some(index) => {
                index as i32
            }
            None => {
                -1
            }
        }
    }
}

//------------------------------------------------------------------------------
/// cargo test -- --nocapture
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::array_indexing;

    #[test]
    fn iterate_over_array_iterates_over_an_array() {

        let array = vec![1, 2, 3, 4, 5];
        // Option 1: Pass by value (moves array)
        // This is called "monomorphization" - Rust creates a specialized
        // version of the function for these specific types at compile time.
        let result: Vec<i32> =
            array_indexing::iterate_over_array(array.clone(), false);
        assert_eq!(result, array);

        // Option 2: Use iter().cloned() to convert &i32 to i32
        let array2 = vec![1, 2, 3, 4, 5];
        let result2: Vec<i32> = array_indexing::iterate_over_array(
        // array2.iter() creates an iterator over &i32 (references)
        // .cloned() converts &i32 -> i32 (clones the values)
        // Without .cloned(), we'd have:
        //   T = &i32 (a reference)
        //   But our function requires T: Clone, and &i32 doesn't work the same
        //   way.
            array2.iter().cloned(), 
            false
        );
        assert_eq!(result2, array2);

        // Uncomment this to print the result out.
        // let _result3: Vec<i32> =
        //     array_indexing::iterate_over_array(array2.iter().cloned(), true);
    }

    #[test]
    fn iterate_over_array_works_with_slices() {
        let array = [10, 20, 30];
        // Use iter().cloned() to convert &i32 references to owned i32 values
        let result: Vec<i32> = array_indexing::iterate_over_array(
            array.iter().cloned(), 
            false
        );
        assert_eq!(result, vec![10, 20, 30]);

        // Uncomment this to print the result out.
        // let _result2: Vec<i32> = array_indexing::iterate_over_array(
        //     array.iter().cloned(), 
        //     true
        // );
    }

    #[test]
    fn iterate_over_array_in_reverse_iterates_over_an_array_in_reverse() {
        
        let array = vec![1, 2, 3, 4, 5];
        let expected_result = vec![5, 4, 3, 2, 1];
        
        // Option 1: Pass by value (moves array)
        let result: Vec<i32> = array_indexing::iterate_over_array_in_reverse(
            array.clone(),
            false
        );
        assert_eq!(result, expected_result);
        
        // Option 2: Use iter().cloned() to convert &i32 to i32
        // The iterator from Vec::iter() implements DoubleEndedIterator, so
        // .rev() works on it
        let array2 = vec![1, 2, 3, 4, 5];
        let result2: Vec<i32> = array_indexing::iterate_over_array_in_reverse(
            array2.iter().cloned(),  // This iterator supports .rev()
            false
        );
        assert_eq!(result2, expected_result);
        
        // Test with printing enabled
        // Uncomment to see the output:
        // let _result3: Vec<i32> = array_indexing::iterate_over_array_in_reverse(
        //     array2.iter().cloned(),
        //     true  // This will print: 5 4 3 2 1
        // );
    }

    #[test]
    fn iterate_over_array_in_reverse_works_with_slices() {

        let array = [10, 20, 30];
        let result: Vec<i32> = array_indexing::iterate_over_array_in_reverse(
            array.iter().cloned(),
            false
        );
        assert_eq!(result, vec![30, 20, 10]);
        
        // Test with printing enabled
        // Uncomment to see the output:
        // let _result2: Vec<i32> = array_indexing::iterate_over_array_in_reverse(
        //     array.iter().cloned(),
        //     true  // This will print: 30 20 10
        // );
    }

    #[test]
    fn fetch_second_element_fetches_second_element() {
        let array = vec![1, 2, 3, 4, 5];
        let expected_result = vec![1, 3, 5];

        let result: Vec<i32> = array_indexing::fetch_second_element(
            array.iter().cloned(),
            false
        );
        assert_eq!(result, expected_result);

        // // Test with printing enabled
        // let _result2: Vec<i32> = array_indexing::fetch_second_element(
        //     array.iter().cloned(),
        //     // This will print: 1 3 5
        //     true
        // );
    }

    #[test]
    fn find_index_of_target_element_finds_index() {
        let array = vec![4, 7, 9, 1];
        
        assert_eq!(
            array_indexing::find_index_of_target_element(
                array.clone(),
                1,
            ),
            3
        );
        
        assert_eq!(
            array_indexing::find_index_of_target_element(
                array.clone(),
                9,
            ),
            2
        );
        
        assert_eq!(
            array_indexing::find_index_of_target_element(
                array.clone(),
                5,
            ),
            -1
        );
    }
}