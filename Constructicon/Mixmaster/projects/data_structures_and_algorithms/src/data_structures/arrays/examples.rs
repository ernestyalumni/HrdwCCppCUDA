// cf. https://doc.rust-lang.org/std/ops/trait.Index.html

use std::ops::Index;

enum Nucleotide
{
  A,
  C,
  G,
  T,
}

struct NucleotideCount
{
  a: usize,
  c: usize,
  g: usize,
  t: usize,
}

impl Index<Nucleotide> for NucleotideCount
{
  type Output = usize;

  fn index(&self, nucleotide: Nucleotide) -> &Self::Output
  {
    match nucleotide
    {
      Nucleotide::A => &self.a,
      Nucleotide::C => &self.c,
      Nucleotide::G => &self.g,
      Nucleotide::T => &self.t,
    }
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn nucleotide_constructs()
  {
    let x = Nucleotide::A;
    match x
    {
      Nucleotide::A => assert!(true),
      Nucleotide::C => assert!(false),
      Nucleotide::G => assert!(false),
      Nucleotide::T => assert!(false),
    }
  }

  #[test]
  fn nucleotide_count_constructs()
  {
    let nucleotide_count = NucleotideCount {a: 14, c: 9, g: 10, t: 12};
    assert_eq!(nucleotide_count[Nucleotide::A], 14);
    assert_eq!(nucleotide_count[Nucleotide::C], 9);
    assert_eq!(nucleotide_count[Nucleotide::G], 10);
    assert_eq!(nucleotide_count[Nucleotide::T], 12);

  }
}