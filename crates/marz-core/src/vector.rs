//! Sparse vector for BM25 field vectors and query vectors.

use std::cell::Cell;

/// A sparse vector stored as a flat array of `(index, value)` pairs.
///
/// This mirrors lunr.Vector: elements are kept sorted by index for efficient
/// dot-product and similarity calculations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vector {
    elements: Vec<(usize, f64)>,
    magnitude: Cell<Option<f64>>,
}

impl Vector {
    /// Create an empty vector.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            magnitude: Cell::new(None),
        }
    }

    /// Insert a value at the given index. Panics if the index already exists.
    pub fn insert(&mut self, index: usize, value: f64) {
        let pos = self.position_for_index(index);
        if pos < self.elements.len() && self.elements[pos].0 == index {
            panic!("duplicate index {}", index);
        }
        self.elements.insert(pos, (index, value));
        self.magnitude.set(None);
    }

    /// Insert or update a value at the given index.
    pub fn upsert<F>(&mut self, index: usize, value: f64, f: F)
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let pos = self.position_for_index(index);
        if pos < self.elements.len() && self.elements[pos].0 == index {
            let current = self.elements[pos].1;
            self.elements[pos].1 = f(current, value);
        } else {
            self.elements.insert(pos, (index, value));
        }
        self.magnitude.set(None);
    }

    /// Compute the Euclidean magnitude.
    pub fn magnitude(&self) -> f64 {
        if let Some(m) = self.magnitude.get() {
            return m;
        }
        let sum: f64 = self.elements.iter().map(|(_, v)| v * v).sum();
        let m = sum.sqrt();
        self.magnitude.set(Some(m));
        m
    }

    /// Dot product with another vector.
    pub fn dot(&self, other: &Self) -> f64 {
        let mut product = 0.0;
        let mut i = 0;
        let mut j = 0;
        while i < self.elements.len() && j < other.elements.len() {
            let (a_idx, a_val) = self.elements[i];
            let (b_idx, b_val) = other.elements[j];
            match a_idx.cmp(&b_idx) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    product += a_val * b_val;
                    i += 1;
                    j += 1;
                }
            }
        }
        product
    }

    /// Cosine similarity with another vector.
    ///
    /// **Note**: this implements lunr's asymmetric normalization:
    /// `dot(self, other) / magnitude(self)`. Only the left-hand magnitude is
    /// used as the denominator, matching lunr.js.
    pub fn similarity(&self, other: &Self) -> f64 {
        let mag = self.magnitude();
        if mag == 0.0 {
            return 0.0;
        }
        self.dot(other) / mag
    }

    /// Return the underlying elements.
    pub fn elements(&self) -> &[(usize, f64)] {
        &self.elements
    }

    /// Serialize to a flat array of `[index, value, ...]`.
    pub fn to_vec(&self) -> Vec<f64> {
        self.elements
            .iter()
            .flat_map(|(idx, val)| vec![*idx as f64, *val])
            .collect()
    }

    /// Build a vector from a flat serialized array.
    pub fn from_vec(elements: &[f64]) -> Self {
        let mut vec = Self::new();
        for chunk in elements.chunks_exact(2) {
            vec.insert(chunk[0] as usize, chunk[1]);
        }
        vec
    }

    fn position_for_index(&self, index: usize) -> usize {
        if self.elements.is_empty() {
            return 0;
        }
        match self.elements.binary_search_by_key(&index, |&(idx, _)| idx) {
            Ok(pos) => pos,
            Err(pos) => pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_dot_product() {
        let mut a = Vector::new();
        a.insert(0, 1.0);
        a.insert(2, 2.0);

        let mut b = Vector::new();
        b.insert(1, 3.0);
        b.insert(2, 4.0);

        assert_eq!(a.dot(&b), 8.0);
    }

    #[test]
    fn vector_asymmetric_similarity() {
        let mut a = Vector::new();
        a.insert(0, 3.0);
        a.insert(1, 4.0);

        let mut b = Vector::new();
        b.insert(0, 1.0);

        // dot = 3, |a| = 5 -> similarity = 0.6
        assert!((a.similarity(&b) - 0.6).abs() < 1e-9);
    }
}
