/// A trait that defines metarounding.
pub trait Metarounding {
    /// Given a point `x` of a relaxed space `relax(C) ⊂ Rⁿ` of `C`,
    /// `Metarounding::round` outputs a pair `(λ, [c1, c2, ...])`
    /// where `λ` is a probability vector over `C` satisfying 
    /// `Σ_{c ∈ C} λ[c] c[i] \leq α x[i]` for all `i \in [n]`
    /// and `[c1, c2, ...]` is a sequence of combinatorial vectors.
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>);
}

