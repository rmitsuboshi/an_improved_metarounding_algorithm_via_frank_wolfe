/// A trait that defines metarounding.
pub trait Metarounding {
    /// Given a point `x` of 
    /// a relaxed space `P \subset \mathbb{R}^n` of `C`,
    /// `Metarounding::round` outputs 
    /// a probability vector `lambda` over `C`
    /// satisfying 
    /// `\sum_{c \in C} \lambda_c c_i \leq x_i` for all `i \in [n]`.
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>);
}

