use grb;
use grb::prelude::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

// Since there might exist an entry of x whose value is significantly small,
// so that we cap the value by `THRESHOLD`.
pub const THRESHOLD: f64 = 1e-9;
pub const HEADER: &str = "time(ms),loss,bestval\n";


/// `Concept`, often called as combinatorial vector.
pub type Concept = Vec<f64>;


/// Set the environment parameter to the given `env`.
pub fn init_env(env: &mut Env) {
    env.set(param::OutputFlag, 0)
        .expect("Failed to set `OutputFlag` parameter to `0`");
    env.set(param::NumericFocus, 3)
        .expect("Failed to set `NumericFocus` parameter to `3`");
    // env.set(param::FeasibilityTol, 1e-9)
    //     .expect(
    //      "Failed to set `FeasibilityTol` parameter to `1e-9`"
    //     );
    // env.set(param::DualReductions, 0).unwrap();
}


/// Solves the primal problem:
/// ```txt
/// Minimize β
/// { β, λ }
/// sub. to. sum( λ[c] * c[i] ) <= β * x[i], for all i = 1, 2, ..., n,
///                 sum( λ[c] ) == 1,
///                        λ[c] >= 0,        for all c in comb_vectors.
/// ```
pub fn solve_primal(env: &Env, x: &[f64], comb_vectors: &[Concept])
    -> grb::Result<(f64, Vec<f64>)>
{
    assert!(comb_vectors.len() > 0);
    for c in comb_vectors {
        assert_eq!(x.len(), c.len());
        assert!(c.iter().zip(x).all(|(ci, xi)| *xi > 0.0 || *ci == 0.0));
    }

    let n_sets = comb_vectors.len();
    let mut model = Model::with_env("Solve LP", env)?;


    let beta = add_ctsvar!(model, name: "beta", bounds: 0.0..)?;
    let lambda = (0..n_sets).map(|c| {
            let name = format!("lambda[{c}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
        })
        .collect::<grb::Result<Vec<_>>>()?;



    let name = "sum( lambda[c] ) == 1";
    model.add_constr(&name, c!(lambda.iter().grb_sum() == 1_f64))?;
    model.update()?;
    for (i, xi) in x.iter().enumerate() {
        let lhs = lambda.iter()
            .zip(comb_vectors)
            .map(|(&lmd, cmb)| lmd * cmb[i])
            .grb_sum();
        let rhs = beta * *xi;
        let name = format!("Coordinate[{i}]");
        model.add_constr(&name, c!(lhs <= rhs))?;
    }
    model.set_objective(beta, Minimize)?;
    model.update()?;
    model.optimize()?;


    let objval = model.get_attr(attr::ObjVal)?;
    let lambda = lambda.iter()
        .map(|lmd| model.get_obj_attr(attr::X, lmd))
        .collect::<grb::Result<Vec<_>>>()?;


    Ok((objval, lambda))
}


/// Solve the dual problem
/// 
/// ```txt
/// Maximize γ
/// sub. to. sum( c[i] * ell[i] ) ≥ γ for all c in comb_vectors
///          sum( x[i] * ell[i] ) ≤ 1
///          0 ≤ ell[i] for all i = 1, 2, ..., n
/// ```
pub fn solve_dual(env: &Env, x: &[f64], comb_vectors: &[Concept])
    -> grb::Result<(f64, Vec<f64>)>
{
    let n_items = x.len();

    // DEBUG
    assert!(x.iter().all(|xi| (0.0..=1.0).contains(xi) && xi.is_finite()));
    assert!(comb_vectors.len() > 0);
    for i in 0..n_items {
        if x[i] <= 0.0 {
            if comb_vectors.iter().all(|c| c[i] > 0.0 && c[i].is_finite()) {
                panic!("Unbounded!");
            }
        }
    }



    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Dual Metarounding", env)?;

    // In this experiment, each combinatorial vector takes value in {0, 1}
    // so we can upper-bound `gamma` by the dimension of `x`.
    let ub = n_items as f64;
    let gamma = add_ctsvar!(model, name: "gamma", bounds: 0.0..ub)?;

    let ell = (0..n_items).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..)
        })
        .collect::<grb::Result<Vec<_>>>()?;
    model.update()?;


    // Add the constraint `sum( ell[i] * x[i] ) == 1`.
    let lx = grb_iproduct(x, &ell);
    let name = "sum( ell[i] * x[i] ) == 1";
    model.add_constr(&name, c!(lx == 1.0))?;
    for (i, cmb) in comb_vectors.iter().enumerate() {
        assert_eq!(cmb.len(), ell.len());
        let lhs = grb_iproduct(cmb, &ell);
        let name = format!("Comb[{i}]");
        model.add_constr(&name, c!(lhs >= gamma))?;
    }
    model.update()?;


    model.set_objective(gamma, Maximize)?;
    model.update()?;


    model.optimize()?;


    let status = model.status()?;
    assert_eq!(status, Status::Optimal, "Failed to solve dual LP");

    let objval = model.get_attr(attr::ObjVal)?;

    let ell = ell.iter()
        .map(|l| model.get_obj_attr(attr::X, l))
        .collect::<grb::Result<Vec<_>>>()?;


    Ok((objval, ell))
}


/// Returns the distance from `src` to `dst`.
/// Currently, I adopted the relative entropy as pseudo-distance.
pub fn distance(src: &[f64], dst: &[f64]) -> f64 {
    dst.iter()
        .zip(src)
        .map(|(&d, &s)| (d - s).powi(2))
        .sum::<f64>()
        .sqrt()
}


/// Returns the current `alpha`.
pub fn instant_alpha(x: &[f64], lambda: &[f64], comb_vectors: &[Concept])
    -> f64
{
    assert_eq!(lambda.len(), comb_vectors.len());

    let n = x.len();
    let mut expectation = vec![0.0; n];
    for (p, comb) in lambda.iter().zip(comb_vectors) {
        assert_eq!(expectation.len(), comb.len());
        for (ei, ci) in expectation.iter_mut().zip(comb) {
            *ei += p * ci;
        }
    }

    assert!((lambda.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    expectation.iter()
        .zip(x)
        .filter_map(|(&ei, &xi)| {
            if xi == 0.0 {
                if ei > 0.0 {
                    panic!("E[c(i)] <= alpha * x(i) is not satisfied!");
                }
                None
            } else {
                Some(ei / xi)
            }
        })
        .reduce(f64::max)
        .expect("There is no alpha satisfying the requirement.")
}


/// An iterator that returns a `dim`-dimensional loss vector
/// whose each entry is drawn i.i.d. 
/// from uniform distribution over `[0.0, 1.0).`
pub struct UniformLossIter {
    dim: usize,
    rng: StdRng,
    distribution: Uniform<f64>,
}


impl UniformLossIter {
    fn new(
        dim: usize,
        seed: u64,
    ) -> Self
    {
        let rng = StdRng::seed_from_u64(seed);
        let distribution = Uniform::from(0.0..1.0);
        Self {
            dim,
            rng,
            distribution,
        }
    }
}


impl Iterator for UniformLossIter {
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        let loss = self.distribution.sample_iter(&mut self.rng)
            .take(self.dim)
            .collect();
        Some(loss)
    }
}


/// Construct the loss vectors for all rounds `1..=T` a priori.
pub fn build_losses(
    dim: usize,
    n_rounds: usize,
    seed: u64,
) -> (Vec<f64>, UniformLossIter)
{
    let iter = UniformLossIter::new(dim, seed);
    let mut loss_sum = vec![0.0; dim];
    for loss in iter.take(n_rounds) {
        loss_sum.iter_mut()
            .zip(loss)
            .for_each(|(a, b)| { *a += b; });
    }
    let iter = UniformLossIter::new(dim, seed);
    (loss_sum, iter)
}



pub fn quadform(matrix: &Vec<Vec<f64>>, vector: &[f64]) -> f64 {
    let n = matrix.len();

    assert!(matrix.iter().all(|row| n == row.len()));
    assert_eq!(n, vector.len(), "Line 190");

    let mut ret = 0_f64;
    for i in 0..n {
        for j in 0..n {
            ret += matrix[i][j] * vector[i] * vector[j];
        }
    }
    ret
}


pub fn choose(dist: &[f64], rv: f64) -> usize {
    // println!("sum: {}", dist.iter().sum::<f64>());
    // assert!((1.0 - dist.iter().sum::<f64>()).abs() < 1e-6);
    let mut sum = 0.0;
    for (i, d) in dist.iter().copied().enumerate() {
        sum += d;
        if rv < sum { return i; }
    }
    return dist.len() - 1;
}


pub fn iproduct<S, T>(a: S, b: T) -> f64
    where S: AsRef<[f64]>,
          T: AsRef<[f64]>,
{
    let a = a.as_ref();
    let b = b.as_ref();

    a.into_iter()
        .zip(b)
        .map(|(&ai, &bi)| ai * bi)
        .sum::<f64>()
}


pub fn grb_iproduct<S, T>(a: S, b: T) -> Expr
    where S: AsRef<[f64]>,
          T: AsRef<[Var]>,
{
    let a = a.as_ref();
    let b = b.as_ref();

    a.into_iter()
        .zip(b)
        .map(|(&ai, &bi)| ai * bi)
        .grb_sum()
}
