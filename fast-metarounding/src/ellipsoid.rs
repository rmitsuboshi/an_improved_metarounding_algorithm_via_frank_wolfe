//! A Metarounding algorithm based on Ellipsoid Method.
//! 
//! Given the input vector `x ∈ [0, 1]^n`,
//! the ellipsoid method finds a feasible solution `λ ∈ [0, 1]^C` s.t.
//! 1. `sum( λ[c] c[i] ) ≤ α x[i]` for all `i = 1, 2, ..., n`,
//! 2. `sum( λ[c] ) ≥ 1`, and
//! 3. `λ[c] ≥ 0` for all `c ∈ C`,
//! where `α > 0` is the upper-bound of the approximation ratio,
//! guaranteed by the `Oracle`.
//!
//! Setting the dummy objective `sum( 0 * λ[c] )`,
//! we can formulate the above feasibility problem as a linear program.
//! ```txt
//!     Minimize sum( 0 * λ[c] )
//!     sub. to. sum( λ[c] c[i] ) ≤ α x[i] for all i = 1, 2, ..., n,
//!              sum( λ[c] ) ≥ 1,
//!              λ[c] ≥ 0 for all c ∈ C.
//! ```
//! This problem has exponentially many variables 
//! since `C` is a huge set in general.
//! Thus, we consider the dual problem.
//! ```txt
//!     Maximize γ - α sum( l[i] * x[i] )
//!     sub. to. sum( l[i] * c[i] ) ≥ γ for all c ∈ C,
//!              l[c] ≥ 0 for all c ∈ C,
//!              γ ≥ 0.
//! ```
//!
//!
//! Now we introduce the following variables to simplify the above dual problem.
//!
//! - `N = n + 1`,
//! - `z = [   l[1],   l[2], ...,   l[n], γ ] ∈ [0, ∞)^N`,
//! - `a = [ -αx[1], -αx[2], ..., -αx[n], 1 ] ∈ R^N`,
//! - `D ∈ R^{CxN}` be the matrix 
//!     whose row vector is `[ -c[1], -c[2], ..., -c[n], 1 ] ∈ R^N`.
//!
//! Then, we can write the dual problem as
//! ```txt
//!     Maximize sum( a[i] * z[i] )
//!     sub. to. Dz ≤ 0,
//!               z ≥ 0.
//! ```
//!
//! Recall that the primal problem achieves the optimal value `0`.
//! Thus, we can convert the above maximization problem 
//! into a feasibility problem.
//! Define the matrix `E` as:
//! ```txt
//!
//!     E = [
//!             D,
//!             a[1], a[2], ..., a[N]
//!         ]
//! ```
//! That is `E` is a matrix whose first `C` rows is the one in `D`,
//! and the last row is the coefficient vector of the objective.
//! With this notation,
//! we can write the dual feasibility problem.
//! ```txt
//!     Find z ∈ R^N such that
//!     1. Ez ≤ 0,
//!     2.  z ≥ 0.
//! ```
//!
//! Note that, by LP duality,
//! the constraint corresponding to the objective is always satisfied.
use grb::prelude::*;
use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct Ellipsoid<'a> {
    /// The number of items.
    _n_items: usize,


    /// The number of sets.
    n_sets: usize,


    /// Tolerance parameter.
    tolerance: f64,


    /// The approximation algorithm that returns a vector in `C`.
    oracle: &'a Oracle,


    /// approximation ratio.
    alpha: f64,


    /// Combinatorial vectors
    comb_vectors: Vec<Vec<f64>>,


    /// Gurobi Env
    env: Env,
}


impl<'a> Ellipsoid<'a> {
    pub fn new(
        alpha: f64,
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        env.set(param::NumericFocus, 3).unwrap();
        // env.set(param::DualReductions, 0).unwrap();

        let (_n_items, n_sets) = oracle.shape();

        let comb_vectors = Vec::new();

        Self {
            _n_items,
            n_sets,
            tolerance,
            alpha,
            oracle,
            comb_vectors,
            env,
        }
    }

    fn max_iter(&self, radius: f64) -> usize {
        let n = self.n_sets as f64;
        let h = 64_f64; // # of bits to represent a number
        let lg = (radius * n.sqrt() * h).ln();
        let coef = 2.0 * n * (n + 1.0);

        (coef * lg).ceil() as usize
    }


    /// This method updates the ellipsoid `P` centered at `x`
    /// according to the following rules:
    ///
    /// 1. Get the gradient `g ∈ ∂f(x)`.
    ///    (In this code, `g` is given as an input.)
    ///
    /// 2. If `√gPg ≤ ε`, return `x`.
    ///
    /// 3. Update ellipsoid P.
    ///
    ///     3-a. q := g / ( √gPg )
    ///     3-b. x := x - ( Pq / (n + 1) )
    ///     3-c. P := ( n²/(n²-1) ) ( P - ( 2/(n+1) Pq qP ) )
    ///
    fn update_ellipsoid(
        &mut self,
        center: &mut [f64],
        p: &mut Vec<Vec<f64>>,
        g: &[f64],
    )
    {
        let n = self.n_sets + 1;
        assert_eq!(n, center.len());
        assert_eq!(n, g.len());

        // `root_gpg = √gPg`
        let root_gpg = quadform(p, &g[..]).sqrt();
        assert!(root_gpg > 0.0);

        // ----------------------------------------------------
        // 3-a. `q = g / ( √gPg )`
        let q = g.iter()
            .map(|gi| gi / root_gpg)
            .collect::<Vec<_>>();

        // ----------------------------------------------------
        // 3-b. `x = x - ( Pq / (n+1) )`
        let mut pq = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                pq[i] += p[i][j] * q[j];
            }
        }


        center.iter_mut()
            .zip(&pq)
            .for_each(|(c, &pqi)| { *c -= pqi / (n as f64 + 1.0); });


        // ----------------------------------------------------
        // 3-c. P := ( n²/(n²-1) ) ( P - ( 2/(n+1) Pq qP ) )
        let coef = n.pow(2) as f64 / (n.pow(2) as f64 - 1.0);
        let mut p_new = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let tmp = (2.0 / (n as f64 + 1.0)) * pq[i] * pq[j];
                // let tmp = (2.0 / (n as f64 + 1.0)) * pq[i] * qp[j];
                p_new[i][j] = coef * (p[i][j] - tmp);
            }
        }
        *p = p_new;
    }
}


impl<'a> Metarounding for Ellipsoid<'a> {
    /// This method solves the following problem [Carr & Vempala, '02]:
    ///
    /// - Input
    ///     - `x ∈ relax(C)`,
    ///     - `alpha` (not the exact one, an upper-bound).
    /// - Output
    ///     A solution to the following problem.
    ///
    ///     ```txt
    ///     Minimize γ + α sum( l[i] * x[i] )
    ///       l, γ
    ///
    ///     sub. to. sum( l[i] * c[i] ) + γ ≥ 1, for all c ∈ C,
    ///                                l[i] ≥ 0, for all i ∈ [n],
    ///                                   γ ≥ 0.
    ///     ```
    /// Note that the optimal value of the above problem is 1.
    /// Without loss of generality, they add the following constraint:
    /// ```txt
    ///     α sum( l[i] * x[i] ) + γ ≤ 1
    /// ```
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        // Initialize.
        let x = x.as_ref();

        // let mut obj = vec![0.0; self.n_sets];
        // obj.push(-1.0);
        let mut obj = x.iter()
            .map(|xi| self.alpha * xi)
            .collect::<Vec<_>>();
        obj.push(1.0);

        // In `ellipsoid_metarounding.m`,
        // N = n + 1
        let n1 = self.n_sets + 1;

        // **One of our results**
        // The optimal solution `ell` is in `[0, M]^n`,
        // where `M = max( 1/x[i] | x[i] ≠ 0 )`.
        // Thus, the ball with radius `M` contains optimal solutions.
        let radius = x.iter()
            .filter_map(|&xi| if xi == 0.0 { None } else { Some(1.0 / xi) })
            .reduce(f64::max)
            .expect("x should have a non-zero element.")
            .powi(2);


        // Construct a initial ellipsoid
        //
        // In `ellipsoid_metarounding.m`,
        // P = R * eye(N)
        // y = zeros(N, 1);
        let mut ellipsoid = (0..n1).map(|i| {
                let mut row = vec![0_f64; n1];
                row[i] = radius;
                row
            }).collect::<Vec<_>>();

        // Center of the initial ellipsoid
        let mut center = vec![0_f64; n1];

        // TODO! add max rounds.
        let max_rounds = self.max_iter(radius);
        println!("[ROUND] {max_rounds}");

        for round in 1..=max_rounds {
            if round % 2_000 == 0 {
                println!("[CURRENT] {round}");
            }
            // If the non-negativity constraint is violated,
            // update the ellipsoid based on the separating plane.
            while let Some(i) = center.iter().position(|c| *c < 0.0) {
                let mut sep = vec![0.0; n1];
                sep[i] = -1.0;
                self.update_ellipsoid(&mut center, &mut ellipsoid, &sep);
            }

            // let obj_val = obj.iter()
            //     .zip(&center)
            //     .map(|(a, b)| a * b)
            //     .sum::<f64>();
            // if obj_val >= 1.0 + self.tolerance {
            //     println!("obj val. {obj_val}");
            //     // Since the optimal value is 1,
            //     // `α sum( ell[i] * x[i] ) + γ ≤ 1` must be satisfied.
            //     // we can use the separating hyperplane.
            //     self.update_ellipsoid(&mut center, &mut ellipsoid, &obj);
            //     continue;
            // }



            let ret = self.oracle.separation_oracle(
                self.alpha, self.tolerance, &center
            );
            match ret {
                Ok(()) => {
                    let q = quadform(&ellipsoid, &obj).sqrt();
                    if q <= self.tolerance {
                        println!("[ FIN ] convergent");
                        break;
                    }
                    self.update_ellipsoid(&mut center, &mut ellipsoid, &obj);
                },
                Err(sep) => {
                    // Check the numerical stability
                    let q = quadform(&ellipsoid, &sep);
                    if q < 0_f64 {
                        println!("[ABORT] xAx < 0");
                        break;
                    }
                    if q == 0_f64 {
                        println!("[ABORT] xAx == 0");
                        break;
                    }


                    self.update_ellipsoid(&mut center, &mut ellipsoid, &sep);

                    let c = sep[..self.n_sets].into_iter()
                        .map(|s| -s)
                        .collect::<Vec<_>>();
                    self.comb_vectors.push(c);
                }
            }
        }
        println!("# of comb: {}", self.comb_vectors.len());
        let (_, lambda) = solve_primal_ub(&self.env, x, self.alpha, &self.comb_vectors);

        let comb_vectors = std::mem::take(&mut self.comb_vectors);
        (lambda, comb_vectors)
    }
}


/// Solves the primal problem [Carr & Vempala, '02]:
/// 
/// ```txt
/// Maximize sum( λ[c] )
/// sub. to. sum( λ[c] * c[i] ) ≤ α * x[i], for all i = 1, 2, ..., n,
///                 sum( λ[c] ) ≤ 1,
///                        λ[c] ≥ 0,        for all c in comb_vectors.
/// ```
pub fn solve_primal_ub(
    env: &Env,
    x: &[f64],
    alpha: f64,
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    for c in comb_vectors {
        assert_eq!(x.len(), c.len());
    }

    let n_sets = comb_vectors.len();

    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("LP [Carr & Vempala, 02]", env).unwrap();

    let lambda = (0..n_sets).map(|c| {
            let name = format!("lambda[{c}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
        })
        .collect::<Vec<_>>();
    model.update().unwrap();


    // Add the constraint `sum( lambda[i] ) == 1`.
    let lhs = lambda.iter().grb_sum();
    let name = "sum( lambda[c] ) == 1";
    model.add_constr(&name, c!(lhs <= 1_f64)).unwrap();
    model.update().unwrap();

    for (i, xi) in x.iter().enumerate() {
        let lhs = lambda.iter()
            .zip(comb_vectors)
            .map(|(&lmd, cmb)| lmd * cmb[i])
            .grb_sum();
        let rhs = alpha * *xi;
        let name = format!("Coordinate[{i}]");
        model.add_constr(&name, c!(lhs <= rhs)).unwrap();
    }
    model.update().unwrap();

    let objective = lambda.iter().grb_sum();
    model.set_objective(objective, Maximize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    assert_eq!(status, Status::Optimal, "Failed to solve the approx. LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let lambda = lambda.iter()
        .map(|lmd| model.get_obj_attr(attr::X, lmd).unwrap())
        .collect::<Vec<_>>();


    (objval, lambda)
}
