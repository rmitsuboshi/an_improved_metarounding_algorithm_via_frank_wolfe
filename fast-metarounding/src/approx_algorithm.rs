//! This file defines an approximation algorithm, named `Oracle`.
//! Given a vector `Oracle`
//! 
//! 
use grb::prelude::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};


use std::collections::HashSet;


/// A set-cover oracle.
/// This struct stores the set cover instances in `sets`.
/// Each column represents a set that covers some rows.
/// The goal of set-cover is to find a small subset of columns
/// that covers all rows.
/// 
/// ```txt
///                       SETS
///                         j                    
///     ┌───────────────────▄───────────────────┐
///     │                                       │
///  I  │                                       │
///  T  │                    (i,j)              │
///  E i▐                   █                   ▌
///  M  │                                       │
///  S  │                                       │
///     │                                       │
///     └───────────────────▀───────────────────┘
/// ```
/// 
/// - Each entry of `sets` is `0` or `1`.
/// - The `j`th column of `sets` shows a set.
/// - `sets[i][j] = 1` w.p. `p` for some given `p ∈ (0, 1)`.
/// 
/// 
/// 
pub struct Oracle {
    n_items: usize,
    n_sets: usize,
    alpha: f64,
    sets: Vec<Vec<f64>>,
    env: Env,
}


impl Oracle {
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_items, self.n_sets)
    }


    #[inline(always)]
    pub fn alpha_bound(&self) -> f64 {
        self.alpha
    }


    #[inline(always)]
    pub fn best_action_hindsight<T>(&self, cumulative_loss: T)
        -> Vec<f64>
        where T: AsRef<[f64]>
    {
        let loss = cumulative_loss.as_ref();

        let name = "Approximation Oracle";
        let mut model = Model::with_env(name, &self.env).unwrap();

        // Define variables
        let vars = (0..self.n_sets).map(|k| {
                let name = format!("c[{k}]");
                add_binvar!(model, name: &name).unwrap()
            })
            .collect::<Vec<_>>();
        model.update().unwrap();


        // Add constraints
        for (k, row) in self.sets.iter().enumerate() {
            assert_eq!(row.len(), vars.len());
            let lhs = row.iter()
                .zip(&vars)
                .map(|(&r, &v)| r * v)
                .grb_sum();
            let name = format!("Item {k}");
            model.add_constr(&name, c!(lhs >= 1_f64)).unwrap();
        }
        model.update().unwrap();

        // !DEBUG
        assert_eq!(loss.len(), vars.len());


        // Set the objective function
        let objective = loss.iter()
            .zip(&vars)
            .map(|(&l, &v)| l * v)
            .grb_sum();
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();

        model.optimize().unwrap();

        // --------------------------------------------------------------
        // At this point, we have an optimal vector x 
        // of the following problem:
        // 
        // ```txt
        //      Minimize sum( l[i] * x[i] )
        //      sub. to. sum( c[i] * x[i] ) ≥ 1, ∀c in `self.sets`
        //               x[i] ∈ {0, 1}           ∀i = 1, 2, ..., n
        // ```
        // --------------------------------------------------------------


        let comb = vars.iter()
            .map(|v| model.get_obj_attr(attr::X, v).unwrap())
            .collect::<Vec<_>>();

        assert!(self.is_valid_setcover(&comb));
        comb
    }


    #[inline(always)]
    pub fn generate_setcover(
        // `n_items == m` in the code by Fujita+
        n_items: usize,
        // `n_sets == n` in the code by Fujita+
        n_sets: usize,
        // The number of relevant sets
        n_relevants: f64,
        // Probability
        p: f64,
        // Random seed
        seed: u64,
    ) -> Self
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let distribution = Uniform::from(0.0..1.0);

        let mut sets = vec![vec![0.0; n_sets]; n_items];
        assert_eq!(sets.len(), n_items);

        // For each item (row) `i`,
        // there exists a set (column) `j` that contains `i`;
        // i.e., `∀i, ∃j, sets[i][j] == 1`.
        for i in 0..n_items {
            // Since `r ∈ [0, 1)` and `n_relevants ∈ [0, n_sets-1]`,
            // `r * n_relevants ∈ [0, n_sets-1)`.
            let r = distribution.sample(&mut rng);
            let j = (r * n_relevants).floor() as usize;
            sets[i][j] = 1.0;
        }


        // `sets[i][j] == 1` with probability `p`.
        for i in 0..n_items {
            for j in 0..n_sets {
                // Draw a random value from U[0, 1]
                let r = distribution.sample(&mut rng);
                if r < p { sets[i][j] = 1.0; }
            }
        }


        // Each row has at least 1 entry with value `1.0`.
        assert!(
            sets.iter().all(|item| item.iter().any(|i| *i == 1_f64))
        );


        // Get the approximation parameter `alpha`.
        // Here, `alpha` is the maximum number of sets 
        // covering an item (# of non-zero entries in each row)
        // `alpha` equals zero if there are no items.
        // 
        // ```txt
        // α ∈ arg max sum(sets[i][0] + sets[i][1] + ... + sets[i][n-1])
        //     subject to i = 0, 1, ..., m-1
        // ```
        // 
        // See p.19 of the following book:
        // https://www.designofapproxalgs.com/book.pdf
        let alpha = sets.iter()
            .map(|row| row.iter().sum::<f64>())
            .reduce(f64::max)
            .unwrap_or(0.0);

        println!("Approx. rate α = {}", alpha);

        assert!(
            alpha > 0.0,
            "\
            Failed to construct a set-cover instance. \
            The maximal nnz was zero!\
            "
        );

        // Construct a Gurobi environment beforehand.
        // Constructing a new environment outputs some text
        // on console so that we only construct it once.
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        Self {
            n_items,
            n_sets,
            alpha,
            sets,
            env,
        }
    }


    /// Returns a initial prediction vector `x` in the relaxed space.
    /// The returned vector `x` is constructed in the following way:
    /// 
    /// 1. Repeat the following procedure `k` times:
    ///     a. Construct a random loss vector.
    ///        Each entry is drawn i.i.d. from the uniform distribution.
    ///     b. Call the approximation oracle to the loss vector
    ///        to obtain a combinatorial vector `c`
    /// 2. Set `x` as the average of `c`s.
    /// ```txt
    /// x = sum( c[1] + c[2] + ... + c[k] ) / k ∈ R^n
    /// where n is the number of sets,
    ///       c[j] ∈ R^n for all j = 1, 2, ..., k.
    /// ```
    #[inline(always)]
    pub fn initial_point(
        &self,
        seed: u64,
    ) -> Vec<f64>
    {
        // let loss = vec![1.0 / self.n_sets as f64; self.n_sets];
        // self.call(loss)
        let mut rng = StdRng::seed_from_u64(seed);
        let distribution = Uniform::from(0.0..1.0);
        let loss = distribution.sample_iter(&mut rng)
            .take(self.n_sets)
            .collect::<Vec<_>>();

        let name = "Initial point generator";
        let mut model = Model::with_env(name, &self.env).unwrap();

        // Define variables
        let vars = (0..self.n_sets).map(|k| {
                let name = format!("x[{k}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        model.update().unwrap();


        // Add constraints
        for (k, row) in self.sets.iter().enumerate() {
            assert_eq!(row.len(), vars.len());
            let lhs = row.iter()
                .zip(&vars)
                .map(|(&r, &v)| r * v)
                .grb_sum();
            let name = format!("Item {k}");
            model.add_constr(&name, c!(lhs >= 1_f64)).unwrap();
        }
        model.update().unwrap();

        // !DEBUG
        assert_eq!(loss.len(), vars.len());


        // Set the objective function
        let objective = loss.iter()
            .zip(&vars)
            .map(|(&l, &v)| l * v)
            .grb_sum();
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();

        model.optimize().unwrap();

        // --------------------------------------------------------------
        // At this point, we have an optimal vector x 
        // of the following problem:
        // 
        // ```txt
        //      Minimize sum( l[i] * x[i] )
        //      sub. to. sum( c[i] * x[i] ) ≥ 1, ∀c in `self.sets`
        //               x ∈ [0, 1]ⁿ
        // ```
        // --------------------------------------------------------------


        vars.iter()
            .map(|v| model.get_obj_attr(attr::X, v).unwrap())
            .collect::<Vec<_>>()
    }


    /// Given a loss vector `loss`,
    /// this method returns a combinatorial vector `c ∈ C` satisfying
    /// 
    /// ```txt
    /// sum( c[i] * ell[i] ) <= α * min { sum( x[i] * ell[i] ) | x ∈ P }
    /// ```
    /// 
    /// This code is based on the one in Fujita et al.,
    /// See 
    /// 
    /// - `matlab/metarounding/approx_oracle.m`
    /// - `matlab/metarounding/lp_setcover2.m`
    #[inline(always)]
    pub fn call<T: AsRef<[f64]>>(&self, loss: T) -> Vec<f64> {
        let loss = loss.as_ref();

        // DEBUG
        assert_eq!(loss.len(), self.n_sets);


        let name = "Approx. Oracle";
        let mut model = Model::with_env(name, &self.env).unwrap();

        // Define variables
        let vars = (0..self.n_sets).map(|k| {
                let name = format!("c[{k}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        model.update().unwrap();


        // Add constraints
        for (k, row) in self.sets.iter().enumerate() {
            assert_eq!(row.len(), vars.len());
            let lhs = row.iter()
                .zip(&vars)
                .map(|(&r, &v)| r * v)
                .grb_sum();
            let name = format!("Item {k}");
            model.add_constr(&name, c!(lhs >= 1_f64)).unwrap();
        }
        model.update().unwrap();

        // !DEBUG
        assert_eq!(loss.len(), vars.len());


        // Set the objective function
        let objective = loss.iter()
            .zip(&vars)
            .map(|(&l, &v)| l * v)
            .grb_sum();
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();

        model.optimize().unwrap();

        // --------------------------------------------------------------------
        // At this point, we have an optimal vector x of the following problem:
        // 
        // ```txt
        //      Minimize sum( l[i] * x[i] )
        //      sub. to. sum( c[i] * x[i] ) ≥ 1, ∀c in `self.sets`
        //               x[i] ∈ [0, 1]           ∀i = 1, 2, ..., n
        // ```
        // 
        // From now on, we will round `x` to the vector 
        // in the discrete space `{0, 1}^n`.
        // --------------------------------------------------------------------


        let comb = vars.iter()
            .map(|vi| {
                let pi = model.get_obj_attr(attr::X, vi).unwrap();
                // Round the solution of LP-relaxed problem
                if pi * self.alpha > 1.0 { 1.0 } else { 0.0 }
            })
            .collect::<Vec<_>>();

        assert!(self.is_valid_setcover(&comb));
        comb
    }


    /// The separation oracle for the following optimization problem:
    /// ```txt
    ///     Minimize γ + α sum( l[i] * x[i] )
    ///     sub. to. sum( l[i] * c[i] ) + γ ≥ 1, for all c ∈ C,      --- (*)
    ///                                l[i] ≥ 0, for all i ∈ [n],
    ///                                   γ ≥ 0.
    /// ```
    /// Given a point `ell_gamma = [l, γ]`, this oracle returns
    /// 1. `Ok(())` if there is no constraint in (*) violated by `ell_gamma`,
    /// 2. `Err([-c, -1])` if there is a combinatorial vector `c`
    ///     violated by `ell_gamma`.
    #[inline(always)]
    pub fn separation_oracle(
        &self,
        alpha_ub: f64,
        tolerance: f64,
        ell_gamma: &[f64],
    ) -> Result<(), Vec<f64>>
    {
        let ell = &ell_gamma[..self.n_sets];
        let gamma = ell_gamma[self.n_sets];
        assert_eq!(self.n_sets + 1, ell_gamma.len());

        let c = self.call(ell);

        // Computes `l * c`.
        let l_dot_c = ell.iter()
            .zip(&c)
            .map(|(li, ci)| li * ci)
            .sum::<f64>();

        if alpha_ub * l_dot_c + gamma < 1.0 - tolerance {
            let mut sep = c.iter().map(|ci| -ci).collect::<Vec<_>>();
            sep.push(-1.0);
            return Err(sep);
        }
        Ok(())
    }


    /// Check whether the given vector is an instance of set covers.
    #[inline(always)]
    fn is_valid_setcover<T: AsRef<[f64]>>(&self, comb: T)
        -> bool
    {
        let comb = comb.as_ref();
        assert_eq!(comb.len(), self.n_sets);

        let mut items = HashSet::new();
        let iter = comb.iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0);
        for (j, _) in iter {
            for item in 0..self.n_items {
                if self.sets[item][j] == 0.0 { continue; }
                items.insert(item);
            }
        }
        items.len() == self.n_items
    }


    pub fn l2_projection<T: AsRef<[f64]>>(&self, y: T)
        -> Vec<f64>
    {
        let y = y.as_ref();

        let n = y.len();

        // Solve LP to obtain the coefficient vector `lambda`.
        let name = "L2 projection";
        let mut model = Model::with_env(name, &self.env)
            .unwrap();


        let vars = (0..n).map(|i| {
                let name = format!("x[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        model.update().unwrap();


        // Add the constraint `sum( ell[i] * x[i] ) == 1`.
        for (i, comb) in self.sets.iter().enumerate() {
            let lhs = comb.iter()
                .zip(&vars)
                .map(|(&ci, &vi)| ci * vi)
                .grb_sum();
            let name = format!("C[{i}]");
            model.add_constr(&name, c!(lhs >= 1.0)).unwrap();
        }
        model.update().unwrap();

        let objective = vars.iter()
            .zip(y)
            .map(|(&vi, &yi)| (0.5 * (vi * vi)) - (yi * vi))
            .grb_sum();

        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();


        model.optimize().unwrap();


        let status = model.status().unwrap();
        assert_eq!(status, Status::Optimal, "Projection failed");

        vars.iter()
            .map(|l| model.get_obj_attr(attr::X, l).unwrap())
            .collect::<Vec<_>>()
    }
}


