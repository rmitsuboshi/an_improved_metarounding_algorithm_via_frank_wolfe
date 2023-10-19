//! A Metarounding algorithm based on ERLPBoost.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct Erlp<'a> {
    /// An upper-bound of infiniy norm of `x`.
    linf_norm: f64,


    /// Regularization parameter
    eta: f64,

    /// The number of items.
    _n_items: usize,


    /// The number of sets.
    n_sets: usize,


    /// Tolerance parameter.
    tolerance: f64,


    /// 0.5 * tolerance
    half_tolerance: f64,


    /// The approximation algorithm that returns a vector in `C`.
    oracle: &'a Oracle,


    /// Combinatorial vectors
    comb_vectors: Vec<Vec<f64>>,


    /// Gurobi Env
    env: Env,
}


impl<'a> Erlp<'a> {
    pub fn new(
        linf_norm: f64,
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("ERLP")
            .expect("Failed to construct a new environment `ERLP`");
        env.set(param::OutputFlag, 0)
            .expect("Failed to set `OutputFlag` parameter to `0`");
        env.set(param::NumericFocus, 3)
            .expect("Failed to set `NumericFocus` parameter to `3`");
        // env.set(param::DualReductions, 0).unwrap();

        let (_n_items, n_sets) = oracle.shape();
        let eta = 2.0 * (n_sets as f64).ln() / tolerance;
        let half_tolerance = tolerance * 0.5;

        let comb_vectors = Vec::new();

        Self {
            linf_norm,
            eta,
            _n_items,
            n_sets,
            tolerance,
            half_tolerance,
            oracle,
            comb_vectors,
            env,
        }
    }
}

impl<'a> Erlp<'a> {
    fn max_iter(&self) -> usize {
        assert!(self.tolerance > 0.0);
        let numer = 16.0 * self.linf_norm.powi(2) * self.eta;
        let denom = self.tolerance;

        ((numer / denom) - 2.0).ceil() as usize
    }


    fn solve_sequential_qp(
        &self,
        model: &mut Model,
        vars: &[Var],
        gamma: &Var,
        x: &[f64],
        c: &[f64],
    ) -> (f64, Vec<f64>)
    {
        assert_eq!(c.len(), vars.len());
        // Add the new constraint
        // ```txt
        // sum( c[i] * l[i] ) ≥ γ
        // ```
        // For the new combinatorial vector `c` given as the input.
        let lhs = c.iter().zip(vars).map(|(&ci, &vi)| ci * vi).grb_sum();
        let k = self.comb_vectors.len();
        let name = format!("C[{k}]");
        model.add_constr(&name, c!(lhs >= gamma))
            .expect(
                "Failed to add a new constraint `sum( c[i] * l[i] ) >= gamma`"
            );
        model.update()
            .expect("Failed to update the model when adding a new constraint");


        let n_sets = self.n_sets as f64;
        // Current estimation of the optimal solution `l`.
        let mut prev = vec![1.0 / n_sets; self.n_sets];

        // !DEBUG
        assert_eq!(prev.len(), vars.len());
        assert_eq!(prev.len(), x.len());

        // ---------------------------------------------------
        // Since the entropic objective cannot be solved by Gurobi,
        // we solve the quadratic approximation problem repeatedly.
        // ---------------------------------------------------
        // Instead of solving
        // 
        // 
        // ```txt
        // Maximize γ - (1/η) sum( x[i]*l[i] ln( x[i]*l[i] / (1/n) ) )
        // sub. to. sum( c[j][i] * x[i] ) ≥ γ, ∀j = 1, 2, ..., t,
        //          sum( x[i] * l[i] ) = 1,
        //          l[i] ≥ 0,                  ∀i = 1, 2, ..., n,
        // ```
        // 
        // We solve the following quadratic problems
        // by updating `p` (Initially, `p[i] = 1/n` for all `i`).
        // 
        // 
        // ```txt
        // Maximize γ
        //         - (1.0/η)*sum( {x[i]*ln( x[i]*p[i] / (1/n) )} * l[i] )
        //         - (0.5/η)*sum( {x[i] / p[i]} * l[i] )
        // sub. to. sum( c[j][i] * x[i] ) ≥ γ, ∀j = 1, 2, ..., t,
        //          sum( x[i] * l[i] ) = 1,
        //          l[i] ≥ 0,                  ∀i = 1, 2, ..., n,
        // ```
        // 
        // This alternative problem differs 
        // from the original one by constant,
        // but it is not the problem since we are interested in
        // the optimal solution.
        // 
        // Then,
        // update `p` as the optimal solution of the alternative problem.
        // ---------------------------------------------------

        let mut g = 0.0;
        let mut prev_objval = f64::MIN / 2.0;
        loop {
            // Set the objective function.
            let regularizer = prev.iter()
                .zip(x)
                .map(|(pi, xi)| {
                    let zi = n_sets * pi * xi;
                    let lg = if zi <= 0.0 { 0.0 } else { zi.ln() };
                    let qd = 0.5 * (xi / pi).min(INFINITY);
                    (xi * lg / self.eta, qd / self.eta)
                })
                .zip(vars)
                .map(|((li, qi), &vi)| (li * vi) + (qi * (vi * vi)))
                .grb_sum();
            let objective = *gamma - regularizer;
            model.set_objective(objective, Maximize)
                .expect("Failed to set the objective function");
            model.update()
                .expect(
                    "Failed to update the model after setting the objective"
                );


            // --------------------------------------
            // Solve the problem and obtain the optimal solution.
            model.optimize()
                .expect("Failed to optimize the problem");


            // If `status` is neither `Optimal` nor `SubOptimal`,
            // (e.g., `Numeric`)
            // the optimization may fail for the rest rounds.
            // Thus, we break the loop for such a case.
            let status = model.status()
                .expect("Failed to get the model status");
            if status != Status::Optimal && status != Status::SubOptimal {
                // println!("status is: {status:?}");
                break;
            }


            // At this point, you can get (Sub)Optimal solution.
            g = model.get_obj_attr(attr::X, gamma)
                .expect("Failed to get optimal solution `gamma`");


            // Get the current solution and calculate L2-distance
            // from the previous solution to the current one.
            let next = vars.iter()
                .map(|v| model.get_obj_attr(attr::X, v))
                .collect::<Result<Vec<_>, _>>()
                .expect("Failed to collect the optimal solution `ell[..]`");


            let objval = g - self.entropy(&next, x) / self.eta;

            // let has_zero = prev.iter().any(|&l| l <= 1e-200);
            let has_zero = prev.iter().zip(x).any(|(&l, &xi)| l <= 0.0 && xi > 0.0);
            if has_zero || (objval - prev_objval) * 10.0 < self.half_tolerance {
                break;
            }
            prev_objval = objval;
            prev = next;

            // let dist = distance(&prev, &next);
            // let has_zero = prev.iter().zip(x).any(|(&l, &xi)| l <= 0.0 && xi > 0.0);
            // println!("gap is: {}", dist * 10.0);
            // let small_gap = dist * 10.0 < self.half_tolerance;
            // if has_zero || small_gap { break; }
            // prev = next;
        }

        // ---------------------------------------------------
        // At this point, an approximate solution is found.
        // Get the objective value `objval`.
        let objval = g - (self.entropy(&prev, x) / self.eta);

        (objval, prev)
    }


    fn entropy<S, T>(&self, loss: S, x: T) -> f64
        where S: AsRef<[f64]>,
              T: AsRef<[f64]>
    {
        let loss = loss.as_ref();
        let x = x.as_ref();
        assert_eq!(loss.len(), x.len());
        let n = loss.len() as f64;

        loss.into_iter()
            .zip(x)
            .map(|(li, xi)| {
                let zi = li * xi;
                if zi <= 0.0 { 0.0 } else { (zi * n).ln() * zi }
            })
            .sum::<f64>()
    }


    /// Construct a new QP instance.
    fn build_qp(&self, x: &[f64]) -> (Model, Vec<Var>, Var) {
        // Construct a new model to refresh the constraints.
        let mut model = Model::with_env("Erlp", &self.env)
            .expect("Failed to construct a new gurobi model");

        // Defines variable `ell`
        let vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            let ub = if x[i] == 0.0 { 1.0 } else { 1.0 / x[i] };
            add_ctsvar!(model, name: &name, bounds: 0.0..ub)
            // add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to construct Gurobi variables `ell[..]`");
        model.update()
            .expect("Failed to update the model after setting `ell[..]`");

        // Defines variable `gamma`.
        let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
            .expect("Failed to construct Gurobi variable `gamma`");


        // Add the constraint `l * x == 1`.
        let lx = vars.iter()
            .zip(x)
            .map(|(&vi, &xi)| vi * xi)
            .grb_sum();
        let name = "sum( ell[i] * x[i] ) <= 1";
        model.add_constr(&name, c!(lx <= 1_f64))
            .expect("Failed to set the constraint `sum( ell[i] * x[i] ) == 1`");
        model.update()
            .expect(
                "\
                Failed to update the model after adding the constraint \
                `sum( ell[i] * x[i] ) == 1`\
                "
            );


        (model, vars, gamma)
    }
}


impl<'a> Metarounding for Erlp<'a> {
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        let x = x.as_ref();
        // let n = x.len() as f64;
        // let x = x.as_ref()
        //     .into_iter()
        //     .map(|&xi| if n >= INFINITY * xi { 0.0 } else { xi })
        //     .collect::<Vec<_>>();
        // let x = &x[..];


        // Initial estimation is the uniform distribution.
        let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];

        // `ghat` is the current estimation of 
        // approximation param., `α`.
        let mut gstar = f64::MAX;
        let mut ghat = f64::MIN;


        let max_iter = self.max_iter();

        let (mut model, vars, gamma) = self.build_qp(x);


        // A vector of combinatorial vectors, 
        // collected by current iteration.
        self.comb_vectors = Vec::new();

        for _ in 1..=max_iter {
            // println!("[ROUND {round}]");
            // Call approximation algorithm and obtain
            // a new combinatorial concept `c`.
            let c = self.oracle.call(&ell);

            assert_eq!(c.len(), ell.len());

            ghat = c.iter()
                .zip(&ell)
                .map(|(ci, li)| ci * li)
                .sum::<f64>()
                .max(ghat);
            self.comb_vectors.push(c.clone());


            // Optimality gap measures the difference between
            // the current objective value `H*(Cλ_{k})` and
            // the estimate `max { sum(c[j+1][i] * l[i] | j ≤ k }`
            // 
            // The variable `optimality_gap` corresponds to `ε_{k}`
            // in our paper.
            let optimality_gap = gstar - ghat;
            if optimality_gap <= self.half_tolerance { break; }


            // Solve the ERLPBoost-like problem
            // by repeatedly solve the quadratic approximation problems.
            (gstar, ell) = self.solve_sequential_qp(
                &mut model, &vars, &gamma, &x, &c
            );
        }


        // Solve LP to obtain the coefficient vector `lambda`.


        // DEBUG
        // println!("# of combs: {}", self.comb_vectors.len());
        // self.env.set(param::DualReductions, 0).unwrap();
        let (_, lambda) = solve_primal_bounded(
            &self.env, x, &self.comb_vectors
        );
        // self.env.set(param::DualReductions, 1).unwrap();


        let comb_vectors = std::mem::take(&mut self.comb_vectors);
        (lambda, comb_vectors)
    }
}



