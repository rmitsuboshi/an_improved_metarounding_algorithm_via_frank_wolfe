//! Metarounding-by-Boosting algorithm, proposed by Fujita et al., '13.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct Soft<'a> {
    /// The constant `L` defined in the paper.
    /// Note that `L` is at most the dimension of `x`.
    l1_norm: f64,

    /// An upper-bound of infiniy norm of `x`.
    linf_norm: f64,


    /// The number of items.
    _n_items: usize,


    /// The number of sets.
    n_sets: usize,


    /// Tolerance parameter.
    tolerance: f64,


    /// The approximation algorithm that returns a vector in `C`.
    oracle: &'a Oracle,


    /// Gurobi Env
    env: Env,
}


impl<'a> Soft<'a> {
    pub fn new(
        linf_norm: f64,
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        env.set(param::NumericFocus, 3).unwrap();
        // env.set(param::DualReductions, 0).unwrap();
        let (_n_items, n_sets) = oracle.shape();
        let l1_norm = n_sets as f64;

        Self {
            l1_norm,
            linf_norm,
            _n_items,
            n_sets,
            tolerance,
            oracle,
            env,
        }
    }


    fn max_iter(&self) -> usize {
        assert!(self.tolerance > 0.0);
        let l = self.l1_norm;
        let numer = 8.0 * l.powi(2) * self.linf_norm.powi(2);
        let numer = numer * (l * self.n_sets as f64).ln();
        let denom = self.tolerance.powi(2);

        ((numer / denom) + 2.0).floor() as usize
    }


    fn solve_without_dual_reduction(
        &mut self,
        ghat: f64,
        x: &[f64],
        comb_vectors: &[Vec<f64>],
        prev: &[f64],
    ) -> Result<Vec<f64>, ()>
    {
        self.env.set(param::DualReductions, 0).unwrap();
        // Construct a new model to refresh the constraints.
        let mut model = Model::with_env("MBB (w/o reduction)", &self.env)
            .unwrap();

        // Defines variable `ell`
        let vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..self.l1_norm).unwrap()
        })
        .collect::<Vec<_>>();
        model.update().unwrap();


        // Add the constraint `l * x <= 1`.
        let lx = vars.iter()
            .zip(x)
            .map(|(&li, &xi)| xi * li)
            .grb_sum();
        let name = "ell * x <= 1";
        model.add_constr(&name, c!(lx <= 1_f64)).unwrap();
        model.update().unwrap();


        // Add the constraint `l * 1 <= L`
        let lhs = vars.iter().grb_sum();
        let name = "ell * 1 <= L";
        model.add_constr(&name, c!(lhs <= self.l1_norm)).unwrap();
        model.update().unwrap();


        // Add constraints `c * l >= ghat` for all `c`
        // collected until current round.
        let rhs = ghat + self.tolerance;
        comb_vectors.iter()
            .enumerate()
            .for_each(|(k, c)| {
                let lhs = c.iter()
                    .zip(&vars)
                    .map(|(&ci, &li)| ci * li)
                    .grb_sum();
                let name = format!("C[{k}]");
                model.add_constr(&name, c!(lhs >= rhs)).unwrap();
            });
        model.update().unwrap();


        let n_sets = self.n_sets as f64;
        let objective = vars.iter()
            .zip(prev)
            .map(|(&v, &p)| {
                assert!(p > 0.0);
                let log = (p * n_sets).ln();
                let lin = (log - 1.0) * v;
                let quad = (0.5_f64 / p) * (v * v);
                lin + quad
            })
            .grb_sum();
        model.set_objective(objective, Minimize).unwrap();
        model.update().unwrap();


        // --------------------------------------
        // Solve the problem and obtain the optimal solution.
        model.optimize().unwrap();

        self.env.set(param::DualReductions, 1).unwrap();

        let status = model.status().unwrap();
        match status {
            // Infeasible implies an ε-optimality.
            Status::Infeasible => { return Err(()); },
            Status::InfOrUnbd => {
                panic!("InfOrUnbd without DualReduction");
            },
            Status::Numeric => {
                println!("Break by status {status:?}");
            },
            _ => {},
        }


        // Get the optimal solution 
        let next = vars.iter()
            .map(|v| model.get_obj_attr(attr::X, v).unwrap())
            .collect::<Vec<_>>();

        Ok(next)
    }


    fn solve_sequential_qp(
        &mut self,
        ghat: f64,
        x: &[f64],
        comb_vectors: &[Vec<f64>],
    ) -> Result<Vec<f64>, ()>
    {
        let mut prev = vec![1.0 / self.n_sets as f64; self.n_sets];
        // Construct a new model to refresh the constraints.
        let mut model = Model::with_env("MBB", &self.env)
            .unwrap();

        // Defines variable `ell`
        let vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..self.l1_norm).unwrap()
        })
        .collect::<Vec<_>>();
        model.update().unwrap();


        // Add the constraint `l * x <= 1`.
        let lx = vars.iter()
            .zip(x)
            .map(|(&li, &xi)| xi * li)
            .grb_sum();
        let name = "ell * x <= 1";
        model.add_constr(&name, c!(lx <= 1_f64)).unwrap();
        model.update().unwrap();


        // Add the constraint `l * 1 <= L`
        let lhs = vars.iter().grb_sum();
        let name = "ell * 1 <= L";
        model.add_constr(&name, c!(lhs <= self.l1_norm)).unwrap();
        model.update().unwrap();


        // Add constraints `c * l >= g_hat` for all `c`
        // collected by current round.
        let rhs = ghat + self.tolerance;
        comb_vectors.iter()
            .enumerate()
            .for_each(|(k, c)| {
                let lhs = c.iter()
                    .zip(&vars)
                    .map(|(&ci, &li)| ci * li)
                    .grb_sum();
                let name = format!("C[{k}]");
                model.add_constr(&name, c!(lhs >= rhs)).unwrap();
            });
        model.update().unwrap();


        let n_sets = self.n_sets as f64;
        loop {
            // Set the objective function.
            let objective = vars.iter()
                .zip(&prev)
                .map(|(&v, &p)| {
                    assert!(p > 0.0);
                    let log = (p * n_sets).ln();
                    let lin = (log - 1.0) * v;
                    let quad = (0.5_f64 / p).min(INFINITY) * (v * v);
                    lin + quad
                })
                .grb_sum();
            model.set_objective(objective, Minimize).unwrap();
            model.update().unwrap();


            // --------------------------------------
            // Solve the problem and obtain the optimal solution.
            model.optimize().unwrap();


            let status = model.status().unwrap();
            match status {
                // Infeasible implies an ε-optimality.
                Status::Infeasible => { return Err(()); },
                Status::InfOrUnbd => {
                    return self.solve_without_dual_reduction(
                        ghat, x, comb_vectors, &prev
                    );
                },
                Status::Numeric => {
                    println!("Break by status {status:?}");
                    break;
                },
                _ => {},
            }


            // Get the optimal solution 
            let next = vars.iter()
                .map(|v| model.get_obj_attr(attr::X, v).unwrap())
                .collect::<Vec<_>>();
            // Calculate the distance
            // from the previous solution to the current one.
            let dist = distance(&prev, &next);

            // Update the iterate
            prev = next;

            // If there exists a zero-valued entry,
            // the optimization cannot proceed for the next iteration.
            // The following line checks whether there exists 
            // any non-positive-valued entry.
            let has_zero = prev.iter().any(|&p| p == 0.0);
            if has_zero { return Err(()); }
            if dist * 10.0 < self.tolerance { break; }
            // if has_zero || dist * 10.0 < self.tolerance { break; }
        }

        Ok(prev)
    }
}


impl<'a> Metarounding for Soft<'a> {
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        let x = x.as_ref();

        assert_eq!(x.len(), self.n_sets);
        self.l1_norm = 1_f64;
        // A vector of combinatorial vectors, 
        // collected by current iteration.
        let mut comb_vectors;
        loop {
            let max_iter = self.max_iter();
            comb_vectors = Vec::new();

            // Initial estimation is the uniform distribution.
            let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];
            let mut ghat = f64::MIN;


            for _ in 1..=max_iter {
                // Call approximation algorithm and obtain
                // a new combinatorial concept `c`.
                let c = self.oracle.call(&ell);

                assert_eq!(c.len(), ell.len());

                let cl = c.iter()
                    .zip(&ell)
                    .map(|(ci, li)| ci * li)
                    .sum::<f64>();
                ghat = ghat.max(cl);

                comb_vectors.push(c);


                // Solve the relative entropy minimization problem
                // by solving the quadratic approximation problems.
                match self.solve_sequential_qp(ghat, x, &comb_vectors) {
                    Ok(res) => { ell = res; },
                    Err(_) => { break; },
                }
            }

            // Obtain a dual lp solution.
            let (_, ell) = solve_dual(&self.env, x, &comb_vectors);
            let s = ell.iter().sum::<f64>();
            if s < self.l1_norm {
                break;
            } else {
                self.l1_norm *= 2.0;
            }
        }

        // Solve LP to obtain the coefficient vector `lambda`.
        let (_, lambda) = solve_primal(&self.env, x, &comb_vectors);


        (lambda, comb_vectors)
    }
}



