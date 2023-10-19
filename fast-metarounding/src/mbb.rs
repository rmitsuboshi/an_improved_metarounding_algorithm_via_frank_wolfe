//! Metarounding-by-Boosting algorithm, proposed by Fujita et al., '13.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct MbB {
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
    oracle: Oracle,


    /// Gurobi Env
    env: Env,
}


impl MbB {
    pub fn new(
        linf_norm: f64,
        tolerance: f64,
        oracle: Oracle,
    ) -> Self
    {
        let l1_norm = 1.0;
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        // env.set(param::DualReductions, 0).unwrap();
        let (_n_items, n_sets) = oracle.shape();

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


    fn solve_sequential_qp(
        &self,
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
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
                .unwrap()
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
                let name = format!("C[{k: >5}]");
                model.add_constr(&name, c!(lhs >= rhs)).unwrap();
            });
        model.update().unwrap();


        let n_sets = self.n_sets as f64;
        loop {
            // Set the objective function.
            let objective = vars.iter()
                .zip(&prev)
                .map(|(&v, &l)| {
                    assert!(l > 0.0);
                    let log = (l * n_sets).ln();
                    let lin = (log - 1.0) * v;
                    let quad = (0.5_f64 / l) * (v * v);
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
                Status::Infeasible | Status::InfOrUnbd => {
                    return Err(());
                },

                // Sub-optimal solution is attained
                Status::Numeric => { break; },
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
            let any_zero = prev.iter().any(|&p| p <= 0.0);


            if any_zero || dist * 10.0 < self.tolerance { break; }
        }

        Ok(prev)
    }
}


impl Metarounding for MbB {
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
            comb_vectors = Vec::new();


            // Initial estimation is the uniform distribution.
            let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];
            let mut ghat = f64::MIN;


            let max_iter = self.max_iter();



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
            if s < self.l1_norm - self.tolerance {
                break;
            } else {
                self.l1_norm *= 2.0;
            }
        }
        // println!("Final L: {}", self.l1_norm);

        // Solve LP to obtain the coefficient vector `lambda`.
        let (_, lambda) = solve_primal(&self.env, x, &comb_vectors);


        (lambda, comb_vectors)
    }
}



