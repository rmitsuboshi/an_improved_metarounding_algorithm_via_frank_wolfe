//! A sophisticated version of Metarounding-by-Boosting algorithm.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct SoftV2<'a> {
    /// The constant `L` defined in the paper.
    /// Note that `L` is at most the dimension of `x`.
    l1_norm: f64,


    /// `m := max{ 1/x[i] | x[i] > 0 }`
    m: f64,


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

    model: Model,
    vars: Vec<Var>,
}


impl<'a> SoftV2<'a> {
    pub fn new(
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("").unwrap();
        init_env(&mut env);
        let (_n_items, n_sets) = oracle.shape();
        let l1_norm = n_sets as f64;

        let model = Model::with_env("SoftV2", &env)
            .expect("Failed to construct a new gurobi model");
        let vars = Vec::new();

        Self {
            m: 1.0,
            l1_norm,
            _n_items,
            n_sets,
            tolerance,
            oracle,
            env,
            model,
            vars,
        }
    }


    fn max_iter(&self, _x: &[f64]) -> usize {
        assert!(self.tolerance > 0.0);
        let max_val = self.oracle.max_entry();
        let n = self.n_sets as f64;
        let numer = 8.0 * (max_val * self.m * n).powi(2);
        let numer = numer * (self.m * n.powi(2)).ln();
        let denom = self.tolerance.powi(2);

        ((numer / denom) + 2.0).floor() as usize
    }


    /// Construct a new QP instance.
    fn build_qp(
        &mut self,
        model_name: &str,
        x: &[f64]
    ) -> grb::Result<()>
    {
        let mut model = Model::with_env(model_name, &self.env)?;
        assert_eq!(x.len(), self.n_sets);
        self.vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..self.m)
        })
        .collect::<grb::Result<Vec<_>>>()?;
        self.model = model;
        self.model.update()?;

        let lx = self.vars.iter()
            .zip(x)
            .map(|(&vi, &xi)| vi * xi)
            .grb_sum();
        let name = "sum( ell[i] * x[i] ) <= 1";
        self.model.add_constr(&name, c!(lx <= 1f64))?;

        let sum = self.vars.iter().grb_sum();
        let name = "sum( ell[..] ) <= L";
        self.model.add_constr(&name, c!(sum <= self.l1_norm))?;
        self.model.update()?;

        Ok(())
    }



    fn grb_objective(&self, prev: &[f64]) -> Expr {
        let n_sets = self.vars.len() as f64;
        self.vars.iter()
            .zip(prev)
            .filter_map(|(&v, &p)| {
                if p == 0.0 {
                    None
                } else {
                assert!(p > 0.0);
                    let log = (p * n_sets).ln();
                    let lin = (log - 1.0).clamp(-INFINITY, INFINITY) * v;
                    let quad = (0.5_f64 / p).clamp(0.0, INFINITY)
                        * (v * v);
                    Some(lin + quad)
                }
                // assert!(p > 0.0);
                // let log = (p * n_sets).ln();
                // let lin = (log - 1.0) * v;
                // let quad = (0.5_f64 / p) * (v * v);
                // lin + quad
            })
            .grb_sum()
    }


    fn solve_without_dual_reduction(
        &mut self,
        ghat: f64,
        x: &[f64],
        comb_vectors: &[Vec<f64>],
        prev: Vec<f64>,
    ) -> Result<Vec<f64>, ()>
    {
        self.env.set(param::DualReductions, 0).unwrap();
        // Construct a new model to refresh the constraints.
        self.build_qp("MBB (w/o reduction)", x)
            .expect("Failed to construct a model");


        // Add constraints `c * l >= g_hat` for all `c`
        // collected by current round.
        let rhs = ghat + self.tolerance;
        comb_vectors.iter()
            .enumerate()
            .for_each(|(k, c)| {
                let lhs = c.iter()
                    .zip(&self.vars)
                    .map(|(&ci, &li)| ci * li)
                    .grb_sum();
                let name = format!("C[{k}]");
                self.model.add_constr(&name, c!(lhs >= rhs)).unwrap();
            });
        self.model.update().unwrap();


        assert_eq!(self.n_sets, self.vars.len());
        let objective = self.grb_objective(&prev);
        self.model.set_objective(objective, Minimize).unwrap();
        self.model.update().unwrap();


        // --------------------------------------
        // Solve the problem and obtain the optimal solution.
        self.model.optimize().unwrap();

        self.env.set(param::DualReductions, 1).unwrap();

        let status = self.model.status().unwrap();
        match status {
            // Infeasible implies an ε-optimality.
            Status::Infeasible => { return Err(()); },
            Status::InfOrUnbd => {
                panic!("InfOrUnbd without DualReduction");
            },
            Status::Numeric => {
                return Ok(prev);
            },
            _ => {},
        }
        self.env.set(param::DualReductions, 1).unwrap();


        // Get the optimal solution 
        let next = self.vars.iter()
            .map(|v| self.model.get_obj_attr(attr::X, v).unwrap())
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
        self.build_qp("MbB", x)
            .expect("Failed to construct a model");


        // Add constraints `c * l >= g_hat` for all `c`
        // collected by current round.
        let rhs = ghat + self.tolerance;
        comb_vectors.iter()
            .enumerate()
            .for_each(|(k, c)| {
                let lhs = c.iter()
                    .zip(&self.vars)
                    .map(|(&ci, &li)| ci * li)
                    .grb_sum();
                let name = format!("C[{k}]");
                self.model.add_constr(&name, c!(lhs >= rhs)).unwrap();
            });
        self.model.update().unwrap();


        // const STABILIZER: f64 = 1e-6;
        let mut prev_objval = f64::MAX;
        loop {
            let objective = self.grb_objective(&prev);
            self.model.set_objective(objective, Minimize).unwrap();
            self.model.update().unwrap();


            // --------------------------------------
            // Solve the problem and obtain the optimal solution.
            self.model.optimize().unwrap();


            let status = self.model.status().unwrap();
            match status {
                // Infeasible implies an ε-optimality.
                Status::Infeasible => { return Err(()); },
                Status::InfOrUnbd => {
                    return self.solve_without_dual_reduction(
                        ghat, x, comb_vectors, prev
                    );
                },
                Status::Numeric => {
                    // println!("Break by status {status:?}");
                    break;
                    // return Err(());
                },
                _ => {},
            }
            self.env.set(param::DualReductions, 1).unwrap();
            let objval = self.model.get_attr(attr::ObjVal)
                .unwrap();


            // Get the optimal solution 
            let next = self.vars.iter()
                .map(|v| self.model.get_obj_attr(attr::X, v).unwrap())
                .collect::<Vec<_>>();
            let has_non_pos = next.iter().any(|&nxt| nxt <= 0.0);
            prev = next;
            if has_non_pos
                || (prev_objval - objval) * 10.0 < self.tolerance
            {
                break;
            }
            prev_objval = objval;
        }

        Ok(prev)
    }
}


impl<'a> Metarounding for SoftV2<'a> {
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        let x = x.as_ref();

        assert_eq!(x.len(), self.n_sets);
        self.m = x.iter().copied()
            .filter_map(|x| if x > 0.0 { Some(1.0/x) } else { None })
            .reduce(f64::max)
            .expect("The input vector `x` should have non-zero entry");
        self.l1_norm = self.m * self.n_sets as f64;
        // A vector of combinatorial vectors, 
        // collected by current iteration.
        let mut comb_vectors = Vec::new();


        // Initial estimation is the uniform distribution.
        let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];
        let mut ghat = f64::MIN;


        let max_iter = self.max_iter(x);



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

        // Solve LP to obtain the coefficient vector `lambda`.
        let (_, lambda) = solve_primal(&self.env, x, &comb_vectors)
            .unwrap();


        (lambda, comb_vectors)
    }
}



