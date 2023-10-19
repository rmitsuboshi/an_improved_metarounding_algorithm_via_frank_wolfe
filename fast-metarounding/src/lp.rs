//! A Metarounding algorithm based on LPBoost.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct Lp<'a> {
    /// The number of items.
    _n_items: usize,


    /// The number of sets.
    n_sets: usize,


    /// Tolerance parameter.
    tolerance: f64,


    /// The approximation algorithm that returns a vector in `C`.
    oracle: &'a Oracle,


    comb_vectors: Vec<Vec<f64>>,


    /// Gurobi Env
    env: Env,
}


impl<'a> Lp<'a> {
    pub fn new(
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        // env.set(param::DualReductions, 0).unwrap();

        let (_n_items, n_sets) = oracle.shape();
        let comb_vectors = Vec::new();

        Self {
            _n_items,
            n_sets,
            tolerance,
            oracle,
            comb_vectors,
            env,
        }
    }


    fn build_lp(&self, x: &[f64]) -> (Model, Vec<Var>, Var) {
        // Construct a new model to refresh the constraints.
        let mut model = Model::with_env("Lp", &self.env)
            .unwrap();

        // Defines variable `ell`
        let vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0).unwrap()
        })
        .collect::<Vec<_>>();
        model.update().unwrap();

        // Defines variable `gamma`.
        let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
            .unwrap();


        // Add the constraint `l * x <= 1`.
        let lx = vars.iter()
            .zip(x)
            .map(|(&vi, &xi)| vi * xi)
            .grb_sum();
        let name = "ell * x == 1";
        model.add_constr(&name, c!(lx == 1_f64)).unwrap();
        model.update().unwrap();


        model.set_objective(gamma, Maximize).unwrap();
        model.update().unwrap();

        (model, vars, gamma)
    }


    fn solve_lp(
        &self,
        model: &mut Model,
        vars: &[Var],
        gamma: &Var,
        c: &[f64],
    ) -> (f64, Vec<f64>)
    {
        assert_eq!(c.len(), vars.len());
        let lhs = c.iter()
            .zip(vars)
            .map(|(&ci, &vi)| ci * vi)
            .grb_sum();
        let k = self.comb_vectors.len();
        let name = format!("C[{k: >5}]");
        model.add_constr(&name, c!(lhs >= gamma)).unwrap();
        model.update().unwrap();


        // --------------------------------------
        // Solve the problem and obtain the optimal solution.
        model.optimize().unwrap();


        let objval = model.get_obj_attr(attr::X, &gamma).unwrap();


        // Get the current primal solution 
        // and calculate a distance
        // from the previous solution to the current one.
        let ell = vars.iter()
            .map(|v| model.get_obj_attr(attr::X, v).unwrap())
            .collect::<Vec<_>>();
        // println!("gamma = {g: >3.3}, objval = {objval: >3.3}");

        (objval, ell)
    }
}


impl<'a> Metarounding for Lp<'a> {
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        let x = x.as_ref();


        // Initial estimation is the uniform distribution.
        let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];

        // `ghat` is the current estimation of 
        // approximation param., `alpha`.
        let mut ghat = f64::MIN;
        let mut gstar = f64::MAX;

        let (mut model, vars, gamma) = self.build_lp(&x);


        // A vector of combinatorial vectors, 
        // collected by current iteration.
        self.comb_vectors = Vec::new();


        // let mut iter = 0;
        loop {
            // Call approximation algorithm and obtain
            // a new combinatorial concept `c`.
            let c = self.oracle.call(&ell);

            assert_eq!(c.len(), ell.len());

            ghat = c.iter()
                .zip(&ell)
                .map(|(ci, li)| ci * li)
                .sum::<f64>()
                .max(ghat);

            let optimality_gap = gstar - ghat;
            if optimality_gap <= self.tolerance {
                break;
            }


            // Solve the LPBoost-like problem
            (gstar, ell) = self.solve_lp(&mut model, &vars, &gamma, &c);
            self.comb_vectors.push(c);
        }


        // Solve LP to obtain the coefficient vector `lambda`.
        let (_, lambda) = solve_primal(&self.env, x, &self.comb_vectors);

        let comb_vectors = std::mem::take(&mut self.comb_vectors);
        (lambda, comb_vectors)
    }
}



