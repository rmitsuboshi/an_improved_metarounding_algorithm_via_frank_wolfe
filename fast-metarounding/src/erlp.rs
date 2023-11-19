//! A Metarounding algorithm based on ERLPBoost.
//! 
//! Paper: Combinatorial Online Prediction via Metarounding
use grb::prelude::*;

use crate::common::*;
use crate::approx_algorithm::Oracle;
use crate::metarounding::Metarounding;


/// A struct that defines the MBB.
pub struct Erlp<'a> {
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


    /// The primal solution
    lambda: Vec<f64>,


    /// Combinatorial vectors
    comb_vectors: Vec<Vec<f64>>,


    /// Gurobi Env
    env: Env,

    model: Model,
    gamma: Var,
    vars: Vec<Var>,
    constrs: Vec<Constr>,
}


impl<'a> Erlp<'a> {
    pub fn new(
        tolerance: f64,
        oracle: &'a Oracle,
    ) -> Self
    {
        let mut env = Env::new("ERLP")
            .expect("Failed to construct a new environment `ERLP`");
        init_env(&mut env);
        let mut model = Model::with_env("Erlp", &env)
            .expect("Failed to construct a new gurobi model");

        let vars = Vec::new();
        let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
            .unwrap();
        let constrs = Vec::new();



        let (_n_items, n_sets) = oracle.shape();
        let half_tolerance = tolerance * 0.5;
        let eta = (n_sets as f64).ln() / half_tolerance;

        let lambda = Vec::new();
        let comb_vectors = Vec::new();

        Self {
            eta,
            _n_items,
            n_sets,
            tolerance,
            half_tolerance,
            oracle,
            lambda,
            comb_vectors,
            env,
            model,
            gamma,
            vars,
            constrs,
        }
    }
}

impl<'a> Erlp<'a> {
    fn max_iter(&self, x: &[f64]) -> usize {
        assert!(self.tolerance > 0.0);
        let m = x.iter()
            .filter_map(|&xi| if xi == 0.0 { None } else { Some(1.0/xi) })
            .reduce(f64::max)
            .expect("The input vector `x` should have a non-zero entry");
        let max_val = self.oracle.max_entry();
        let numer = 16.0 * (m * max_val).powi(2) * self.eta;
        let denom = self.tolerance;

        ((numer / denom) - 2.0).ceil() as usize
    }


    /// Solves the sequential quadratic programming
    /// for the given `model`, `vars`, and `gamma`
    /// with adding a new constraint
    /// `sum( c * x) ≥ γ`.
    ///
    /// The entropic objective cannot be solved by Gurobi,
    /// we solve the quadratic approximation problem repeatedly.
    ///
    /// ---------------------------------------------------
    /// Instead of solving
    /// 
    /// 
    /// ```txt
    /// Maximize γ - (1/η) sum( x[i]*l[i] ln( x[i]*l[i] / (1/n) ) )
    /// sub. to. sum( c[j][i] * x[i] ) ≥ γ, ∀j = 1, 2, ..., t,
    ///          sum( x[i] * l[i] ) = 1,
    ///          l[i] ≥ 0,                  ∀i = 1, 2, ..., n,
    /// ```
    /// 
    /// We solve the following quadratic problems
    /// by updating `p` (Initially, `p[i] = 1/n` for all `i`).
    /// 
    /// 
    /// ```txt
    /// Maximize γ
    ///         - (1.0/η)*sum( {x[i]*ln( x[i]*p[i] / (1/n) )} * l[i] )
    ///         - (0.5/η)*sum( {x[i] / p[i]} * l[i] )
    /// sub. to. sum( c[j][i] * x[i] ) ≥ γ, ∀j = 1, 2, ..., t,
    ///          sum( x[i] * l[i] ) = 1,
    ///          l[i] ≥ 0,                  ∀i = 1, 2, ..., n,
    /// ```
    /// 
    /// This alternative problem differs 
    /// from the original one by constant,
    /// but it is not the problem since we are interested in
    /// the optimal solution.
    /// 
    /// Then,
    /// update `p` as the optimal solution of the alternative problem.
    /// ---------------------------------------------------
    fn solve_sequential_qp(&mut self, x: &[f64], c: &[f64])
        -> grb::Result<(f64, Vec<f64>)>
    {
        assert!(
            x.iter()
            .all(|&xi| xi.is_finite() && (0f64..=1f64).contains(&xi))
        );
        assert!(c.iter().all(|&ci| ci.is_finite()));
        let c_id = self.comb_vectors.len();
        let name = format!("C[{c_id}]");
        let lhs = c.iter().zip(&self.vars).map(|(&ci, &vi)| ci * vi).grb_sum();
        let constr = self.model.add_constr(&name, c!(lhs >= self.gamma))?;
        self.constrs.push(constr);
        self.model.update()?;


        let n_sets = self.n_sets as f64;
        // Current estimation of the optimal solution `l`.
        let mut prev = vec![1.0 / n_sets; self.n_sets];

        let mut g = 0.0;
        let mut prev_objval = f64::MIN / 2.0;
        loop {
            assert!(
                prev.iter().all(|&pi| pi.is_finite()),
                "loss is {:?}",
                prev
            );
            // Set the objective function.
            let objective = self.grb_objective(&prev, x);
            self.model.set_objective(objective, Maximize)?;
            self.model.update()?;
            self.model.optimize()?;


            let status = self.model.status()?;
            if status != Status::Optimal && status != Status::SubOptimal {
                break;
            }


            // At this point, you can get (Sub)Optimal solution.
            g = self.model.get_obj_attr(attr::X, &self.gamma)?;
            self.lambda = self.constrs.iter()
                .map(|c| self.model.get_obj_attr(attr::Pi, c).map(f64::abs))
                .collect::<grb::Result<Vec<_>>>()?;


            // Get the current solution and calculate L2-distance
            // from the previous solution to the current one.
            let next = self.vars.iter()
                .map(|v| self.model.get_obj_attr(attr::X, v))
                .collect::<Result<Vec<_>, _>>()?;


            let has_non_pos = next.iter().any(|&nxt| nxt <= 0.0);
            // let objval = g - self.entropy(&next, x) / self.eta;
            let objval = self.model.get_attr(attr::ObjVal)?;
            assert!(prev.iter().all(|&pi| pi >= 0.0), "prev: {:?}", prev);
            prev = next;
            if has_non_pos || (objval - prev_objval) * 10.0 < self.half_tolerance {
                break;
            }
            prev_objval = objval;
        }

        assert!(
            prev.iter().all(|&p| p >= 0.0),
            "(x, p) = {:?}",
            x.iter()
                .zip(&prev)
                .filter(|(_, &p)| p < 0.0)
                .collect::<Vec<_>>()
        );

        // ---------------------------------------------------
        // At this point, an approximate solution is found.
        // Get the objective value `objval`.
        let objval = g - (self.entropy(&prev, x) / self.eta);

        Ok((objval, prev))
    }


    fn entropy<S, T>(&self, loss: S, x: T) -> f64
        where S: AsRef<[f64]>,
              T: AsRef<[f64]>
    {
        let loss = loss.as_ref();
        assert!(loss.iter().all(|&l| l >= 0.0));
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
    fn build_qp(&mut self, x: &[f64]) -> grb::Result<()> {
        let mut model = Model::with_env("Erlp", &self.env)?;
        assert_eq!(x.len(), self.n_sets);
        self.vars = (0..self.n_sets).map(|i| {
            let name = format!("ell[{i}]");
            let ub = if x[i] == 0.0 { 0.0 } else { (1.0 / x[i]).min(INFINITY) };
            add_ctsvar!(model, name: &name, bounds: 0.0..ub)
        })
        .collect::<grb::Result<Vec<_>>>()?;
        self.gamma = add_ctsvar!(model, name: "gamma", bounds: ..)?;
        self.model = model;
        self.model.update()?;

        let lx = self.vars.iter()
            .zip(x)
            .map(|(&vi, &xi)| vi * xi)
            .grb_sum();
        let name = "sum( ell[i] * x[i] ) == 1";
        self.model.add_constr(&name, c!(lx == 1f64))?;
        self.model.update()?;
        self.constrs = Vec::new();
        Ok(())
    }


    fn grb_objective(
        &mut self,
        prev: &[f64],
        x: &[f64],
    ) -> Expr
    {
        let ub = x.iter()
            .filter_map(|&xi| if xi == 0.0 { None } else { Some(1.0/xi) })
            .reduce(f64::max)
            .unwrap();
        let n_sets = prev.len() as f64;
        let regularizer = prev.iter()
            .zip(x)
            .zip(&self.vars)
            .filter_map(|((pi, xi), &vi)| {
                if *xi == 0.0 || *pi == 0.0 {
                    None
                } else {
                    let linear = xi * (n_sets * pi * xi).ln()
                        .clamp(-INFINITY, (ub * n_sets).ln())
                        / self.eta;
                    let quad = 0.5 * (xi / pi).min(INFINITY) / self.eta;
                    assert!(linear.is_finite(), "linear: {}, pi: {}, xi: {}", linear, pi, xi);
                    assert!(quad.is_finite(), "quad: {}, pi: {}, xi: {}", quad, pi, xi);
                    let expr = linear * vi + quad * (vi * vi);
                    Some(expr)
                }
            })
            .grb_sum();
        self.gamma - regularizer
    }
}


impl<'a> Metarounding for Erlp<'a> {
    fn round<T: AsRef<[f64]>>(&mut self, x: T)
        -> (Vec<f64>, Vec<Vec<f64>>)
    {
        let x = x.as_ref();
        assert!(
            x.iter().all(|&xi| xi >= 0.0),
            "Input vector should be non-negative"
        );


        // Initial estimation is the uniform distribution.
        let mut ell = vec![1.0 / self.n_sets as f64; self.n_sets];

        // `ghat` is the current estimation of 
        // approximation param., `α`.
        let mut gstar = f64::MAX;
        let mut ghat = f64::MIN;


        let max_iter = self.max_iter(x);

        self.build_qp(x).expect("Failed to build QP");


        // A vector of combinatorial vectors, 
        // collected by current iteration.
        self.comb_vectors = Vec::new();

        for _ in 1..=max_iter {
            let c = self.oracle.call(&ell);
            assert_eq!(c.len(), ell.len());

            ghat = c.iter()
                .zip(&ell)
                .map(|(ci, li)| ci * li)
                .sum::<f64>()
                .max(ghat);

            // Optimality gap measures the difference between
            // the current objective value `H*(Cλ_{k})` and
            // the estimate `max { sum(c[j+1][i] * l[i] | j ≤ k }`
            // 
            // The variable `optimality_gap` corresponds to `ε_{k}`
            // in our paper.
            let optimality_gap = gstar - ghat;
            if optimality_gap <= self.half_tolerance { break; }
            self.comb_vectors.push(c.clone());


            (gstar, ell) = self.solve_sequential_qp(&x, &c)
                .expect("Failed to solve sequential QP");
        }
        let lambda = std::mem::take(&mut self.lambda);
        // let (_, lambda) = solve_primal(&self.env, x, &self.comb_vectors)
        //     .expect("Failed to obtain the primal solution");
        let comb_vectors = std::mem::take(&mut self.comb_vectors);
        (lambda, comb_vectors)
    }
}
