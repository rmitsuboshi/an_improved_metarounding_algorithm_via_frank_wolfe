use grb::prelude::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

// Since there might exist an entry of x whose value is significantly small,
// so that we cap the value by `THRESHOLD`.
pub const THRESHOLD: f64 = 1e-9;


/// Solves the primal problem:
/// 
/// ```txt
/// 
/// Minimize β
/// { β, λ }
/// 
/// sub. to. sum( λ[c] * c[i] ) <= β * x[i], for all i = 1, 2, ..., n,
///                 sum( λ[c] ) == 1,
///                        λ[c] >= 0,        for all c in comb_vectors.
/// ```
pub fn solve_primal(
    env: &Env,
    x: &[f64],
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    for c in comb_vectors {
        assert_eq!(x.len(), c.len());
    }

    let n_sets = comb_vectors.len();

    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Solve LP", env).unwrap();

    let beta = add_ctsvar!(model, name: "beta", bounds: ..).unwrap();
    let lambda = (0..n_sets).map(|c| {
            let name = format!("lambda[{c}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0).unwrap()
        })
        .collect::<Vec<_>>();
    model.update().unwrap();


    // Add the constraint `sum( lambda[i] ) == 1`.
    let lhs = lambda.iter().grb_sum();
    let name = "sum( lambda[c] ) == 1";
    model.add_constr(&name, c!(lhs == 1_f64)).unwrap();
    model.update().unwrap();

    for (i, xi) in x.iter().enumerate() {
        let lhs = lambda.iter()
            .zip(comb_vectors)
            .map(|(&lmd, cmb)| lmd * cmb[i])
            .grb_sum();
        let rhs = beta * *xi;
        let name = format!("Coordinate[{i}]");
        model.add_constr(&name, c!(lhs <= rhs)).unwrap();
    }
    model.update().unwrap();

    model.set_objective(beta, Minimize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    assert_eq!(status, Status::Optimal, "Failed to solve the primal LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let lambda = lambda.iter()
        .map(|lmd| model.get_obj_attr(attr::X, lmd).unwrap())
        .collect::<Vec<_>>();


    (objval, lambda)
}


/// Solves the soft-margin-like primal problem:
/// 
/// ```txt
/// 
/// Minimize β + M sum( ξ[i] )
/// sub. to. sum( λ[c] * c[i] ) <= β * x[i] + ξ[i], for all i = 1, 2, ..., n,
///                 sum( λ[c] ) == 1,
///                        λ[c] >= 0,               for all c in comb_vectors,
///                        ξ[i] >= 0,               for all i = 1, 2, ..., n.
/// ```
pub fn solve_primal_bounded(
    env: &Env,
    x: &[f64],
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    assert!(comb_vectors.len() > 0);
    for c in comb_vectors {
        assert_eq!(x.len(), c.len());
    }

    let t = comb_vectors.len();
    let n = x.len();


    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Solve LP", env).unwrap();

    let beta = add_ctsvar!(model, name: "beta", bounds: 0.0..)
        .expect("Failed to construct Gurobi variable `beta`");
    let lambda = (0..t).map(|c| {
            let name = format!("lambda[{c}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to construct Gurobi variables `lambda[..]`");
    let psi = (0..n).map(|i| {
            let name = format!("psi[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to construct Gurobi variables `psi[..]`");
    model.update().unwrap();


    // Add the constraint `sum( λ[i] ) == 1`.
    let lhs = lambda.iter().grb_sum();
    let name = "sum( lambda[c] ) == 1";
    model.add_constr(&name, c!(lhs == 1_f64)).unwrap();
    model.update().unwrap();


    for (i, xi) in x.iter().enumerate() {
        let lhs = lambda.iter()
            .zip(comb_vectors)
            .map(|(&lmd, cmb)| lmd * cmb[i])
            .grb_sum();
        let rhs = (beta * *xi) + psi[i];
        let name = format!("Coordinate[{i}]");
        model.add_constr(&name, c!(lhs <= rhs)).unwrap();
    }
    model.update().unwrap();

    assert!(x.iter().any(|&xi| xi > 0.0));

    let m = x.iter()
        // .filter_map(|xi| if *xi <= THRESHOLD { None } else { Some((1.0 / *xi).min(INFINITY)) })
        .filter_map(|xi| if *xi == 0.0 { None } else { Some(1.0 / *xi) })
        .reduce(f64::max)
        .map(|m| m.clamp(0.0, INFINITY))
        .expect("The input vector should have a non-zero entry");
    let objective = beta + (m * psi.iter().grb_sum());
    // let objective = psi.iter()
    //     .zip(x)
    //     .filter(|(_, xi)| **xi > 0.0)
    //     .map(|(&psii, &xi)| (1.0 / xi).clamp(0.0, INFINITY) * psii)
    //     .grb_sum()
    //     + beta;
    model.set_objective(objective, Minimize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    if status != Status::Optimal {
        println!("x = {x:?}");
    }
    assert_eq!(status, Status::Optimal, "Failed to solve the primal LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let lambda = lambda.iter()
        .map(|lmd| model.get_obj_attr(attr::X, lmd).unwrap())
        .collect::<Vec<_>>();


    (objval, lambda)
}


/// Get the optimal primal solution from the dual problem.
/// 
/// ```txt
/// Maximize γ
/// sub. to. sum( c[i] * ell[i] ) ≥ γ for all c in comb_vectors
///          sum( x[i] * ell[i] ) ≤ 1
///          0 ≤ ell[i] ≤ 1/x[i] for all i s.t. x[i] ≠ 0
///          ell[i] = 0          for all i s.t. x[i] = 0
/// ```
pub fn solve_primal_from_bounded_dual(
    env: &Env,
    x: &[f64],
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    let n_items = x.len();

    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Solve dual LP", env).unwrap();

    let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
        .unwrap();

    // let ub = x.iter()
    //     .filter_map(|&xi| if xi == 0.0 { None } else { Some(1.0 / xi) })
    //     .reduce(f64::max)
    //     .unwrap_or(INFINITY);
    let ell = (0..n_items).map(|i| {
            let name = format!("ell[{i}]");
            let ub = if x[i] == 0.0 { 0.0 } else { 1.0 / x[i] } + 1.0;
            add_ctsvar!(model, name: &name, bounds: 0.0..ub)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to construct the dual variables `ell[..]`");
    model.update().unwrap();


    // Add the constraint `sum( ell[i] * x[i] ) == 1`.
    let lhs = ell.iter()
        .zip(x)
        .map(|(&li, &xi)| li * xi)
        .grb_sum();
    let name = "sum( ell[i] * x[i] ) == 1";
    model.add_constr(&name, c!(lhs == 1_f64)).unwrap();
    model.update().unwrap();

    let constraints = comb_vectors.iter()
        .enumerate()
        .map(|(i, cmb)| {
            assert_eq!(cmb.len(), ell.len());
            let lhs = ell.iter()
                .zip(cmb)
                .map(|(&l, &c)| l * c)
                .grb_sum();
            let name = format!("Comb[{i}]");
            model.add_constr(&name, c!(lhs >= gamma))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to set the constraints");
    model.update().unwrap();

    model.set_objective(gamma, Maximize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    assert_eq!(status, Status::Optimal, "Failed to solve bounded dual LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let lambda = constraints.iter()
        .map(|c| model.get_obj_attr(attr::Pi, c).map(f64::abs))
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to get the dual solution");
    // println!("lambda = {lambda:?}");
    // println!("sum = {}", lambda.iter().sum::<f64>());
    assert!(lambda.iter().all(|&l| l >= 0.0));
    assert!((1.0 - lambda.iter().sum::<f64>()).abs() < 1e-6);

    (objval, lambda)
}


/// Get the optimal primal solution from the dual problem.
/// 
/// ```txt
/// Maximize γ
/// sub. to. sum( c[i] * ell[i] ) ≥ γ for all c in comb_vectors
///          sum( x[i] * ell[i] ) ≤ 1
///          0 ≤ ell[i] ≤ 1/x[i] for all i s.t. x[i] ≠ 0
///          ell[i] = 0          for all i s.t. x[i] = 0
/// ```
pub fn solve_primal_from_dual(
    env: &Env,
    x: &[f64],
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    let n_items = x.len();

    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Solve dual LP", env).unwrap();

    let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
        .unwrap();

    // let univ_ub = x.iter()
    //     .filter_map(|&xi| if xi == 0.0 { Some(1.0 / xi) } else { None })
    //     .reduce(f64::max)
    //     .map(|ub| ub.clamp(0.0, INFINITY))
    //     .unwrap_or(INFINITY);
    let ell = (0..n_items).map(|i| {
            let name = format!("ell[{i}]");
            let ub = if x[i] == 0.0 { 0.0 } else { 1.0 / x[i] };
            add_ctsvar!(model, name: &name, bounds: 0.0..ub)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to construct the dual variables `ell[..]`");
    model.update().unwrap();


    // Add the constraint `sum( ell[i] * x[i] ) == 1`.
    let lhs = ell.iter()
        .zip(x)
        .map(|(&li, &xi)| li * xi)
        .grb_sum();
    let name = "sum( ell[i] * x[i] ) == 1";
    model.add_constr(&name, c!(lhs == 1_f64)).unwrap();
    model.update().unwrap();

    let constraints = comb_vectors.iter()
        .enumerate()
        .map(|(i, cmb)| {
            assert_eq!(cmb.len(), ell.len());
            let lhs = ell.iter()
                .zip(cmb)
                .map(|(&l, &c)| l * c)
                .grb_sum();
            let name = format!("Comb[{i}]");
            model.add_constr(&name, c!(lhs >= gamma))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to set the constraints");
    model.update().unwrap();

    model.set_objective(gamma, Maximize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    if status != Status::Optimal {
        println!("x = {x:?}");
        for (i, comb) in comb_vectors.iter().enumerate() {
            let ones = comb.iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > 0.0 { Some(i.to_string()) } else { None })
                .collect::<Vec<_>>()
                .join(" ");
            println!("COMB[{i}] = {ones}");
        }
    }
    assert_eq!(status, Status::Optimal, "Failed to solve dual LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let lambda = constraints.iter()
        .map(|c| model.get_obj_attr(attr::Pi, c).map(f64::abs))
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to get the dual solution");
    // println!("lambda = {lambda:?}");
    // println!("sum = {}", lambda.iter().sum::<f64>());
    assert!(lambda.iter().all(|&l| l >= 0.0));
    assert!((1.0 - lambda.iter().sum::<f64>()).abs() < 1e-6);

    (objval, lambda)
}


/// Solve the dual problem
/// 
/// ```txt
/// Maximize γ
/// sub. to. sum( c[i] * ell[i] ) ≥ γ for all c in comb_vectors
///          sum( x[i] * ell[i] ) ≤ 1
///          0 ≤ ell[i] for all i = 1, 2, ..., n
/// ```
pub fn solve_dual(
    env: &Env,
    x: &[f64],
    comb_vectors: &[Vec<f64>],
) -> (f64, Vec<f64>)
{
    // let n_sets = comb_vectors.len();
    let n_items = x.len();

    // Solve LP to obtain the coefficient vector `lambda`.
    let mut model = Model::with_env("Solve dual LP", env).unwrap();

    let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
        .unwrap();

    let ub = x.iter()
        .filter_map(|xi| if *xi == 0.0 { None } else { Some(1.0 / *xi) })
        .reduce(f64::max)
        .expect("The input vector should have a non-zero entry");
    let ell = (0..n_items).map(|i| {
            let name = format!("ell[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..ub)
                .unwrap()
        })
        .collect::<Vec<_>>();
    model.update().unwrap();


    // Add the constraint `sum( ell[i] * x[i] ) == 1`.
    let lhs = ell.iter()
        .zip(x)
        .map(|(&li, &xi)| li * xi)
        .grb_sum();
    let name = "sum( ell[i] * x[i] ) == 1";
    model.add_constr(&name, c!(lhs == 1_f64)).unwrap();
    model.update().unwrap();

    for (i, cmb) in comb_vectors.iter().enumerate() {
        assert_eq!(cmb.len(), ell.len());
        let lhs = ell.iter()
            .zip(cmb)
            .map(|(&l, &c)| l * c)
            .grb_sum();
        let name = format!("Comb[{i}]");
        model.add_constr(&name, c!(lhs >= gamma)).unwrap();
    }
    model.update().unwrap();

    model.set_objective(gamma, Maximize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    assert_eq!(status, Status::Optimal, "Failed to solve dual LP");

    let objval = model.get_attr(attr::ObjVal).unwrap();

    let ell = ell.iter()
        .map(|l| model.get_obj_attr(attr::X, l).unwrap())
        .collect::<Vec<_>>();


    (objval, ell)
}


/// Returns the distance from `src` to `dst`.
/// Currently, I adopted the relative entropy as pseudo-distance.
pub fn distance(src: &[f64], dst: &[f64]) -> f64 {
    dst.iter()
        .zip(src)
        .map(|(&d, &s)| (d - s).powi(2))
        .sum::<f64>()
        .sqrt()
    // assert!(src.iter().all(|&s| s > 0.0));
    // dst.iter()
    //     .zip(src)
    //     .map(|(&d, &s)| if d <= 0.0 { 0.0 } else { d * (d / s).ln() })
    //     .sum()
}


/// Returns the current `alpha`.
pub fn instant_alpha(
    x: &[f64],
    lambda: &[f64],
    comb_vectors: &[Vec<f64>],
) -> f64
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


/// Construct the loss vectors for all rounds `1..=T` a priori.
pub fn build_losses(
    dim: usize,
    n_rounds: usize,
    seed: u64,
) -> Vec<Vec<f64>>
{
    let mut rng = StdRng::seed_from_u64(seed);
    let distribution = Uniform::from(0.0..1.0);

    (0..n_rounds).map(|_| {
        distribution.sample_iter(&mut rng)
            .take(dim)
            .collect()
    })
    .collect()
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
    assert!((1.0 - dist.iter().sum::<f64>()).abs() < 1e-6);
    let mut sum = 0.0;
    for (i, d) in dist.iter().copied().enumerate() {
        sum += d;
        if rv < sum { return i; }
    }
    return dist.len() - 1;
}
