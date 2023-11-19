use std::time::Instant;
use std::fs::File;
use std::io::prelude::*;

use grb;
use grb::prelude::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::common::*;
use crate::dispatch::*;
use crate::approx_algorithm::Oracle;


/// The struct that might be
/// returned by `separation_or_decomposition_via_ellipsoid_method`.
pub(self) struct Decomposition {
    pub(self) coefficients: Vec<f64>,
    pub(self) vs: Vec<(Vec<f64>, Vec<f64>)>,
}


impl Decomposition {
    /// Construct a new instance of `Decomposition`.
    /// The arguments must have the same length.
    /// If not, this method panics.
    pub(self) fn new(
        coefficients: Vec<f64>,
        vs: Vec<(Vec<f64>, Vec<f64>)>
    ) -> Self
    {
        assert_eq!(vs.len(), coefficients.len());
        Self { coefficients, vs, }
    }
}


/// Performs the algorithm proposed by Kakade-Kalai-Ligett.
/// For each round `t = 1, 2, ...`,
/// the algorithm calls the approximation oracle `O(T)` times.
/// Their algorithm achieves `O(√T)` α-regret in expectation.
pub fn garber17(
    args: &Args,
    n_rounds: usize,
    oracle: &Oracle,
    alpha: f64,
)
{
    let mut file = File::create(args.output_name())
        .expect("Failed to create file");
    file.write_all(HEADER.as_bytes())
        .expect("Failed to write header");
    let name = "Garber17";
    let mut env = Env::new(name).unwrap();
    env.set(param::OutputFlag, 0).unwrap();


    let mut rng = StdRng::seed_from_u64(args.algo_seed);
    let uniform = Uniform::from(0.0..1.0);
    let (_, n_sets) = oracle.shape();


    // ***********************************
    // Initial prediction over `relax(C)`.
    // TODO choose the proper init. point `s`.
    let mut s = vec![1.0; n_sets];
    s = oracle.call(&s);
    let mut y_tilde = s.iter().map(|&si| alpha * si).collect::<Vec<_>>();

    let diam = (n_sets as f64).sqrt();

    // ***********************************************
    // Set the parameter `η` and `ε`.
    // Here in this setting, the parameters are
    //  - `R = max { ‖s‖ | s ∈ K } = √n`,
    //  - `F = max { ‖f‖ | f ∈ F } = √n`.

    // First option:
    //      - `η = α * R / (F * T^{2/3})`
    //      - `ε = α * R / T^{1/3}
    // This setting guarantees
    //      - `O(α * R * F * T^{2/3})` expected regret,
    //      - `O(n² * ln( T * (α + 1) / α )` oracle call per round.
    // let eta = alpha / (n_rounds as f64).cbrt().powi(2);
    // let epsilon = alpha * diam / (n_rounds as f64).cbrt();


    // Second option:
    //      - `η = α * R / (F * √T)`
    //      - `ε = α * R / √T
    // This setting guarantees
    //      - `O(α * R * F * √T)` expected regret,
    //      - `O(n² * √T * ln( T * (α + 1) / α )` oracle call per round.
    let eta = alpha / (n_rounds as f64).powi(2).sqrt();
    let epsilon = alpha * diam / (n_rounds as f64).sqrt();


    let seed = args.loss_seed;
    let (loss_sum, loss_iter) = build_losses(n_sets, n_rounds, seed);

    let best = oracle.best_action_hindsight(&loss_sum);
    let mut play_loss_sum = 0.0;
    let mut best_loss_sum = 0.0;

    println!("--------------- OCO STARTS ---------------");
    println!("{: >5}\t{: >6}\t{: >6}", "ROUND", "PLAY", "BEST");
    for (round, loss) in (1..=n_rounds).zip(loss_iter) {
        let now = Instant::now();


        // *****************************************
        // Update the cumulative loss vector & value
        let loss_val = iproduct(&loss, &s);
        play_loss_sum += loss_val;


        let best_loss_val = iproduct(&best, &loss);
        best_loss_sum += best_loss_val;
        println!("{round: >5}\t{loss_val: >6.2}\t{best_loss_val: >6.2}");


        let y = y_tilde.iter()
            .zip(loss)
            .map(|(yi, fi)| yi - eta * fi)
            .collect::<Vec<_>>();
        let (y_tilde_new, decomp) = infeasible_projection(
            &env, y, epsilon, oracle, diam, alpha
        );


        y_tilde = y_tilde_new;

        let r = uniform.sample(&mut rng);
        let ix = choose(&decomp.coefficients, r);
        s = decomp.vs[ix].1.clone();

        let time = now.elapsed().as_millis();
        let line = format!("{time},{loss_val},{best_loss_val}\n");
        file.write_all(line.as_bytes())
            .expect("Failed to write log");
    }
    println!("--------------- OCO FINISH ---------------");
    println!("CUMULATIVE LOSS");
    println!("\t* [PLAY] {play_loss_sum: >10.2}");
    println!("\t* [BEST] {best_loss_sum: >10.2}");
    println!("[1-REGRET] {: >10.2}", play_loss_sum - best_loss_sum);
}


/// The infeasible projection algorithm,
/// depicted in Algorithm 2 of the following paper:
///
/// Dan Garber,
/// Efficient Online Linear Optimization with Approximation Algorithms,
/// NIPS 2017.
///
fn infeasible_projection(
    env: &Env,
    y: Vec<f64>,
    epsilon: f64,
    oracle: &Oracle,
    diam: f64,
    alpha: f64,
) -> (Vec<f64>, Decomposition)
{
    assert!(0.0 < epsilon && epsilon <= (alpha + 2.0) * diam);
    const UNIV_CONST: f64 = 1.0;
    let norm = y.iter().map(|yi| yi.powi(2)).sum::<f64>().sqrt();
    let denom = 1.0_f64.max(norm / (alpha * diam));
    let mut y_tilde = y.iter()
        .map(|&yi| yi / denom)
        .collect::<Vec<_>>();

    let n_rounds = (alpha * diam / epsilon).powi(2).ceil() as usize;
    for _ in 1..=n_rounds {
        let ret = separation_or_decomposition_via_ellipsoid_method(
            env, &y_tilde, epsilon, UNIV_CONST, alpha, diam, oracle
        );

        match ret {
            Ok(w) => {
                assert!(w.iter().any(|&wi| wi > 0.0));
                y_tilde.iter_mut()
                    .zip(w)
                    .for_each(|(yi, wi)| { *yi -= epsilon * wi; });
            },
            Err(decomp) => { return (y_tilde, decomp); },
        }
    }
    unreachable!()
}


/// A procedure defined in Lemma 3.1. of the following paper:
///
/// Dan Garber,
/// Efficient Online Linear Optimization with Approximation Algorithms,
/// NIPS 2017.
///
/// This procedure tries to find a `w ∈ Rⁿ` such that
/// ```txt
///     (x - z) * w ≥ ε, for all z ∈ αK,
///     ‖w‖ ≤ 1
/// ```
fn separation_or_decomposition_via_ellipsoid_method<T>(
    env: &Env,
    x: T,            // x ∈ Rⁿ
    epsilon: f64,    // ε ∈ (0, (α + 2)R]
    univ_const: f64, // c ∈ R s.t. c > 0
    alpha: f64,      // α ∈ R
    diam: f64,       // R ∈ Rⁿ
    oracle: &Oracle, // O : Rⁿ → K
) -> Result<Vec<f64>, Decomposition>
    where T: AsRef<[f64]>
{
    assert!(0.0 < epsilon && epsilon <= (alpha + 2.0) * diam);
    let x = x.as_ref();
    let norm_x = x.iter().map(|xi| xi.powi(2)).sum::<f64>().sqrt();

    let n = x.len();
    let n_rounds = {
        let antilog = ((alpha + 1.0) * diam + norm_x) / epsilon;
        (univ_const * n.pow(2) as f64 * antilog.ln()).ceil() as usize
    };


    let mut vs = Vec::new();


    // Construct a initial ellipsoid `ball`
    let mut center = vec![0.5f64; n];
    let radius = 2.0;
    // let radius = 4.0 * alpha.powi(2);
    let mut ball = (0..n).map(|i| {
            let mut row = vec![0_f64; n];
            row[i] = radius;
            row
        }).collect::<Vec<_>>();
    for _ in 1..=n_rounds {

        // Call the separation oracle `(v, s) ← O(-w)`
        let neg_w = center.iter()
            .map(|ci| -ci)
            .collect::<Vec<_>>();
        let (v, s) = extended_approximation_oracle(neg_w, oracle, alpha, diam);



        // If `(x - v) * w < ε`,
        // Use `v - x` as the normal vector of the separating hyperplane.
        let lhs = x.iter()
            .zip(&v)
            .map(|(&xi, &vi)| xi - vi)
            .zip(&center)
            .map(|(xi_vi, &wi)| xi_vi * wi)
            .sum::<f64>();
        if lhs < epsilon {
            let sep = v.iter()
                .zip(x)
                .map(|(&vi, &xi)| vi - xi)
                .collect::<Vec<_>>();
            update_ellipsoid(&mut center, &mut ball, &sep);
            vs.push((v, s));
            continue;
        }


        // If ‖w‖ > 1, use `w` as the separating hyperplane.
        let w_norm = center.iter()
            .map(|ci| ci.powi(2))
            .sum::<f64>()
            .sqrt();
        if w_norm > 1.0 {
            let sep = center.clone();
            // let sep = center.iter().map(|&ci| -ci).collect::<Vec<_>>();
            update_ellipsoid(&mut center, &mut ball, &sep);
            vs.push((v, s));
            continue;
        }

        // ----------------------------------------------------
        // At this point, the procedure successfully finds `w`.
        println!("Success.");
        return Ok(center);
    }

    // At this point, the ellipsoid method failed in `n_rounds` rounds.
    let coef = solve_convex_optimization(env, &x, &vs)
        .expect("Failed to solve QP");
    let decomp = Decomposition::new(coef, vs);
    Err(decomp)
}


/// This method updates the ellipsoid finding a feasible solution of:
/// ```txt
/// S = { x ∈ Rⁿ | Ax ≤ b, x ≥ 0 }
/// ```
///
/// Updates the ellipsoid `P := p` centered at `x := center`
/// according to the following rules:
///
/// 1. Let `(a, b) ∈ RⁿxR` be a constraint such that `a * x > b.`
///
/// 2. If `√aPa ≤ ε`, return `x`.
///
/// 3. Update ellipsoid P.
///
///     3-a. q := a / ( √aPa )
///     3-b. x := x - ( Pq / (n + 1) )
///     3-c. P := ( n²/(n²-1) ) ( P - ( 2/(n+1) Pq qP ) )
///
///
/// ```txt
///      Input: x ∈ Rⁿ
///     Output: w ∈ Rⁿ s.t. (x - z) * w ≥ ε, ∀z ∈ αK,
///                         ‖w‖ ≤ 1.
/// ```
fn update_ellipsoid(
    center: &mut [f64],
    p: &mut Vec<Vec<f64>>,
    a: &[f64],
)
{
    let n = center.len();
    assert_eq!(n, center.len());
    assert_eq!(n, a.len());

    // `root_apa = √aPa`
    let apa = quadform(p, &a[..]);
    let root_apa = apa.sqrt();
    assert!(root_apa > 0.0, "root_apa: {root_apa},\na = {a:?}");


    // ----------------------------------------------------
    // 3-a. `q = a / ( √aPa )`
    let q = a.iter()
        .map(|ai| ai / root_apa)
        .collect::<Vec<_>>();


    // ----------------------------------------------------
    // 3-b. `x = x - ( Pq / (n+1) )`
    let pq = p.iter().map(|pi| iproduct(pi, &q)).collect::<Vec<_>>();


    center.iter_mut()
        .zip(&pq)
        .for_each(|(c, &pqi)| { *c -= pqi / (n as f64 + 1.0); });


    // ----------------------------------------------------
    // 3-c. P := ( n²/(n²-1) ) ( P - ( 2/(n+1) Pq qP ) )
    let nf = n as f64;
    let coef = (nf / (nf + 1.0)) * (nf / (nf - 1.0));
    let mut p_new = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            // tmp := (2 / (n+1)) Pq qP
            let tmp = (2.0 / (nf + 1.0)) * pq[i] * pq[j];
            // P := coef (P - tmp)
            //      ^^^^      ^^^
            //    n²/(n²-1)   (2/(n+1) PqqP)
            p_new[i][j] = coef * (p[i][j] - tmp);
        }
    }
    *p = p_new;
}


fn solve_convex_optimization(
    env: &Env,
    x: &[f64],
    vs: &[(Vec<f64>, Vec<f64>)],
) -> grb::Result<Vec<f64>>
{
    let name = "Conv. Optim. Model";
    let mut model = Model::with_env(name, env)?;

    let n_vs = vs.len();
    let vars = (0..n_vs).map(|i| {
            let name = format!("a[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..)
        })
        .collect::<grb::Result<Vec<_>>>()?;
    model.update()?;

    // Add the simplex constraint.
    let name = "sum( a[i] ) == 1";
    model.add_constr(&name, c!(vars.iter().grb_sum() == 1f64))?;
    model.update()?;


    // Set the objective function
    let objective = (0..n_vs).map(|i| {
        let ai = vars[i];
        let vi = &vs[i].0;
        let vi_norm = vi.iter().map(|vii| vii.powi(2)).sum::<f64>();
        let x_vi = iproduct(x, vi);
        (0..i).map(|j| {
            let aj = vars[j];
            let vj = &vs[j].0;
            let vi_vj = iproduct(vi, vj);
            vi_vj * (ai * aj) // non-diag.
        })
        .grb_sum()
        + 0.5 * vi_norm * (ai * ai) // diag.
        - x_vi * ai // linear.
    })
    .grb_sum();
    model.set_objective(objective, Minimize)?;
    model.update()?;

    model.optimize()?;


    let status = model.status()?;
    assert_eq!(status, Status::Optimal, "Failed to solve the approx. LP");

    vars.iter()
        .map(|ai| model.get_obj_attr(attr::X, ai))
        .collect::<grb::Result<Vec<_>>>()
}


/// The extended approximation oracle `Ô : Rⁿ → (K + B(0, (1+α)R), K)`
/// for loss objectives,
/// defined Lemma 2.1. of the following paper:
///
/// Dan Garber,
/// Efficient Online Linear Optimization with Approximation Algorithms,
/// NIPS 2017.
///
/// Let `O : Rⁿ → K` be the approximation oracle s.t.
/// ```txt
/// ∀c ∈ Rⁿ, O(c) * c ≤ α min { x * c | x ∈ K }
/// ```
///
/// Given a vector `c ∈ Rⁿ`, define `c⁺, c⁻` as
/// ```txt
///     ∀i ∈ [n], c⁺[i] = max(0, c[i]),
///     ∀i ∈ [n], c⁻[i] = min(0, c[i]),
/// ```
/// so that `c = c⁺ + c⁻`.
/// Then, `Ô(c)` returns a pair `(v, s)` of vectors such that
/// ```txt
///     v = O(c⁺) - (αR / ‖c⁻‖) c⁻,
///     s = O(c⁺).
/// ```
///
/// The arguments corresponds to the followings:
#[inline(always)]
fn extended_approximation_oracle<T>(
    c: T,            // c ∈ Rⁿ
    oracle: &Oracle, // O : Rⁿ → K
    alpha: f64,      // α ∈ R
    diam: f64,       // R ∈ Rⁿ
) -> (Vec<f64>, Vec<f64>)
    where T: AsRef<[f64]>
{
    assert!(alpha >= 1.0);

    let c = c.as_ref();
    let pos = c.iter().map(|ci| ci.max(0.0)).collect::<Vec<_>>();
    let neg = c.iter().map(|ci| ci.min(0.0)).collect::<Vec<_>>();

    // DEBUG
    assert!(pos.iter().all(|&p| p >= 0.0));
    assert!(neg.iter().all(|&n| n <= 0.0));


    let n = neg.len();
    let neg_norm = neg.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    let normalized_neg = if neg_norm == 0.0 {
        vec![0.0; n]
    } else {
        let inv_neg_norm = 1.0 / neg_norm;
        neg.into_iter().map(|v| v * inv_neg_norm).collect()
    };


    let s = oracle.call(pos);
    let v = s.iter()
        .zip(&normalized_neg)
        .map(|(&si, &nnegi)| si - alpha * diam * nnegi)
        .collect::<Vec<_>>();
    (v, s)
}




