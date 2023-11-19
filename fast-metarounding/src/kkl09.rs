use std::time::Instant;
use std::fs::File;
use std::io::prelude::*;

use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::common::*;
use crate::dispatch::*;
use crate::approx_algorithm::Oracle;

/// Performs the algorithm proposed by Kakade-Kalai-Ligett.
/// For each round `t = 1, 2, ...`,
/// the algorithm calls the approximation oracle `O(T)` times.
/// Their algorithm achieves `O(√T)` α-regret in expectation.
pub fn kkl09(
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
    let mut rng = StdRng::seed_from_u64(args.algo_seed);
    let (_, n_sets) = oracle.shape();


    // ***********************************
    // Initial prediction over `relax(C)`.
    let mut x = vec![1.0; n_sets];
    let mut c = x.clone();


    // ***********************************************
    // Set the parameter `η = (α + 1) * R / (W * √T)`.
    // Here in this setting, the parameters are
    //  - `R = max  ‖c‖   = √n =: domain_diam`,
    //  - `W = max ‖loss‖ = √n`.
    let eta = (alpha + 1.0) / (n_rounds as f64).sqrt();


    // ***************************************
    // Set the parameter `δ = (α + 1) R² / T`.
    let domain_diam = (n_sets as f64).sqrt();
    let delta = (alpha + 1.0) * domain_diam.powi(2) / n_rounds as f64;


    // *************************************************
    // Set the parameter `λ = (α + 1) / (4 (α + 2)² T)`.
    let lambda = (alpha + 1.0)
        / (4.0 * (alpha + 2.0).powi(2) * n_rounds as f64);


    let seed = args.loss_seed;
    let (loss_sum, loss_iter) = build_losses(n_sets, n_rounds, seed);

    let best = oracle.best_action_hindsight(&loss_sum);
    let mut play_loss_sum = 0.0;
    let mut best_loss_sum = 0.0;


    println!("---------- OCO STARTS ----------");
    println!("{: >4}\t{: >6}\t{: >6}", "ROUND", "PLAY", "BEST");
    for (round, loss) in (1..=n_rounds).zip(loss_iter) {
        print!("{round: >4}\t");
        let now = Instant::now();


        let loss_val = iproduct(&loss, &c);
        play_loss_sum += loss_val;
        print!("{loss_val: >6.2}\t");


        let best_loss_val = iproduct(&best, &loss);
        best_loss_sum += best_loss_val;
        println!("{best_loss_val: >6.2}");



        let z = x.iter()
            .zip(&loss)
            .map(|(&xi, &li)| xi - eta * li)
            .collect::<Vec<_>>();


        (x, c) = approx_projection(
            x, c, z, oracle, &mut rng, alpha, delta, lambda, domain_diam,
        );

        let time = now.elapsed().as_millis();
        let line = format!("{time},{loss_val},{best_loss_val}\n");
        file.write_all(line.as_bytes())
            .expect("Failed to write log");
        // lines.push(line);
    }
    println!("---------- OCO FINISH ----------\n");
    println!("\t* [PLAY] {play_loss_sum: >10.2}");
    println!("\t* [BEST] {best_loss_sum: >10.2}");
    // (lines, cumulative_vec)
}


/// The algorithm depicted in Fig. 3.3 of the paper:
///
/// Kakade, Kalai, and Ligett
/// Playing Games with Approximation Algorithms
fn approx_projection(
    mut x: Vec<f64>,
    mut s: Vec<f64>,
    z: Vec<f64>,
    oracle: &Oracle,
    rng: &mut StdRng,
    alpha: f64,
    delta: f64,
    lambda: f64,
    domain_diam: f64,
) -> (Vec<f64>, Vec<f64>)
{
    assert!((0.0..=1.0).contains(&lambda));
    let uniform = Uniform::from(0.0..1.0);


    loop {
        let v = x.iter()
            .zip(&z)
            .map(|(xi, zi)| *xi - *zi)
            .collect::<Vec<_>>();

        // Call the extended approximation oracle.
        // The extended approximation oracle `B : Rⁿ → S x Rⁿ` takes
        // `v ∈ Rⁿ` as input and returns `(t, y)` defined as:
        // (i)  if `v ∈ W₊`, the algorithm returns `(t, y) = (A(v), Φ(A(v))`
        //      where `A` is the approximation algorithm (`Oracle`).
        // (ii) otherwise, let `w = Π(v) ∈ W₊`, `t = A(w)`, and
        //      `y = t + R(α + 1) (w - v) / ||w - v||`.
        //      This case, the algorithm returns `(t, y)`.
        let (t, y) = if v.iter().all(|&vi| vi >= 0.0) {
            // Case (i)
            let t = oracle.call(&v);
            let y = t.clone();
            (t, y)
        } else {
            // Case (ii)

            // Let `w = Π(v) ⊂ W₊ = [0, ∞)ⁿ`.
            let w = v.iter().map(|vi| vi.max(0.0)).collect::<Vec<_>>();

            // Call the approx. oracle `A` to get `t = A(w)`.
            let t = oracle.call(&w);


            // `norm := || w - v ||`.
            let norm = w.iter()
                .zip(&v)
                .map(|(&wi, &vi)| (wi - vi).powi(2))
                .sum::<f64>()
                .sqrt();
            // `coef := R (α + 1) / norm`.
            let coef = domain_diam * (alpha + 1.0) / norm;
            let y = w.into_iter()
                .zip(&v)
                .map(|(wi, &vi)| coef * (wi - vi)) // `= R(α+1) (w-v) / ‖w-v‖`
                .zip(&t)
                .map(|(wvi, &ti)| ti + wvi)
                .collect::<Vec<_>>();
            (t, y)
        };


        // `v := x - z`.
        // `lhs := sum( x[i] * (x[i] - z[i]) )`.
        let lhs = v.iter()
            .zip(&x)
            .map(|(&vi, &xi)| vi * xi)
            .sum::<f64>();


        // `lhs := sum( y[i] * (x[i] - z[i]) )`.
        let rhs = v.iter()
            .zip(&y)
            .map(|(&vi, &yi)| vi * yi)
            .sum::<f64>();

        if lhs <= rhs + delta {
            return (x, s);
        } else {
            let rv = uniform.sample(rng);
            if rv <= lambda {
                s = t;
            }
            x = y.into_iter()
                .zip(x)
                .map(|(yi, xi)| lambda * yi + (1.0 - lambda) * xi)
                .collect::<Vec<_>>();
        }
    }
}
