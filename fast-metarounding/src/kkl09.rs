use std::time::Instant;

use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::common::*;
use crate::approx_algorithm::Oracle;

/// Performs the algorithm proposed by Kakade-Kalai-Ligett.
/// For each round `t = 1, 2, ...`,
/// the algorithm calls the approximation oracle `O(T)` times.
/// Their algorithm achieves `O(√T)` α-regret in expectation.
pub fn kkl09(
    seed_loss: u64,
    seed_choose: u64,
    n_rounds: usize,
    oracle: &Oracle,
    alpha: f64,
) -> (Vec<String>, Vec<f64>)
{
    let mut rng = StdRng::seed_from_u64(seed_choose);
    let (_, n_sets) = oracle.shape();


    // ********************************************
    // Get the loss vector defined before the game.
    let losses = build_losses(n_sets, n_rounds, seed_loss);


    // ***********************************
    // Initial prediction over `relax(C)`.
    let mut x = vec![1.0; n_sets];
    let mut c = x.clone();
    let mut mean = c.clone();


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



    let mut lines = Vec::with_capacity(n_rounds);
    let mut cumulative_vec = vec![0.0; n_sets];


    // *******************************************************************
    // Computes the cumulative loss over `n_rounds` losses a priori
    // to obtain the best prediction in hindsight.
    // The resulting vector is used for compute the regret for each round.
    let mut loss_sum = vec![0.0; n_sets];
    for loss in &losses {
        loss_sum.iter_mut()
            .zip(loss)
            .for_each(|(a, &b)| { *a += b; });
    }
    let best = oracle.best_action_hindsight(&loss_sum);
    let mut play_loss_sum = 0.0;
    let mut mean_loss_sum = 0.0;
    let mut best_loss_sum = 0.0;


    println!("---------- OCO STARTS ----------");
    println!("{: >4}\t{: >6}\t{: >6}\t{: >6}", "ROUND", "PLAY", "MEAN", "BEST");
    for (round, loss) in (1..=n_rounds).zip(losses) {
        print!("{round: >4}\t");
        let now = Instant::now();


        // *****************************************
        // Update the cumulative loss vector & value
        cumulative_vec.iter_mut()
            .zip(&loss)
            .for_each(|(cl, &l)| { *cl += l; });


        let loss_val = loss.iter()
            .zip(&c)
            .map(|(&l, &c)| l * c)
            .sum::<f64>();
        play_loss_sum += loss_val;
        print!("{loss_val: >6.2}\t");


        let mean_loss_val = loss.iter().copied()
            .zip(mean)
            .map(|(l, m)| l * m)
            .sum::<f64>();
        mean_loss_sum += mean_loss_val;
        print!("{mean_loss_val: >6.2}\t");


        let best_loss_val = best.iter()
            .zip(&loss)
            .map(|(&ci, &li)| ci * li)
            .sum::<f64>();
        best_loss_sum += best_loss_val;
        println!("{best_loss_val: >6.2}");



        let z = x.iter()
            .zip(&loss)
            .map(|(&xi, &li)| xi - eta * li)
            .collect::<Vec<_>>();


        (x, c, mean) = approx_projection(
            x, c, z, oracle, &mut rng, alpha, delta, lambda, domain_diam,
        );

        let time = now.elapsed().as_millis();
        let line = format!("{time},{loss_val},{mean_loss_val},{best_loss_val}\n");
        lines.push(line);
    }
    println!("---------- OCO FINISH ----------\n");
    println!("\t* [PLAY] {play_loss_sum: >10.2}");
    println!("\t* [MEAN] {mean_loss_sum: >10.2}");
    println!("\t* [BEST] {best_loss_sum: >10.2}");
    (lines, cumulative_vec)
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
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
{
    assert!((0.0..=1.0).contains(&lambda));
    let uniform = Uniform::from(0.0..1.0);


    let mut mean = s.clone();
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
            return (x, s, mean);
        } else {
            mean = mean.iter()
                .zip(&t)
                .map(|(&si, &ti)| (1.0 - lambda) * si + lambda * ti)
                .collect::<Vec<_>>();
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
