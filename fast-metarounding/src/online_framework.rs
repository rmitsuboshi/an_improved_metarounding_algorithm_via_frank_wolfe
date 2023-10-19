use std::time::Instant;

use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::common::*;
use crate::metarounding::Metarounding;
use crate::approx_algorithm::Oracle;


/// Performs Online Combinatorial Linear Optimization
/// by Online Gradient Descent + Metarounding.
pub fn perform_oco<M>(
    seed_loss: u64,
    seed_choose: u64,
    n_rounds: usize,
    oracle: &Oracle,
    mut metarounder: M,
) -> (Vec<String>, Vec<f64>)
    where M: Metarounding
{
    let mut rng = StdRng::seed_from_u64(seed_choose);
    let uniform = Uniform::from(0.0..1.0);

    let (_, n_sets) = oracle.shape();

    // ********************************************
    // Get the loss vector defined before the game.
    let losses = build_losses(n_sets, n_rounds, seed_loss);


    // ***********************************
    // Initial prediction over `relax(C)`.
    let mut x = vec![1.0; n_sets];
    let mut prediction = x.clone();
    let mut mean = x.clone();


    // ******************************************
    // Lipschitz constant and diameter of domain.
    let lipz = (n_sets as f64).sqrt();
    let diam = (n_sets as f64).sqrt();


    let mut lines = Vec::with_capacity(n_rounds);
    let mut cumulative_vec = vec![0.0; n_sets];

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
    // let best_loss_sum = best.iter()
    //     .zip(loss_sum)
    //     .map(|(&a, b)| a * b)
    //     .sum::<f64>();

    println!("---------- OCO STARTS ----------");
    println!("{: >4}\t{: >6}\t{: >6}\t{: >6}", "ROUND", "PLAY", "MEAN", "BEST");
    for (round, loss) in (1..=n_rounds).zip(losses) {
        let now = Instant::now();
        // // **********************************
        // // Get a prediction via Metarounding.
        // assert!(x.iter().any(|&xi| xi > 0.0));
        // let (dist, combs) = metarounder.round(&x);
        // let r = uniform.sample(&mut rng);
        // let ix = choose(&dist, r);
        // let prediction = &combs[ix];


        // *****************************************
        // Update the cumulative loss vector & value
        cumulative_vec.iter_mut()
            .zip(&loss)
            .for_each(|(cl, &l)| { *cl += l; });

        let loss_val = loss.iter()
            .zip(&prediction)
            .map(|(&l, &c)| l * c)
            .sum::<f64>();
        play_loss_sum += loss_val;


        // let mut mean = vec![0.0; n_sets];
        // for (&d, comb) in dist.iter().zip(&combs) {
        //     assert_eq!(mean.len(), comb.len());
        //     mean.iter_mut()
        //         .zip(comb)
        //         .for_each(|(mi, ci)| { *mi += d * ci; });
        // }
        let mean_loss_val = loss.iter().copied()
            .zip(mean)
            .map(|(l, m)| l * m)
            .sum::<f64>();
        mean_loss_sum += mean_loss_val;


        let best_loss_val = best.iter()
            .zip(&loss)
            .map(|(&ci, &li)| ci * li)
            .sum::<f64>();
        best_loss_sum += best_loss_val;
        println!(
            "{r: >4}\t{l: >6.2}\t{m: >6.2}\t{b: >6.2}",
            r = round,
            l = loss_val,
            m = mean_loss_val,
            b = best_loss_val,
        );


        // **********************
        // Update `x ∈ relax(C)` by OGD.
        // ```txt
        // At round t,
        //      1. y_t ← x_t - (D / (G √t)) ∇f_t (x_t)
        //      2. x_{t+1} ← Project(y_t)
        // ```
        // Here, the projection is the standard one (L2 projection).
        // That is, the optimal solution of:
        // ```txt
        //      Minimize 0.5 * || y - x ||^2
        //      sub. to  x ∈ relax(C)
        // ```
        let eta = diam / (lipz * (round as f64).sqrt());
        let y = x.iter()
            .zip(&loss)
            .map(|(&xi, &li)| xi - eta * li)
            .collect::<Vec<_>>();
        x = oracle.l2_projection(&y);


        // **********************************
        // Get a prediction via Metarounding.
        assert!(x.iter().any(|&xi| xi > 0.0));
        let (dist, combs) = metarounder.round(&x);
        let r = uniform.sample(&mut rng);
        let ix = choose(&dist, r);
        prediction = combs[ix].clone();
        mean = vec![0.0; n_sets];
        for (&d, comb) in dist.iter().zip(&combs) {
            assert_eq!(mean.len(), comb.len());
            mean.iter_mut()
                .zip(comb)
                .for_each(|(mi, ci)| { *mi += d * ci; });
        }

        assert!(x.iter().all(|xi| (0.0..=1.0).contains(xi)));


        let time = now.elapsed().as_millis();
        lines.push(
            format!("{time},{loss_val},{mean_loss_val},{best_loss_val}\n")
        );
    }
    println!("---------- OCO FINISH ----------\n");
    println!("\t* [PLAY] {play_loss_sum: >10.2}");
    println!("\t* [MEAN] {mean_loss_sum: >10.2}");
    println!("\t* [BEST] {best_loss_sum: >10.2}");
    (lines, cumulative_vec)
}

