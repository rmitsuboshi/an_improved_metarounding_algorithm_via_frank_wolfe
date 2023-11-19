use std::time::Instant;
use std::fs::File;
use std::io::prelude::*;

use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::dispatch::*;
use crate::common::*;
use crate::metarounding::Metarounding;
use crate::approx_algorithm::Oracle;


/// Performs Online Combinatorial Linear Optimization
/// by Online Gradient Descent + Metarounding.
pub fn perform_oco<M>(
    args: &Args,
    n_rounds: usize,
    oracle: &Oracle,
    mut metarounder: M,
)
    where M: Metarounding
{
    let mut file = File::create(args.output_name())
        .expect("Failed to create file");
    file.write_all(HEADER.as_bytes())
        .expect("Failed to write header");


    let mut rng = StdRng::seed_from_u64(args.algo_seed);
    let uniform = Uniform::from(0.0..1.0);

    let (_, n_sets) = oracle.shape();

    let seed = args.loss_seed;
    let (loss_sum, loss_iter) = build_losses(n_sets, n_rounds, seed);


    // ***********************************
    // Initial prediction over `relax(C)`.
    let mut x = vec![1.0; n_sets];
    let mut prediction = x.clone();


    // ******************************************
    // Lipschitz constant and diameter of domain.
    let lipz = (n_sets as f64).sqrt();
    let diam = (n_sets as f64).sqrt();


    let best = oracle.best_action_hindsight(&loss_sum);
    let mut play_loss_sum = 0.0;
    let mut best_loss_sum = 0.0;

    println!("---------- OCO STARTS ----------");
    println!("{: >5}\t{: >6}\t{: >6}", "ROUND", "PLAY", "BEST");
    for (round, loss) in (1..=n_rounds).zip(loss_iter) {
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

        let loss_val = iproduct(&loss, &prediction);
        play_loss_sum += loss_val;


        let best_loss_val = iproduct(&best, &loss);
        best_loss_sum += best_loss_val;
        println!("{round: >5}\t{loss_val: >6.2}\t{best_loss_val: >6.2}");


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

        assert!(x.iter().all(|xi| (0.0..=1.0).contains(xi)));


        let time = now.elapsed().as_millis();
        let line = format!("{time},{loss_val},{best_loss_val}\n");
        file.write_all(line.as_bytes())
            .expect("Failed to write log");
    }
    println!("---------- OCO FINISH ----------\n");
    println!("\t* [PLAY] {play_loss_sum: >10.2}");
    println!("\t* [BEST] {best_loss_sum: >10.2}");
}

