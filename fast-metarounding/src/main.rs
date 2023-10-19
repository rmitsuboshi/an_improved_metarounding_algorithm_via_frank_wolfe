pub mod lp;
pub mod soft;
pub mod softv2;
pub mod erlp;
pub mod common;
pub mod ellipsoid;
pub mod metarounding;
pub mod approx_algorithm;

pub mod kkl09;
pub mod garber17;
pub mod online_framework;

use std::thread;
use std::fs::File;
use std::io::prelude::*;
use std::time::Instant;

use argh::FromArgs;

use common::instant_alpha;
use metarounding::Metarounding;
use lp::Lp;
use soft::*;
use softv2::*;
use erlp::Erlp;
use ellipsoid::Ellipsoid;
use kkl09::*;
use garber17::*;
use approx_algorithm::Oracle;
use online_framework::perform_oco;


/// runs a metarounding algorithm or the exact optimization
#[derive(FromArgs)]
struct Args {
    /// specify the running algorithm. soft, erlp, and lp are available.
    #[argh(positional, short='a')]
    algo: String,


    /// specify the output file. default is `output.csv`.
    #[argh(positional, short='f', default="String::from(\"output.csv\")")]
    file: String,


    /// specify the number of rounds.
    /// if `0`, it runs a metarounding,
    /// otherwise the program performs an online combinatorial optimization.
    #[argh(option, short='n', default="0")]
    nrounds: u64,
}


fn main() -> std::io::Result<()> {
    let args: Args = argh::from_env();


    if args.nrounds == 0 {
        println!("[METAROUNDING ONCE]");
        run_metarounding_once(args)
    } else {
        println!("[   PERFORM OCO   ]");
        run_oco(args)
    }
}


fn run_oco(args: Args) -> std::io::Result<()> {
    let header = "time[ms],loss,meanloss,bestval\n";
    let probability = 0.2;
    let n_items = 20;
    let n_sets = 100;
    let ratio = 0.2 * n_sets as f64;

    let seed_choose = 123;
    let seed_oracle = 456;
    let seed_loss   = 789;



    let linf_norm = 1.0;
    let tolerance = 0.01;

    let n_rounds = args.nrounds as usize;


    let oracle = Oracle::generate_setcover(
        n_items, n_sets, ratio, probability, seed_oracle
    );


    let (lines, _) = match args.algo.as_str() {
        "soft" => {
            let metarounder = Soft::new(linf_norm, tolerance, &oracle);
            perform_oco(seed_loss, seed_choose, n_rounds, &oracle, metarounder)
        },
        "softv2" => {
            let metarounder = SoftV2::new(linf_norm, tolerance, &oracle);
            perform_oco(seed_loss, seed_choose, n_rounds, &oracle, metarounder)
        },
        "erlp" => {
            let metarounder = Erlp::new(linf_norm, tolerance, &oracle);
            perform_oco(seed_loss, seed_choose, n_rounds, &oracle, metarounder)
        },
        "lp" => {
            let metarounder = Lp::new(tolerance, &oracle);
            perform_oco(seed_loss, seed_choose, n_rounds, &oracle, metarounder)
        },
        "ellipsoid" => {
            let alpha = oracle.alpha_bound();
            let metarounder = Ellipsoid::new(alpha, tolerance, &oracle);
            perform_oco(seed_loss, seed_choose, n_rounds, &oracle, metarounder)
        },
        "kkl09" => {
            let alpha = oracle.alpha_bound();
            kkl09(seed_loss, seed_choose, n_rounds, &oracle, alpha)
        },
        "garber17" => {
            let alpha = oracle.alpha_bound();
            garber17(seed_loss, seed_choose, n_rounds, &oracle, alpha)
        },
        _ => {
            panic!("{algo} is not available!", algo = args.algo);
        },
    };

    let mut file = File::create(args.file)?;
    file.write_all(header.as_bytes())?;

    for line in lines {
        file.write_all(line.as_bytes())?;
    }
    Ok(())
}


fn run_metarounding_once(args: Args) -> std::io::Result<()> {
    let header = "n_items,n_sets,alpha,time\n";
    // The number of combinatorial sets
    // let set_sizes  = [10, 50, 100, 200, 500, 1_000, 10_000];
    // let set_sizes = [10, 50, 100];
    let set_sizes = [20];
    // The number of items
    // let item_sizes = [10, 20, 100];
    // let item_sizes = [10, 20, 100, 1_000];
    let item_sizes = [100];
    let seed = 1234;
    // Probability
    let p = 0.2;
    // The maximal L_∞ norm of combinatorial vectors.
    let linf_norm = 1_f64;
    // Accuracy
    let tolerance = 0.01;

    let mut handles = Vec::new();
    for n_items in item_sizes {
        for n_sets in set_sizes {
            let r = 0.2 * n_sets as f64;

            let handle = match args.algo.as_str() {
                "soft" => {
                    thread::spawn(move || run_soft(
                        seed, n_items, n_sets, r, p, linf_norm, tolerance
                    ))
                },
                "softv2" => {
                    thread::spawn(move || run_softv2(
                        seed, n_items, n_sets, r, p, linf_norm, tolerance
                    ))
                },
                "erlp" => {
                    thread::spawn(move || run_erlp(
                        seed, n_items, n_sets, r, p, linf_norm, tolerance
                    ))
                },
                "lp" => {
                    thread::spawn(move || run_lp(
                            seed, n_items, n_sets, r, p, tolerance
                    ))
                },
                "ellipsoid" => {
                    thread::spawn(move || run_ellipsoid(
                            seed, n_items, n_sets, r, p, tolerance
                    ))
                },
                _ => { panic!("{algo} is not available!", algo = args.algo); },
            };
            handles.push(handle);
        }
    }

    let mut file = File::create(args.file)?;
    file.write_all(header.as_bytes())?;
    for handle in handles {
        let line = handle.join().unwrap();
        file.write_all(line.as_bytes())?;
    }
    Ok(())
}


fn run_soft(
    seed: u64,
    n_items: usize,
    n_sets: usize,
    r: f64,
    probability: f64,
    linf_norm: f64,
    tolerance: f64,
) -> String
{
    let oracle = Oracle::generate_setcover(
        n_items, n_sets, r, probability, seed
    );
    let x = oracle.initial_point(seed);


    let mut soft = Soft::new(linf_norm, tolerance, &oracle);


    let now = Instant::now();
    let (lambda, comb_vectors) = soft.round(&x);
    let time = now.elapsed().as_millis();
    let alpha = instant_alpha(&x, &lambda, &comb_vectors);
    format!("{n_items},{n_sets},{alpha},{time}\n")
}


fn run_softv2(
    seed: u64,
    n_items: usize,
    n_sets: usize,
    r: f64,
    probability: f64,
    linf_norm: f64,
    tolerance: f64,
) -> String
{
    let oracle = Oracle::generate_setcover(
        n_items, n_sets, r, probability, seed
    );
    let x = oracle.initial_point(seed);


    let mut soft = SoftV2::new(linf_norm, tolerance, &oracle);


    let now = Instant::now();
    let (lambda, comb_vectors) = soft.round(&x);
    let time = now.elapsed().as_millis();
    let alpha = instant_alpha(&x, &lambda, &comb_vectors);
    format!("{n_items},{n_sets},{alpha},{time}\n")
}


fn run_erlp(
    seed: u64,
    n_items: usize,
    n_sets: usize,
    r: f64,
    probability: f64,
    linf_norm: f64,
    tolerance: f64,
) -> String
{
    let oracle = Oracle::generate_setcover(
        n_items, n_sets, r, probability, seed
    );
    let x = oracle.initial_point(seed);


    let mut erlp = Erlp::new(linf_norm, tolerance, &oracle);


    let now = Instant::now();
    let (lambda, comb_vectors) = erlp.round(&x);
    let time = now.elapsed().as_millis();
    let alpha = instant_alpha(&x, &lambda, &comb_vectors);
    format!("{n_items},{n_sets},{alpha},{time}\n")
}


fn run_lp(
    seed: u64,
    n_items: usize,
    n_sets: usize,
    r: f64,
    probability: f64,
    tolerance: f64,
) -> String
{
    let oracle = Oracle::generate_setcover(
        n_items, n_sets, r, probability, seed
    );
    let x = oracle.initial_point(seed);


    let mut lp = Lp::new(tolerance, &oracle);


    let now = Instant::now();
    let (lambda, comb_vectors) = lp.round(&x);
    let time = now.elapsed().as_millis();
    let alpha = instant_alpha(&x, &lambda, &comb_vectors);
    format!("{n_items},{n_sets},{alpha},{time}\n")
}


fn run_ellipsoid(
    seed: u64,
    n_items: usize,
    n_sets: usize,
    r: f64,
    probability: f64,
    tolerance: f64,
) -> String
{
    let oracle = Oracle::generate_setcover(
        n_items, n_sets, r, probability, seed
    );
    let x = oracle.initial_point(seed);


    let alpha = oracle.alpha_bound();
    println!("[ALPHA] {alpha}");
    let mut ellipsoid = Ellipsoid::new(alpha, tolerance, &oracle);


    let now = Instant::now();
    let (lambda, comb_vectors) = ellipsoid.round(&x);
    let time = now.elapsed().as_millis();
    let alpha = instant_alpha(&x, &lambda, &comb_vectors);
    format!("{n_items},{n_sets},{alpha},{time}\n")
}
