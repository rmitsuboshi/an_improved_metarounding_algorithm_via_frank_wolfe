use argh::FromArgs;


/// runs a metarounding algorithm or the exact optimization
#[derive(FromArgs)]
pub struct Args {
    /// specify the running algorithm. soft, erlp, and lp are available.
    #[argh(positional, short='a')]
    pub algo: String,


    /// specify the output file name **without** extension.
    /// default is 
    /// `output_a[algo]_l[loss_seed]_p[prob_seed]_c[algo_seed]_n[nrounds].csv`.
    #[argh(positional, short='f', default="String::from(\"output\")")]
    pub file: String,


    /// specify the seed for generating the loss vectors.
    /// default is `1234`.
    #[argh(option, short='l', default="1234")]
    pub loss_seed: u64,


    /// specify the seed for generating set-cover instances.
    /// default is `5678`.
    #[argh(option, short='p', default="5678")]
    pub prob_seed: u64,


    /// specify the seed for algorithms.
    /// default is `777`.
    #[argh(option, short='c', default="777")]
    pub algo_seed: u64,


    /// specify the number of rounds.
    /// if `0`, it runs a metarounding,
    /// otherwise the program performs an online combinatorial optimization.
    #[argh(option, short='n', default="0")]
    pub nrounds: u64,
}


impl Args {
    pub fn output_name(&self) -> String {
        format!(
            "{file}_a{algo}_l{loss_seed}_p{prob_seed}_c{algo_seed}_\
            n{nrounds}.csv",
            file = self.file,
            algo = self.algo,
            loss_seed = self.loss_seed,
            prob_seed = self.prob_seed,
            algo_seed = self.algo_seed,
            nrounds = self.nrounds,
        )
    }
}
