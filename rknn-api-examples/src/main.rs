pub mod examples;
use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
enum CLIOptions {
    DynshapeInference(examples::dynshape_inference::Example),
}

impl CLIOptions {
    fn execute(&self) -> Result<()> {
        match self {
            Self::DynshapeInference(example) => example.execute(),
        }
    }
}

fn main() {
    let options = CLIOptions::parse();
    match options.execute() {
        Ok(_) => println!("Done!"),
        Err(_) => println!("Ooops!"),
    }
}
