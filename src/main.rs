use clap::Parser;
use pydssp_rs::{assign, read_pdbtext};
use ndarray::{Array1, Axis};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input PDB files
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output result file
    #[arg(short, long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    // Check if input is only one and it is a list of PDB files?
    // But clap handles Vec<PathBuf>.
    // User requested: `dssp input_01.pdb input_02.pdb ... -o output.result`

    let mut outfile = fs::File::create(&args.output).expect("Failed to create output file");

    for input_path in args.input {
        let filename = input_path.to_string_lossy().to_string();
        match fs::read_to_string(&input_path) {
            Ok(content) => {
                let (coord, sequence) = read_pdbtext(&content);

                if coord.len_of(Axis(0)) == 0 {
                    eprintln!("Warning: {} is empty or invalid PDB", filename);
                    continue;
                }

                // Create donor mask (Proline check)
                let l = coord.len_of(Axis(0));
                let mut donor_mask = Array1::<f64>::ones(l);
                for (i, c) in sequence.chars().enumerate() {
                    if c == 'P' {
                        donor_mask[i] = 0.0;
                    }
                }

                let result = assign(&coord.view(), Some(&donor_mask));

                let mut ss_string = String::with_capacity(l);
                for i in 0..l {
                    if result[[i, 1]] == 1 {
                        ss_string.push('H');
                    } else if result[[i, 2]] == 1 {
                        ss_string.push('E');
                    } else {
                        ss_string.push('-');
                    }
                }

                // Format: > filename \n sequence \n ss_string
                writeln!(outfile, "> {}", filename).unwrap();
                writeln!(outfile, "{}", sequence).unwrap();
                writeln!(outfile, "{}", ss_string).unwrap();
            }
            Err(e) => {
                eprintln!("Failed to read {}: {}", filename, e);
            }
        }
    }
}
