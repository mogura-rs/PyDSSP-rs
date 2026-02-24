use clap::Parser;
use pydssp_rs::{assign, read_pdbtext};
use ndarray::{Array1, Axis};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input PDB files
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output result file (optional, defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    // Determine output: File or Stdout
    let mut writer: Box<dyn Write> = match args.output {
        Some(path) => {
            let file = fs::File::create(path).expect("Failed to create output file");
            Box::new(file)
        }
        None => Box::new(io::stdout()),
    };

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
                writeln!(writer, "> {}", filename).unwrap();
                writeln!(writer, "{}", sequence).unwrap();
                writeln!(writer, "{}", ss_string).unwrap();
            }
            Err(e) => {
                eprintln!("Failed to read {}: {}", filename, e);
            }
        }
    }
}
