# dssp-rs

A Rust implementation of the DSSP algorithm, ported from [PyDSSP](https://github.com/ShintaroMinami/PyDSSP).

This project provides a fast and efficient way to assign secondary structures to protein backbone coordinates using the DSSP algorithm. It utilizes the `ndarray` crate for numerical operations.

## Features

- **DSSP Algorithm:** Implements the core logic for hydrogen bond energy calculation and secondary structure assignment (Helix, Strand, Loop).
- **NDArray Backend:** Uses `ndarray` for efficient array manipulations.
- **CLI Tool:** Includes a command-line interface `dssp` for processing PDB files.
- **High Correlation:** Verified against the original DSSP implementation with >97% correlation on the TS50 dataset.

## Installation

Ensure you have Rust installed.

```bash
git clone https://github.com/mogura-rs/dssp-rs.git
cd dssp-rs
cargo build --release
```

The binary will be available at `target/release/dssp`.

## Usage

### Command Line Interface

You can run the `dssp` tool to process PDB files.

```bash
# Process a single file and output to stdout
cargo run --bin dssp -- input.pdb

# Process multiple files
cargo run --bin dssp -- input1.pdb input2.pdb

# Save output to a file
cargo run --bin dssp -- input.pdb -o output.result
```

### Library

Add `dssp-rs` to your `Cargo.toml`.

```toml
[dependencies]
dssp-rs = { git = "https://github.com/mogura-rs/dssp-rs" }
ndarray = "0.17.2"
```

```rust
use dssp_rs::{read_pdbtext, assign};
use ndarray::Axis;

fn main() {
    let pdb_content = std::fs::read_to_string("input.pdb").unwrap();
    let (coord, sequence) = read_pdbtext(&pdb_content);

    // coord is Array3 (L, 4, 3) [N, CA, C, O]
    // sequence is String (1-letter code)

    // Create donor mask (exclude Proline)
    let l = coord.len_of(Axis(0));
    let mut donor_mask = ndarray::Array1::<f64>::ones(l);
    for (i, c) in sequence.chars().enumerate() {
        if c == 'P' { donor_mask[i] = 0.0; }
    }

    // Assign secondary structure
    // Returns Array2 (L, 3) where [Loop, Helix, Strand] are 0/1
    let result = assign(&coord.view(), Some(&donor_mask));

    println!("{:?}", result);
}
```

## License

MIT
