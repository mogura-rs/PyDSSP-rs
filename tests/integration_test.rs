use ndarray::prelude::*;
use dssp_rs::{assign, read_pdbtext};
use std::fs;
use std::path::Path;

fn read_dssp_reference(path: &Path) -> Array1<i8> {
    let content = fs::read_to_string(path).expect("Failed to read DSSP file");
    let mut sw = false;
    let mut indices = Vec::new();

    for line in content.lines() {
        if line.contains('!') { continue; } // skip chain break
        if sw {
            if line.len() > 16 {
                let c = line.chars().nth(16).unwrap_or(' ');
                let idx = match c {
                    ' ' | 'S' | 'T' => 0,
                    'H' | 'G' | 'I' => 1,
                    'E' | 'B' => 2,
                    _ => 0, // Default to loop
                };
                indices.push(idx);
            }
        }
        if line.starts_with("  # ") {
            sw = true;
        }
    }
    Array1::from(indices)
}

#[test]
fn test_ts50_correlation() {
    let testset_dir = Path::new("tests/testset/TS50");
    let list_file = testset_dir.join("list");
    let list_content = fs::read_to_string(&list_file).expect("Failed to read list file");
    let targets: Vec<&str> = list_content.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

    let mut correlations = Vec::new();

    for target in &targets {
        let pdb_file = testset_dir.join("pdb").join(format!("{}.pdb", target));
        let dssp_file = testset_dir.join("dssp").join(format!("{}.dssp", target));

        // Read Reference
        let reference_idx = read_dssp_reference(&dssp_file);

        // Read PDB and Assign
        let pdb_content = fs::read_to_string(&pdb_file).expect("Failed to read PDB");
        let (coord, sequence) = read_pdbtext(&pdb_content);

        // Skip if empty (some failures?)
        if coord.len_of(Axis(0)) == 0 {
            println!("Skipping {} due to empty coords", target);
            continue;
        }

        // Create donor mask
        // donor_mask = sequence != 'P'
        // Python: sequence != 'PRO' (using 3 letter codes?).
        // My read_pdbtext returns 1-letter sequence.
        // So check 'P'.
        // sequence is String.
        let l = coord.len_of(Axis(0));
        let mut donor_mask = Array1::<f64>::ones(l);
        for (i, c) in sequence.chars().enumerate() {
             if c == 'P' {
                 donor_mask[i] = 0.0;
             }
        }

        let pydssp_result = assign(&coord.view(), Some(&donor_mask));

        // Convert one-hot (L, 3) to index (L)
        // result is (L, 3) with 0/1.
        // We want index 0, 1, 2.
        // argmax.
        let mut pydssp_idx = Array1::<i8>::zeros(l);
        for i in 0..l {
            // Priority? Loop=0, Helix=1, Strand=2.
            // If multiple set? Python assigns logic is:
            // loop = ~helix * ~strand.
            // So helix and strand are mutually exclusive?
            // Python: stack([loop, helix, strand]).
            // If helix is true -> 1. If strand is true -> 2.
            // If neither -> loop -> 0.
            // Can both helix and strand be true?
            // "strand = ladder". "helix = h3|h4|h5".
            // It seems possible? But unlikely in valid DSSP.
            // Usually argmax is fine.
            if pydssp_result[[i, 1]] == 1 {
                pydssp_idx[i] = 1;
            } else if pydssp_result[[i, 2]] == 1 {
                pydssp_idx[i] = 2;
            } else {
                pydssp_idx[i] = 0;
            }
        }

        // Compare
        // Need to align lengths?
        // Usually PDB and DSSP should match in length if no missing residues.
        // TS50 should be clean.
        // If lengths differ, we truncate to min length (Python code assumes match).
        let len = std::cmp::min(reference_idx.len(), pydssp_idx.len());
        if len == 0 { continue; }

        let mut matches = 0;
        for i in 0..len {
            if reference_idx[i] == pydssp_idx[i] {
                matches += 1;
            }
        }
        let correlation = matches as f64 / len as f64;
        correlations.push(correlation);
        // println!("Target: {}, Correlation: {}", target, correlation);
    }

    let mean_correlation: f64 = correlations.iter().sum::<f64>() / correlations.len() as f64;
    println!("Mean Correlation: {}", mean_correlation);
    assert!(mean_correlation > 0.97, "Correlation too low: {}", mean_correlation);
}
