use ndarray::Array3;
use std::collections::HashMap;

pub fn read_pdbtext(pdbstring: &str) -> Result<(Array3<f64>, Vec<String>), String> {
    let lines = pdbstring.lines();
    let mut coords: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut sequence: Vec<String> = Vec::new();
    let mut check: Vec<Vec<usize>> = Vec::new();

    let mut current_atoms: Option<Vec<[f64; 3]>> = None;
    let mut current_atom_check: Vec<usize> = Vec::new();

    let mut resid_old: Option<String> = None;
    let mut resname_old: Option<String> = None;

    let mut atomnum: HashMap<&str, usize> = HashMap::new();
    atomnum.insert(" N  ", 0);
    atomnum.insert(" CA ", 1);
    atomnum.insert(" C  ", 2);
    atomnum.insert(" O  ", 3);

    for line in lines {
        if line.starts_with("ATOM") {
            if line.len() < 54 { continue; }

            let atom_name = &line[12..16];
            let resname = &line[17..20];
            let resid = &line[21..26];

            let iatom = atomnum.get(atom_name);

            if let Some(ro) = &resid_old {
                if resid != ro {
                    // New residue
                    if let Some(atoms) = current_atoms {
                        coords.push(atoms);
                        sequence.push(resname_old.unwrap());
                        check.push(current_atom_check);
                    }
                    current_atoms = Some(Vec::new());
                    current_atom_check = Vec::new();
                    resid_old = Some(resid.to_string());
                    resname_old = Some(resname.to_string());
                }
            } else {
                // First residue
                current_atoms = Some(Vec::new());
                resid_old = Some(resid.to_string());
                resname_old = Some(resname.to_string());
            }

            if let Some(&idx) = iatom {
                let x = line[30..38].trim().parse::<f64>().map_err(|_| "Parse error x".to_string())?;
                let y = line[38..46].trim().parse::<f64>().map_err(|_| "Parse error y".to_string())?;
                let z = line[46..54].trim().parse::<f64>().map_err(|_| "Parse error z".to_string())?;

                if let Some(ref mut atoms) = current_atoms {
                    atoms.push([x, y, z]);
                    current_atom_check.push(idx);
                }
            }
        }
    }

    // Add last residue
    if let Some(atoms) = current_atoms {
        coords.push(atoms);
        sequence.push(resname_old.unwrap());
        check.push(current_atom_check);
    }

    if coords.is_empty() {
        return Err("No atoms found or empty PDB".to_string());
    }

    // Validation
    for (i, c) in check.iter().enumerate() {
        if c.len() != 4 {
             return Err(format!("Residue {} (index {}) has {} backbone atoms, expected 4 (N, CA, C, O)", i, sequence[i], c.len()));
        }
        if c != &[0, 1, 2, 3] {
             return Err(format!("Residue {} (index {}) atoms are not in N, CA, C, O order or duplicates exist: {:?}", i, sequence[i], c));
        }
    }

    // Convert to Array3
    let dim1 = coords.len();
    let dim2 = 4;
    let dim3 = 3;

    let mut flat_data = Vec::with_capacity(dim1 * dim2 * dim3);
    for residue_atoms in coords {
        for atom in residue_atoms {
            flat_data.push(atom[0]);
            flat_data.push(atom[1]);
            flat_data.push(atom[2]);
        }
    }

    let arr = Array3::from_shape_vec((dim1, dim2, dim3), flat_data)
        .map_err(|e| e.to_string())?;

    Ok((arr, sequence))
}
