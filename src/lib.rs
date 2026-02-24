use ndarray::prelude::*;
use ndarray::{Axis, Zip, s};
use std::collections::BTreeMap;

// Constants
pub const CONST_Q1Q2: f64 = 0.084_f64;
pub const CONST_F: f64 = 332.0_f64;
pub const DEFAULT_CUTOFF: f64 = -0.5_f64;
pub const DEFAULT_MARGIN: f64 = 1.0_f64;

// Helper: Norm last axis
fn norm_last_axis(a: &ArrayView3<f64>) -> Array2<f64> {
    a.map_axis(Axis(2), |v| v.dot(&v).sqrt())
}

// Helper: Norm vectors (2D)
fn norm_vectors(a: &ArrayView2<f64>) -> Array1<f64> {
    a.map_axis(Axis(1), |v| v.dot(&v).sqrt())
}

// Helper: Normalize
fn normalize(a: &Array2<f64>) -> Array2<f64> {
    let norms = norm_vectors(&a.view());
    let mut out = a.clone();
    Zip::from(out.rows_mut())
        .and(&norms)
        .for_each(|mut row, &n| {
            if n > 1e-6 {
                row /= n;
            }
        });
    out
}

pub fn get_hydrogen_atom_position(coord: &ArrayView3<f64>) -> Array2<f64> {
    let l = coord.len_of(Axis(0));
    let n_next = coord.slice(s![1..l, 0, ..]);
    let c_prev = coord.slice(s![0..l-1, 2, ..]);
    let vec_cn = &n_next - &c_prev;
    let ca_next = coord.slice(s![1..l, 1, ..]);
    let vec_can = &n_next - &ca_next;

    let vec_cn = normalize(&vec_cn);
    let vec_can = normalize(&vec_can);

    let vec_nh = &vec_cn + &vec_can;
    let vec_nh = normalize(&vec_nh);

    let h_pos = &n_next + &(vec_nh * 1.01_f64);
    h_pos
}

pub fn get_hbond_map(
    coord: &ArrayView3<f64>,
    donor_mask: Option<&Array1<f64>>,
    cutoff: f64,
    margin: f64,
) -> Array2<f64> {
    let l = coord.len_of(Axis(0));
    let h_coords = get_hydrogen_atom_position(coord);
    let n_coords = coord.slice(s![1..l, 0, ..]);
    let c_coords = coord.slice(s![0..l-1, 2, ..]);
    let o_coords = coord.slice(s![0..l-1, 3, ..]);

    // Python broadcasting:
    // nmap = repeat(..., '... m c -> ... m n c', n=l-1) -> varies along rows (m) -> (N)
    // omap = repeat(..., '... n c -> ... m n c', m=l-1) -> varies along cols (n) -> (O)
    // result (N, O)

    let n_broad = n_coords.insert_axis(Axis(1)); // (L-1, 1, 3) -> Rows
    let h_broad = h_coords.view().insert_axis(Axis(1));

    let c_broad = c_coords.insert_axis(Axis(0)); // (1, L-1, 3) -> Cols
    let o_broad = o_coords.insert_axis(Axis(0));

    // d_on = norm(omap - nmap) -> norm(Col(O) - Row(N))
    let d_on = norm_last_axis(&(&o_broad - &n_broad).view());
    let d_ch = norm_last_axis(&(&c_broad - &h_broad).view());
    let d_oh = norm_last_axis(&(&o_broad - &h_broad).view());
    let d_cn = norm_last_axis(&(&c_broad - &n_broad).view());

    let mut e = Array2::<f64>::zeros(d_on.raw_dim());
    Zip::from(&mut e)
        .and(&d_on)
        .and(&d_ch)
        .and(&d_oh)
        .and(&d_cn)
        .for_each(|val, &don, &dch, &doh, &dcn| {
            *val = CONST_Q1Q2 * (1./don + 1./dch - 1./doh - 1./dcn) * CONST_F;
        });

    let mut e_padded = Array2::<f64>::zeros((l, l));
    e_padded.slice_mut(s![1..l, 0..l-1]).assign(&e);

    let mut local_mask = Array2::<f64>::ones((l, l));
    for i in 0..l {
        local_mask[[i, i]] = 0.0;
        if i > 0 { local_mask[[i, i-1]] = 0.0; }
        if i > 1 { local_mask[[i, i-2]] = 0.0; }
    }

    let d_mask = if let Some(dm) = donor_mask {
        // Mask Donors (Rows). dm is (L). Want (L, 1).
        dm.view().insert_axis(Axis(1)).broadcast((l, l)).unwrap().to_owned()
    } else {
        Array2::ones((l, l))
    };

    let mut hbond_map = e_padded;
    hbond_map.mapv_inplace(|v| {
        let val = cutoff - margin - v;
        let clipped = val.max(-margin).min(margin);
        ( (clipped / margin * std::f64::consts::FRAC_PI_2).sin() + 1.0_f64 ) / 2.0
    });

    hbond_map = hbond_map * local_mask * d_mask;
    hbond_map
}

// Helper: Diagonals
fn diagonal(a: &ArrayView2<f64>, offset: isize) -> Array1<bool> {
    let (rows, cols) = a.dim();
    let len = if offset >= 0 { rows as isize - offset } else { rows as isize + offset };
    if len <= 0 { return Array1::from_elem(0, false); }

    let mut res = Vec::with_capacity(len as usize);
    for k in 0..len {
        let r = ((if offset >= 0 { 0 } else { -offset }) + k) as usize;
        let c = ((if offset >= 0 { offset } else { 0 }) + k) as usize;
        if r < rows && c < cols {
             res.push(a[[r, c]] > 0.0);
        }
    }
    Array1::from(res)
}

pub fn assign(
    coord: &ArrayView3<f64>,
    donor_mask: Option<&Array1<f64>>,
) -> Array2<i8> {
    let l = coord.len_of(Axis(0));
    let hbmap = get_hbond_map(coord, donor_mask, DEFAULT_CUTOFF, DEFAULT_MARGIN);

    let hbmap_t = hbmap.t(); // View (N, O)

    // Turn 3, 4, 5
    let turn3 = diagonal(&hbmap_t, 3);
    let turn4 = diagonal(&hbmap_t, 4);
    let turn5 = diagonal(&hbmap_t, 5);

    // Helper to calculate helix pattern
    let calc_helix = |turn: &Array1<bool>, _offset: usize| -> Array1<bool> {
        let len = turn.len();
        if len < 2 { return Array1::from_elem(l, false); }
        let t_prev = turn.slice(s![0..len-1]);
        let t_next = turn.slice(s![1..len]);
        let product = &t_prev & &t_next;

        let mut out = Array1::from_elem(l, false);
        let p_len = product.len();
        if 1 + p_len <= l {
            out.slice_mut(s![1..1+p_len]).assign(&product);
        }
        out
    };

    let h3 = calc_helix(&turn3, 3);
    let h4 = calc_helix(&turn4, 4);
    let h5 = calc_helix(&turn5, 5);

    let roll_1d = |a: &Array1<bool>, shift: usize| -> Array1<bool> {
        let mut out = Array1::from_elem(a.len(), false);
        for i in 0..a.len() {
            if i >= shift {
                out[i] = a[i - shift];
            }
        }
        out
    };

    let h4_roll1 = roll_1d(&h4, 1);
    let h4_roll2 = roll_1d(&h4, 2);
    let h4_roll3 = roll_1d(&h4, 3);
    let helix4 = &h4 | &h4_roll1 | &h4_roll2 | &h4_roll3;

    let roll_left_1d = |a: &Array1<bool>, shift: usize| -> Array1<bool> {
        let mut out = Array1::from_elem(a.len(), false);
        for i in 0..a.len() {
            if i + shift < a.len() {
                out[i] = a[i + shift];
            }
        }
        out
    };

    let helix4_neg = helix4.mapv(|x| !x);
    let helix4_left = roll_left_1d(&helix4, 1);
    let helix4_left_neg = helix4_left.mapv(|x| !x);

    let h3 = &h3 & &helix4_left_neg & &helix4_neg;
    let h5 = &h5 & &helix4_left_neg & &helix4_neg;

    let h3_roll1 = roll_1d(&h3, 1);
    let h3_roll2 = roll_1d(&h3, 2);
    let helix3 = &h3 | &h3_roll1 | &h3_roll2;

    let h5_roll1 = roll_1d(&h5, 1);
    let h5_roll2 = roll_1d(&h5, 2);
    let h5_roll3 = roll_1d(&h5, 3);
    let h5_roll4 = roll_1d(&h5, 4);
    let helix5 = &h5 | &h5_roll1 | &h5_roll2 | &h5_roll3 | &h5_roll4;

    // Bridge
    let mut p_bridge = Array2::from_elem((l, l), false);
    let mut a_bridge = Array2::from_elem((l, l), false);

    for i in 0..l.saturating_sub(2) {
        for j in 0..l.saturating_sub(2) {
            let val1 = hbmap_t[[i, j+1]] > 0.0 && hbmap_t[[j+1, i+2]] > 0.0;
            let val2 = hbmap_t[[j, i+1]] > 0.0 && hbmap_t[[i+1, j+2]] > 0.0;

            if val1 || val2 {
                p_bridge[[i+1, j+1]] = true;
            }

            let val3 = hbmap_t[[i+1, j+1]] > 0.0 && hbmap_t[[j+1, i+1]] > 0.0;
            let val4 = hbmap_t[[i, j+2]] > 0.0 && hbmap_t[[j, i+2]] > 0.0;

            if val3 || val4 {
                a_bridge[[i+1, j+1]] = true;
            }
        }
    }

    // Ladder
    let mut ladder = Array1::from_elem(l, false);
    for i in 0..l {
        let mut row_sum = 0;
        for j in 0..l {
            if p_bridge[[i, j]] || a_bridge[[i, j]] {
                row_sum += 1;
            }
        }
        if row_sum > 0 {
            ladder[i] = true;
        }
    }

    let helix = &helix3 | &helix4 | &helix5;
    let strand = ladder;

    let loop_ = helix.mapv(|x| !x) & strand.mapv(|x| !x);

    // Result: 0=Loop, 1=Helix, 2=Strand
    let mut result = Array2::from_elem((l, 3), 0i8);
    for i in 0..l {
        result[[i, 0]] = if loop_[i] { 1 } else { 0 };
        result[[i, 1]] = if helix[i] { 1 } else { 0 };
        result[[i, 2]] = if strand[i] { 1 } else { 0 };
    }

    result
}

pub fn read_pdbtext(text: &str) -> (Array3<f64>, String) {
    let mut coords = Vec::new();
    let mut indices = Vec::new();
    let mut sequence = String::new();

    fn res_name_to_char(n: &str) -> char {
        match n {
            "ALA" => 'A', "CYS" => 'C', "ASP" => 'D', "GLU" => 'E',
            "PHE" => 'F', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
            "LYS" => 'K', "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
            "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R', "SER" => 'S',
            "THR" => 'T', "VAL" => 'V', "TRP" => 'W', "TYR" => 'Y',
            _ => 'X',
        }
    }

    let mut current_res_id = String::new();
    let mut current_res: BTreeMap<String, [f64; 3]> = BTreeMap::new();
    let mut current_res_name = String::new();

    for line in text.lines() {
        if line.starts_with("ATOM") {
            if line.len() < 54 { continue; }
            let atom_name = line[12..16].trim();
            if !["N", "CA", "C", "O"].contains(&atom_name) { continue; }

            // Residue ID includes sequence number and insertion code (cols 22-26 and 26)
            // 0-based: 22..26 is num, 26 is ins.
            // Actually PDB spec: 23-26 is ResSeq, 27 is iCode.
            // 0-based: 22..26 (4 chars), 26 (1 char).
            // Let's take 22..27.
            let res_id_full = line[22..27].to_string();

            if res_id_full != current_res_id {
                if !current_res.is_empty() {
                    let n = *current_res.get("N").unwrap_or(&[0.0,0.0,0.0]);
                    let ca = *current_res.get("CA").unwrap_or(&[0.0,0.0,0.0]);
                    let c = *current_res.get("C").unwrap_or(&[0.0,0.0,0.0]);
                    let o = *current_res.get("O").unwrap_or(&[0.0,0.0,0.0]);
                    coords.push(n[0]); coords.push(n[1]); coords.push(n[2]);
                    coords.push(ca[0]); coords.push(ca[1]); coords.push(ca[2]);
                    coords.push(c[0]); coords.push(c[1]); coords.push(c[2]);
                    coords.push(o[0]); coords.push(o[1]); coords.push(o[2]);
                    sequence.push(res_name_to_char(&current_res_name));
                    indices.push(current_res_id.clone());
                }
                current_res.clear();
                current_res_id = res_id_full;
                current_res_name = line[17..20].trim().to_string();
            }

            let x = line[30..38].trim().parse::<f64>().unwrap_or(0.0);
            let y = line[38..46].trim().parse::<f64>().unwrap_or(0.0);
            let z = line[46..54].trim().parse::<f64>().unwrap_or(0.0);

            // Only insert if not exists (handle alt locs: keep first/A)
            current_res.entry(atom_name.to_string()).or_insert([x, y, z]);
        }
    }
    if !current_res.is_empty() {
        let n = *current_res.get("N").unwrap_or(&[0.0,0.0,0.0]);
        let ca = *current_res.get("CA").unwrap_or(&[0.0,0.0,0.0]);
        let c = *current_res.get("C").unwrap_or(&[0.0,0.0,0.0]);
        let o = *current_res.get("O").unwrap_or(&[0.0,0.0,0.0]);
        coords.push(n[0]); coords.push(n[1]); coords.push(n[2]);
        coords.push(ca[0]); coords.push(ca[1]); coords.push(ca[2]);
        coords.push(c[0]); coords.push(c[1]); coords.push(c[2]);
        coords.push(o[0]); coords.push(o[1]); coords.push(o[2]);
        sequence.push(res_name_to_char(&current_res_name));
        indices.push(current_res_id);
    }

    let l = indices.len();
    let arr = if l > 0 {
        Array3::from_shape_vec((l, 4, 3), coords).unwrap()
    } else {
        Array3::zeros((0, 4, 3))
    };
    (arr, sequence)
}
