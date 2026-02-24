use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};

const CONST_Q1Q2: f64 = 0.084;
const CONST_F: f64 = 332.0;
const DEFAULT_CUTOFF: f64 = -0.5;
const DEFAULT_MARGIN: f64 = 1.0;

fn norm(v: &ArrayView2<f64>) -> Array2<f64> {
    let sq = v.mapv(|x| x * x).sum_axis(Axis(1));
    sq.mapv(f64::sqrt).insert_axis(Axis(1))
}

fn normalize(v: &Array2<f64>) -> Array2<f64> {
    let n = norm(&v.view());
    v / &n
}

pub fn get_hydrogen_atom_position(coord: &Array3<f64>) -> Array2<f64> {
    let l = coord.len_of(Axis(0));
    let n_curr = coord.slice(s![1.., 0, ..]);
    let c_prev = coord.slice(s![..l-1, 2, ..]);
    let ca_curr = coord.slice(s![1.., 1, ..]);

    let vec_cn = &n_curr - &c_prev;
    let vec_cn_n = normalize(&vec_cn.to_owned());

    let vec_can = &n_curr - &ca_curr;
    let vec_can_n = normalize(&vec_can.to_owned());

    let vec_nh = &vec_cn_n + &vec_can_n;
    let vec_nh_n = normalize(&vec_nh);

    &n_curr + &(1.01 * &vec_nh_n)
}

pub fn get_hbond_map(
    coord: &Array3<f64>,
    donor_mask: Option<&Array1<f64>>,
) -> Array2<f64> {
    let l = coord.len_of(Axis(0));
    let h_pos = get_hydrogen_atom_position(coord);

    // (Donor, Acceptor) = (N, C/O) matrix.
    let n_pts = coord.slice(s![1.., 0, ..]);
    let h_pts = h_pos.view();
    let c_pts = coord.slice(s![..l-1, 2, ..]);
    let o_pts = coord.slice(s![..l-1, 3, ..]);

    let pairwise_dist = |p1: ArrayView2<f64>, p2: ArrayView2<f64>| -> Array2<f64> {
        let p1_b = p1.insert_axis(Axis(1));
        let p2_b = p2.insert_axis(Axis(0));
        let diff = &p1_b - &p2_b;
        diff.mapv(|x| x*x).sum_axis(Axis(2)).mapv(f64::sqrt)
    };

    let d_on = pairwise_dist(n_pts, o_pts);
    let d_ch = pairwise_dist(h_pts, c_pts);
    let d_oh = pairwise_dist(h_pts, o_pts);
    let d_cn = pairwise_dist(n_pts, c_pts);

    let term1 = d_on.mapv(|x| 1.0/x);
    let term2 = d_ch.mapv(|x| 1.0/x);
    let term3 = d_oh.mapv(|x| 1.0/x);
    let term4 = d_cn.mapv(|x| 1.0/x);

    let e_mat = (term1 + term2 - term3 - term4) * (CONST_Q1Q2 * CONST_F);

    let mut e_final = Array2::<f64>::zeros((l, l));
    e_final.slice_mut(s![1.., ..l-1]).assign(&e_mat);

    let cutoff = DEFAULT_CUTOFF;
    let margin = DEFAULT_MARGIN;

    let mut hbond_map = e_final.mapv(|e| {
        let val = cutoff - margin - e;
        let clipped = val.max(-margin).min(margin);
        (f64::sin(clipped / margin * std::f64::consts::PI / 2.0) + 1.0) / 2.0
    });

    for i in 0..l {
        for j in 0..l {
            if i == j || i == j + 1 || i == j + 2 {
                hbond_map[[i, j]] = 0.0;
            }
        }
    }

    if let Some(mask) = donor_mask {
        hbond_map = hbond_map * mask.slice(s![..]).insert_axis(Axis(1));
    }

    hbond_map
}

pub fn assign(
    coord: &Array3<f64>,
    donor_mask: Option<&Array1<f64>>,
) -> Array1<usize> {
    let l = coord.len_of(Axis(0));
    // 1. Get hbond_map (N, C)
    let hbmap_nc = get_hbond_map(coord, donor_mask);

    // 2. Transpose to (C, N) -> "i:C=O, j:N-H"
    let hbmap = hbmap_nc.t().to_owned(); // (L, L)

    // 3. Turns
    // turn3[i] = hbmap[i, i+3] > 0
    let mut turn3 = Array1::<bool>::from_elem(l.saturating_sub(3), false);
    for i in 0..turn3.len() {
        if hbmap[[i, i+3]] > 0.0 { turn3[i] = true; }
    }
    let mut turn4 = Array1::<bool>::from_elem(l.saturating_sub(4), false);
    for i in 0..turn4.len() {
        if hbmap[[i, i+4]] > 0.0 { turn4[i] = true; }
    }
    let mut turn5 = Array1::<bool>::from_elem(l.saturating_sub(5), false);
    for i in 0..turn5.len() {
        if hbmap[[i, i+5]] > 0.0 { turn5[i] = true; }
    }

    // 4. Helices
    // h3 = turn3[i] * turn3[i+1] (length L-4)
    // Padded to L: [1, 3] -> 1 before, 3 after.
    let mut h3 = Array1::<bool>::from_elem(l, false);
    if turn3.len() >= 2 {
        for i in 0..turn3.len()-1 {
            if turn3[i] && turn3[i+1] {
                h3[i+1] = true; // Offset 1 because of padding [1, 3]
            }
        }
    }

    let mut h4 = Array1::<bool>::from_elem(l, false);
    if turn4.len() >= 2 {
        for i in 0..turn4.len()-1 {
            if turn4[i] && turn4[i+1] {
                h4[i+1] = true; // Offset 1 because of padding [1, 4]
            }
        }
    }

    let mut h5 = Array1::<bool>::from_elem(l, false);
    if turn5.len() >= 2 {
        for i in 0..turn5.len()-1 {
            if turn5[i] && turn5[i+1] {
                h5[i+1] = true; // Offset 1 because of padding [1, 5]
            }
        }
    }

    // helix4 = h4 + roll(h4, 1) + roll(h4, 2) + roll(h4, 3)
    let roll_or = |arr: &Array1<bool>, shift: usize| -> Array1<bool> {
        let mut res = Array1::from_elem(arr.len(), false);
        for i in 0..arr.len() {
            if i >= shift && arr[i - shift] {
                res[i] = true;
            }
        }
        res
    };

    let h4_roll1 = roll_or(&h4, 1);
    let h4_roll2 = roll_or(&h4, 2);
    let h4_roll3 = roll_or(&h4, 3);

    let helix4 = h4.mapv(|x| x) | &h4_roll1 | &h4_roll2 | &h4_roll3;

    // h3 = h3 * ~roll(helix4, -1) * ~helix4
    // roll(helix4, -1) -> shift left.
    let roll_left = |arr: &Array1<bool>, shift: usize| -> Array1<bool> {
        let mut res = Array1::from_elem(arr.len(), false);
        for i in 0..arr.len() {
            if i + shift < arr.len() && arr[i + shift] {
                res[i] = true;
            }
        }
        res
    };
    let helix4_left = roll_left(&helix4, 1);

    // Update h3
    for i in 0..l {
        if helix4[i] || helix4_left[i] {
            h3[i] = false;
        }
    }

    // Update h5
    for i in 0..l {
        if helix4[i] || helix4_left[i] {
            h5[i] = false;
        }
    }

    // helix3 = h3 + roll(h3, 1) + roll(h3, 2)
    let h3_roll1 = roll_or(&h3, 1);
    let h3_roll2 = roll_or(&h3, 2);
    let helix3 = &h3 | &h3_roll1 | &h3_roll2;

    // helix5 = h5 + roll(h5, 1..4)
    let h5_roll1 = roll_or(&h5, 1);
    let h5_roll2 = roll_or(&h5, 2);
    let h5_roll3 = roll_or(&h5, 3);
    let h5_roll4 = roll_or(&h5, 4);
    let helix5 = &h5 | &h5_roll1 | &h5_roll2 | &h5_roll3 | &h5_roll4;

    // 5. Bridges
    // Manual loop implementation for p_bridge and a_bridge
    // p_bridge padded [1, 1], [1, 1] for i, j?
    // Result vector? No, ladder is vector.
    // p_bridge is matrix (L, L).
    // Loop valid i, j from 1 to L-2 (to allow +/- 1 window).

    let mut p_bridge = Array2::<bool>::from_elem((l, l), false);
    let mut a_bridge = Array2::<bool>::from_elem((l, l), false);

    for i in 1..l-1 {
        for j in 1..l-1 {
            // Window start r = i-1, c = j-1.
            // unfoldmap[r, c, u, v] = hbmap[r+u, c+v]
            // unfoldmap_rev[r, c, u, v] = hbmap[c+u, r+v]

            // p_bridge term:
            // (unfoldmap[i-1, j-1, 0, 1] * unfoldmap_rev[i-1, j-1, 1, 2]) +
            // (unfoldmap_rev[i-1, j-1, 0, 1] * unfoldmap[i-1, j-1, 1, 2])

            // map[0, 1] -> offset u=0, v=1 -> hbmap[r, c+1] -> hbmap[i-1, j]
            // map[1, 2] -> offset u=1, v=2 -> hbmap[r+1, c+2] -> hbmap[i, j+1]

            // rev[1, 2] -> hbmap[c+1, r+2] -> hbmap[j, i+1]
            // rev[0, 1] -> hbmap[c, r+1] -> hbmap[j-1, i]

            // Term 1: hbmap[i-1, j] && hbmap[j, i+1]
            let term1 = hbmap[[i-1, j]] > 0.0 && hbmap[[j, i+1]] > 0.0;

            // Term 2: hbmap[j-1, i] && hbmap[i, j+1]
            let term2 = hbmap[[j-1, i]] > 0.0 && hbmap[[i, j+1]] > 0.0;

            if term1 || term2 {
                p_bridge[[i, j]] = true;
            }

            // a_bridge term:
            // (unfoldmap[1, 1] * unfoldmap_rev[1, 1]) + (unfoldmap[0, 2] * unfoldmap_rev[0, 2])
            // map[1, 1] -> hbmap[i, j]
            // rev[1, 1] -> hbmap[j, i]

            // map[0, 2] -> hbmap[i-1, j+1]
            // rev[0, 2] -> hbmap[j-1, i+1]

            let term_a1 = hbmap[[i, j]] > 0.0 && hbmap[[j, i]] > 0.0;
            let term_a2 = hbmap[[i-1, j+1]] > 0.0 && hbmap[[j-1, i+1]] > 0.0;

            if term_a1 || term_a2 {
                a_bridge[[i, j]] = true;
            }
        }
    }

    // Ladder
    // ladder = (p_bridge + a_bridge).sum(-1) > 0
    let mut ladder = Array1::<bool>::from_elem(l, false);
    for i in 0..l {
        for j in 0..l {
            if p_bridge[[i, j]] || a_bridge[[i, j]] {
                ladder[i] = true;
                break;
            }
        }
    }

    // 6. Assignment
    // helix = helix3 + helix4 + helix5
    let helix = &helix3 | &helix4 | &helix5;
    let strand = ladder; // ladder is bool array

    // Loop is implicit.
    // OneHot: Loop, Helix, Strand.
    // Index: 0, 1, 2.
    // Priority?
    // Python: onehot = np.stack([loop, helix, strand], axis=-1)
    // If both helix and strand?
    // Python returns onehot. `assign` returns `argmax`.
    // If both true, argmax returns first index? No, last index?
    // numpy argmax returns first occurrence of max value.
    // Order: Loop (0), Helix (1), Strand (2).
    // If Strand is true (1), and Helix is true (1). Argmax is 1?
    // Wait, indices: 0, 1, 2.
    // If Loop=1, Helix=0, Strand=0 -> 0.
    // If Loop=0, Helix=1, Strand=0 -> 1.
    // If Loop=0, Helix=0, Strand=1 -> 2.
    // If Loop=0, Helix=1, Strand=1 -> 1. (Since 1 comes before 2).
    // But `loop = (~helix * ~strand)`.
    // So if Helix or Strand is true, Loop is false.
    // So checks: Helix? Strand?
    // If both, Helix wins (index 1 < 2).

    let mut result = Array1::<usize>::zeros(l);
    for i in 0..l {
        if helix[i] {
            result[i] = 1; // Helix (H)
        } else if strand[i] {
            result[i] = 2; // Strand (E)
        } else {
            result[i] = 0; // Loop (C/L)
        }
    }

    result
}
