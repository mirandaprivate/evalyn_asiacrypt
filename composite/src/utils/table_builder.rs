// Build lookup table for sigmoid
// 
// A typical NN layer in an MLP (Multi-Layer Perceptron) uses the sigmoid activation function.
// This function maps any real-valued number into the range (0, 1).
// The sigmoid function is defined as:
// 
//     sigmoid(x) = 1 / (1 + exp(-x))
// 
// To efficiently compute the sigmoid function for a range of inputs,
// we can precompute the values and store them in a lookup table.
// 
// In 8-bit quantized NN, an MLP layer is
// 
//     Y = Round ( 127 * sigmoid( [(1/127) W] * [ (1/127) X ]  + [ (1/127 * 1/127 * 128) B  ] ) )
// 
// Here, integers in W * X + 128 * B is at most 2^25 bits [- 2^24 + 1, 2^24 - 1]
// 
// We define phi:
//     phi(x) = Round(127 * sigmoid( [1 / 127] * [1/127] * x )
// 
// Then we can find three vectors, alpha, beta, and gamma, each of length 2^10, such that
//
//     phi(alpha) = phi(beta) = phi(gamma)
//     0 <= beta - alpha < 2^15
//
// To prove y = phi(x), it is enough to reveal (alpha, beta, gamma) that resides in the 2^10 lookup table, such that:
//     
//    alpha <= x < beta,  y = gamma,
//    
// For 16-bit quantization, integers in W * X + 128 * B is at most 2^42 bits
// The lookup table size is 2^{18} with
// 
//     0 <= beta - alpha < 2^{25}
//
use std::f64::consts::E as Exponential;
use ark_ff::PrimeField;
use rayon::prelude::*;

// Use i32 to store the internal results for 8-bit quantization and i64 for 16-bit quantization
// Need manually change the type for 16-bit quantization
//
use crate::{MyInt, MyShortInt};
// type MyInt = i64; // i32 for 8-bit quantization, and i64 for 16-bit quantization

pub const LOOKUPCONFIG8: (usize, usize, usize) = (25, 8 , 13);
pub const LOOKUPCONFIG16: (usize, usize, usize) = (42, 16, 25);

pub struct ActivationTable {
    alpha: Vec<MyInt>,
    beta: Vec<MyInt>,
    gamma: Vec<MyInt>,
    input_bw: usize,
    output_bw: usize,
    increment_bw: usize,
    table_len: usize,
}

impl ActivationTable {
    pub fn new() -> Self {
        // Use the parameters directly or set defaults based on MyShortInt type
        let (input_bw, output_bw, increment_bw) = if std::mem::size_of::<MyShortInt>() == 1 {
            // i8 case
            LOOKUPCONFIG8
        } else if std::mem::size_of::<MyShortInt>() == 2 {
            // i16 case
            LOOKUPCONFIG16
        } else {
            // Use provided parameters for other cases
            (64, 32, 30)
        };

        let (alpha, beta, gamma) = build_table(output_bw, input_bw, increment_bw);
        let table_len = alpha.len();
        ActivationTable { alpha, beta, gamma, input_bw, output_bw, increment_bw, table_len}
    }

    pub fn get_index(&self, x: MyInt) -> usize {
        // Use binary search to find index i such that alpha[i] <= x < alpha[i+1]
        self.alpha.binary_search(&x).unwrap_or_else(|i| i.saturating_sub(1))
    }

    pub fn get_lookup_table(&self) -> (Vec<MyInt>, Vec<MyInt>, Vec<MyInt>) {
        (self.alpha.clone(), self.beta.clone(), self.gamma.clone())
    }

    pub fn phi_via_lookup(&self, x: MyInt) -> MyInt {
        // Find (alpha, beta, gamma) such that phi(alpha) <= phi(x) < phi(beta)
        let index = self.get_index(x);
        self.gamma[index]
    }

    // Direct computation is faster than lookup
    pub fn phi(&self, x: MyInt) -> MyInt {
        phi(x, self.output_bw)
    }

    pub fn check_y_phi_x(&self, x: MyInt, y: MyInt) -> (bool, usize) {
        let index = self.get_index(x);
        (self.gamma[index] == y && self.alpha[index] <= x && x <= self.beta[index], index)
    }

    pub fn get_input_range(&self) -> (MyInt, MyInt) {
        
        let max_input = (1u64 << (self.input_bw as u64 - 1)) as MyInt - 1;
        let min_input = - max_input;

        (min_input, max_input)
    }

    pub fn get_output_range(&self) -> (MyInt, MyInt) {
        let max_output = (1u64 << (self.output_bw as u64 - 1)) as MyInt - 1;
        let min_output = -max_output;
        (min_output, max_output)
    }

    pub fn get_output_bw(&self) -> usize { self.output_bw }

    pub fn get_diff_table_len(&self) -> usize {
        (1u32 << (self.increment_bw as u32)) as usize
    }

    pub fn get_increment_bw(&self) -> usize {
        self.increment_bw
    }

    pub fn get_lookup_table_len(&self) -> usize {
        self.table_len
    }

    pub fn get_index_vec(&self, x_vec: &Vec<MyInt>) -> Vec<usize> {
        x_vec.par_iter().map(|&x| self.get_index(x)).collect()
    }


    // Generate the auxiliary vectors for subsequent lookup proofs for
    // It is more efficient to compute the auxiliary inputs with index_vec 
    // 
    pub fn gen_lookup_inputs(&self, x_vec: &Vec<MyInt>) -> (Vec<MyInt>, Vec<MyInt>, Vec<MyInt>, Vec<MyInt>, Vec<MyInt>) {
        // Precompute indices
        let index_vec = self.get_index_vec(x_vec);
        let target_len = index_vec.len();

        // Generate (alpha, beta, gamma)
        let target_alpha: Vec<MyInt> = index_vec.par_iter().map(|&i| self.alpha[i]).collect();
        let target_beta:  Vec<MyInt> = index_vec.par_iter().map(|&i| self.beta[i]).collect();
        let target_gamma: Vec<MyInt> = index_vec.par_iter().map(|&i| self.gamma[i]).collect();

        // Build auxiliary vectors: table_auxiliary counts occurrences; target_auxiliary counts occurrences before current position
        let mut table_auxiliary = vec![0; self.table_len];
        let mut target_auxiliary = vec![0; target_len];
        for (pos, &idx) in index_vec.iter().enumerate() {
            target_auxiliary[pos] = table_auxiliary[idx];
            table_auxiliary[idx] += 1;
        }

        (target_alpha, target_beta, target_gamma, target_auxiliary, table_auxiliary)
    }

    // Here the methods for computing the grand products are slightly different
    // // Define four grand products:
    //
    //     Theta1 = \prod_{i=1}^m (1 + z * target-alpha_i + z^2 * target-beta_i + z^3 * target-gamma_i  + z^4 * target-auxiliary_i)
    //     Theta2 = \prod_{j=1}^n (1 + z * table-alpha_j + z^2 * table-beta_j + z^3 * table-gamma_j + z^4 * table-auxiliary_j)
    //     Theta3 = \prod_{i=1}^m (1 + z * target-alpha_i + z^2 * target-beta_i + z^3 * target-gamma_i + z^4 * ( target-auxiliary + 1))
    //     Theta4 = \prod_{j=1}^n (1 + z * table-alpha_j + z^2 * table-beta_j + z^3 * table-gamma_j)
    //
    // For any challenge z, we have:
    //     
    //     Theta1 * Theta2 = Theta3 * Theta4
    // 
    pub fn gen_lookup_theta<F:PrimeField>(
        &self,
        target_alpha: &Vec<MyInt>,
        target_beta: &Vec<MyInt>,
        target_gamma: &Vec<MyInt>,
        target_auxiliary: &Vec<usize>,
        table_auxiliary: &Vec<usize>,
        z: F,
    ) -> (F, F, F, F)
    where
        F: PrimeField,
    {
        // Precompute powers of z
        let z2 = z * z;          // z^2
        let z3 = z2 * z;         // z^3
        let z4 = z2 * z2;        // z^4

        // Parallel chunk size
        let num_cpus = rayon::current_num_threads();
        let tgt_chunk = (target_alpha.len() + num_cpus - 1) / num_cpus;
        let tbl_chunk = (self.alpha.len() + num_cpus - 1) / num_cpus;

        // Theta1 & Theta3 over target
        let (theta1, theta3) = target_alpha
            .par_chunks(tgt_chunk)
            .zip(target_beta.par_chunks(tgt_chunk))
            .zip(target_gamma.par_chunks(tgt_chunk))
            .zip(target_auxiliary.par_chunks(tgt_chunk))
            .map(|(((a_chunk, b_chunk), g_chunk), t_chunk)| {
                let mut local_t1 = F::one();
                let mut local_t3 = F::one();
                for (((&a, &b), &g), &t_val) in a_chunk.iter().zip(b_chunk).zip(g_chunk).zip(t_chunk) {
                    // Local conversion
                    let a_f = F::from(a as u64);
                    let b_f = F::from(b as u64);
                    let g_f = F::from(g as u64);
                    let t_f = F::from(t_val as u64);
                    // (1 + z a + z^2 b + z^3 g + z^4 t)
                    let base = F::one() + z * a_f + z2 * b_f + z3 * g_f;
                    local_t1 *= base + z4 * t_f;
                    local_t3 *= base + z4 * (t_f + F::one());
                }
                (local_t1, local_t3)
            })
            .reduce(|| (F::one(), F::one()), |(a1, a3), (l1, l3)| (a1 * l1, a3 * l3));

        // Theta2 & Theta4 over table (self.alpha/beta/gamma 与 table_auxiliary)
        let (theta2, theta4) = self
            .alpha
            .par_chunks(tbl_chunk)
            .zip(self.beta.par_chunks(tbl_chunk))
            .zip(self.gamma.par_chunks(tbl_chunk))
            .zip(table_auxiliary.par_chunks(tbl_chunk))
            .map(|(((a_chunk, b_chunk), g_chunk), s_chunk)| {
                let mut local_t2 = F::one();
                let mut local_t4 = F::one();
                for (((&a, &b), &g), &s_val) in a_chunk.iter().zip(b_chunk).zip(g_chunk).zip(s_chunk) {
                    let a_f = F::from(a as u64);
                    let b_f = F::from(b as u64);
                    let g_f = F::from(g as u64);
                    let s_f = F::from(s_val as u64);
                    let base = F::one() + z * a_f + z2 * b_f + z3 * g_f;
                    local_t2 *= base + z4 * s_f; // Theta2 term
                    local_t4 *= base;             // Theta4 term
                }
                (local_t2, local_t4)
            })
            .reduce(|| (F::one(), F::one()), |(a2, a4), (l2, l4)| (a2 * l2, a4 * l4));

        (theta1, theta2, theta3, theta4)
    }

}


pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + Exponential.powf(-x))
}

pub fn phi (x: MyInt, output_bw: usize) -> MyInt {
    // max_output = 2^{output_bw-1}-1  (note: add parentheses to avoid shift precedence errors)
    let max_output = (1u32 << (output_bw - 1)) - 1; 
    let denom = (max_output as f64) * (max_output as f64);
    let x_f64 = (x as f64) / denom;
    let sig = sigmoid(x_f64);
    ( (max_output as f64) * sig ).round() as MyInt
}


pub fn phi_inv_round (y: MyInt, output_bw: usize, input_bw: usize) -> MyInt {
    // max_output = 2^{output_bw-1}-1  (note: add parentheses to avoid shift precedence errors)
    let max_output = (1u32 << (output_bw - 1)) as f64 - 1.0; // e.g. output_bw=8 -> 127
    let max_input: MyInt = ((1u64 << (input_bw as u64 - 1)) - 1) as MyInt;
    let min_input: MyInt = -max_input;
    // Avoid y=0 or y=max_output causing logit(0)/logit(1) -> ±∞/NaN
    let y_clamped_int = if y <= 0 { (min_input as MyInt) - 1 } else if (y as f64) >= max_output { (max_input as MyInt) + 1 } else { y };
    let y_f64 = (y_clamped_int as f64) / max_output;
    let x_f64 = sigmoid_inv(y_f64); // finite value
    ( x_f64 * max_output * max_output ).ceil() as MyInt
}


pub fn sigmoid_inv(y: f64) -> f64 {
    // logit(y) = ln(y / (1 - y)) = - ln(1/y - 1)
    -((1.0 / y) - 1.0).ln()
}


pub fn phi_inv_max(y: MyInt, output_bw: usize, input_bw: usize) -> MyInt {

    let max_output: MyInt = ((1u64 << (output_bw as u64 - 1)) - 1) as MyInt;
    let max_input: MyInt = ((1u64 << (input_bw as u64 - 1)) - 1) as MyInt;
    let min_input: MyInt = -max_input;

    if y == 0 {
        let max_x_for_zero_y = max_x_for_zero_y(output_bw, input_bw);
        if max_x_for_zero_y > min_input {
            return max_x_for_zero_y;
        } else {
            return min_input;
        }
    }
    
    if y >= max_output { return max_input; }

    let mut lo = phi_inv_round(y, output_bw, input_bw);
    let mut cur_y = y;
    if phi(lo, output_bw) > y {
        cur_y -= 1;
        lo = phi_inv_round(cur_y, output_bw, input_bw);
    }
    if lo < min_input {
        lo = min_input;
    }

    let mut hi = phi_inv_round(y + 1, output_bw, input_bw);
    let mut cur_y = y + 1;
    if phi(hi, output_bw) <= y {
        cur_y += 1;
        hi = phi_inv_round(cur_y, output_bw, input_bw);
    }
    if hi > max_input + 1 {
        hi = max_input + 1;
    }

    // Find the maximum x between lower bound and upper bound such that phi(x) = y
    // If x > max_input then return max_input

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if phi(mid, output_bw) > y { hi = mid; } else { lo = mid + 1; }
    }
    let mut candidate = lo - 1; // Convert right-open interval to plateau end guess
    while phi(candidate, output_bw) > y { candidate -= 1; }
    while phi(candidate + 1, output_bw) == y { candidate += 1; }

    assert_eq!(phi(candidate, output_bw), y);
    assert!(phi(candidate + 1, output_bw) > y);

    candidate
}

// Return the maximum x where phi(x)=0 (if 0 is unreachable, degenerate to minimum endpoint).
pub fn max_x_for_zero_y(output_bw: usize, input_bw: usize) -> MyInt {
    let shift = ((input_bw as i32 - 1).clamp(0, 62)) as u32;
    let max_input: MyInt = ((1i64 << shift) - 1) as MyInt;
    let min_input: MyInt = -max_input;

    // If the minimum endpoint is already >0, then 0 is unreachable, return min_input (no plateau)
    if phi(min_input, output_bw) > 0 { return min_input; }
    // If the maximum endpoint is still 0, the entire segment is 0, return max_input
    if phi(max_input, output_bw) == 0 { return max_input; }

    // lo(1) - 1
    let mut lo = min_input; // phi(lo)==0
    let mut hi = max_input + 1; // exclusive
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if phi(mid, output_bw) >= 1 { hi = mid; } else { lo = mid + 1; }
    }
    let mut candidate = lo - 1; // Maximum x making phi(x)=0
    while phi(candidate, output_bw) > 0 { candidate -= 1; }
    while phi(candidate + 1, output_bw) == 0 { candidate += 1; }
    candidate
}



pub fn build_table(
    output_bw: usize,
    input_bw: usize,
    increment_bw: usize,
) -> (
    Vec<MyInt>,
    Vec<MyInt>,
    Vec<MyInt>,
 ) {
    let mut alpha_vec: Vec<MyInt> = Vec::new();
    let mut beta_vec: Vec<MyInt> = Vec::new();
    let mut gamma_vec: Vec<MyInt> = Vec::new();

    let max_input = (1u64 << (input_bw as u64 - 1)) as MyInt - 1;
    let min_input = -max_input;

    let _max_output = (1u64 << (output_bw as u64 - 1)) as MyInt - 1;
    let _min_output = 0;

    let increment = (1u64 << (increment_bw as u64)) as MyInt;

    let mut cur_alpha = min_input;
    let mut cur_beta = min_input;
    let mut cur_gamma;
    // let mut count = 0;

    while cur_beta < max_input {
        
        cur_gamma = phi(cur_alpha, output_bw);
        let cur_cap_x = phi_inv_max(cur_gamma, output_bw, input_bw);

        cur_beta = cur_alpha + increment;
        cur_beta = std::cmp::min(cur_beta, cur_cap_x + 1);

        alpha_vec.push(cur_alpha.clone());
        beta_vec.push(cur_beta.clone());
        gamma_vec.push(cur_gamma.clone());

        // count += 1;
        // if count % (1 << 10) == 0 {
        //     println!("Processed {} * 2^10 entries, cur_alpha: {}, cur_beta: {}, cur_gamma: {}, cur_range: {}", count/ (1<<10), cur_alpha, cur_beta, cur_gamma, cur_beta - cur_alpha);
        // }

        cur_alpha = cur_beta;

        while cur_alpha + increment <= cur_cap_x {
            cur_beta = cur_alpha + increment;

            alpha_vec.push(cur_alpha.clone());
            beta_vec.push(cur_beta.clone());
            gamma_vec.push(cur_gamma.clone());

            // count += 1;
            // if count % (1 << 10) == 0 {
            //     println!("Processed {} * 2^10 entries, cur_alpha: {}, cur_beta: {}, cur_gamma: {}, cur_range: {}", count/ (1<<10), cur_alpha, cur_beta, cur_gamma, cur_beta - cur_alpha);
            // }

            cur_alpha = cur_beta;
        }
    }

    let table_len = alpha_vec.len();
    let len_padded = table_len.next_power_of_two();
    let log_len = len_padded.ilog2() as usize;

    let alpha_pad = alpha_vec[table_len-1].clone();
    let beta_pad = beta_vec[table_len-1].clone();
    let gamma_pad = gamma_vec[table_len-1].clone();

    alpha_vec.extend(vec![alpha_pad; len_padded - table_len]);
    beta_vec.extend(vec![beta_pad; len_padded - table_len]);
    gamma_vec.extend(vec![gamma_pad; len_padded - table_len]);


    beta_vec = beta_vec.into_iter().map(|x| x - 1).collect(); // Decrement each element by 1 such that the intervals are discrete

    println!("Lookup table of length 2^{} built", log_len);


    (alpha_vec, beta_vec, gamma_vec)
}

// Compute the auxiliary vectors vec(t) and vec(s) in lookup proof
// For proving all entries of an m-sized **target** vec(c) reside in an n-sized **table** \vec{v}
// we need to construct m-vector vec(t) and n-vector vec(s)
// 
// ```text
// **table auxiliary** s_j, j \in [n] counts the occurance of v_j in vec(c)
// **target auxiliary** t_i, i \in [m] records how many times each c_i appears in the previous entries {c_1, ... ,c_{i-1}}
// 
// Define four grand products:
// 
//     Theta1 = \prod_{i=1}^m (1 + zc_i + z^2 t_i)
//     Theta2 = \prod_{j=1}^n (1 + zv_j + z^2 s_j)
//     Theta3 = \prod_{i=1}^m (1 + zc_i + z^2 (t_i + 1))
//     Theta4 = \prod_{j=1}^n (1 + zv_j)
// 
// For any challenge z, we have:
//     
//     Theta1 * Theta2 = Theta3 * Theta4
// ```
// 
//  In the following function, we require table to be a sorted vector
// 
pub fn lookup_auxiliary_builder(target: &Vec<MyInt>, table: &Vec<MyInt>) -> (Vec<MyInt>, Vec<MyInt>) {
    
    let table_size = table.len();
    let target_size = target.len();

    let mut table_auxiliary = vec![0; table_size];
    let mut target_auxiliary = Vec::new();

    for i in 0..target_size {
        let elem = target[i];
        let j = table.binary_search(&elem).unwrap_or_else(|_| panic!("Element {} not found in table", elem));

        target_auxiliary.push(table_auxiliary[j]);
        table_auxiliary[j] += 1;
    }

    (target_auxiliary, table_auxiliary)
}

// The auxiliary inputs for range proof
// All elements in target is non-negative and at most k bits
// The table is constructed as [0, 1, ..., 2^k - 1]
pub fn range_auxiliary_builder(target: &Vec<MyInt>, k: usize) -> (Vec<MyInt>, Vec<MyInt>) {
    
    let table = (0..(1 << k)).collect::<Vec<MyInt>>();

    lookup_auxiliary_builder(target, &table)
}


pub fn compute_target_theta<F: PrimeField>(target: &Vec<MyInt>, target_auxiliary: &Vec<MyInt>, z: F) -> (F, F) {
    let num_cpus = rayon::current_num_threads();
    let chunk_size = (target.len() + num_cpus - 1) / num_cpus;
    // Parallel computation of local theta1 and theta3 for each thread (using rayon par_chunks)
    let (theta1, theta3): (F, F) = target
        .par_chunks(chunk_size)
        .zip(target_auxiliary.par_chunks(chunk_size))
        .map(|(target_chunk, auxiliary_chunk)| {
            let mut local_theta1 = F::one();
            let mut local_theta3 = F::one();
            let z_squared = z * z;
            for (&c_i, &t_i) in target_chunk.iter().zip(auxiliary_chunk.iter()) {
                let c_field = F::from(c_i as u64);
                let t_field = F::from(t_i as u64);
                local_theta1 *= F::one() + z * c_field + z_squared * t_field;
                local_theta3 *= F::one() + z * c_field + z_squared * (t_field + F::one());
            }
            (local_theta1, local_theta3)
        })
        .reduce(|| (F::one(), F::one()), |(a1, a3), (l1, l3)| (a1 * l1, a3 * l3));
    (theta1, theta3)
}

pub fn compute_table_theta<F: PrimeField>(table: &Vec<MyInt>, table_auxiliary: &Vec<MyInt>, z: F) -> (F, F) {
    let num_cpus = rayon::current_num_threads();
    let chunk_size = (table.len() + num_cpus - 1) / num_cpus;
    // Parallel computation of local theta2 and theta4 for each thread
    let (theta2, theta4): (F, F) = table
        .par_chunks(chunk_size)
        .zip(table_auxiliary.par_chunks(chunk_size))
        .map(|(table_chunk, auxiliary_chunk)| {
            let mut local_theta2 = F::one();
            let mut local_theta4 = F::one();
            let z_squared = z * z;
            for (&v_j, &s_j) in table_chunk.iter().zip(auxiliary_chunk.iter()) {
                let v_field = F::from(v_j as u64);
                let s_field = F::from(s_j as u64);
                local_theta2 *= F::one() + z * v_field + z_squared * s_field;
                local_theta4 *= F::one() + z * v_field;
            }
            (local_theta2, local_theta4)
        })
        .reduce(|| (F::one(), F::one()), |(a2, a4), (l2, l4)| (a2 * l2, a4 * l4));
    (theta2, theta4)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;

    
    #[test]
    fn test_table_phi_via_lookup_range() {
        // Full traversal of interval [- (2^{25}-1), 2^{25}-1], no sampling.
        let table = ActivationTable::new();
        let max_input: MyInt = ((1i64 << (25 as u32 - 1)) - 1) as MyInt;
        let min_input: MyInt = -max_input;

        // Test lookup version first (to avoid cache/branch prediction bias affecting comparison order, can be swapped as needed)
        let t_lookup_start = Instant::now();
        for x in min_input..=max_input { table.phi_via_lookup(x); }
        let dur_lookup = t_lookup_start.elapsed();

        let t_direct_start = Instant::now();
    for x in min_input..=max_input { table.phi(x); }
        let dur_direct = t_direct_start.elapsed();

        // Do another strict point-by-point comparison (avoid putting comparison in timing loop to prevent performance interference)
        for x in min_input..=max_input {
            let d = table.phi(x);
            let v = table.phi_via_lookup(x);
            assert_eq!(d, v, "Mismatch at x={} direct={} via={}", x, d, v);
        }

        println!(
            "Full traverse: N={} direct={:.2}s lookup={:.2}s speedup={:.2}x",
            (max_input - min_input + 1),
            dur_direct.as_secs_f64(),
            dur_lookup.as_secs_f64(),
            if dur_lookup.as_nanos()>0 { dur_direct.as_secs_f64()/dur_lookup.as_secs_f64() } else { 0.0 }
        );
    }

    #[test]
    fn test_table_activation_gen_lookup_inputs_random() {
    let table = ActivationTable::new();
        let (min_input, max_input) = table.get_input_range();

        // Randomly generate x_vec
        let mut rng = StdRng::seed_from_u64(2024);
        let sample_len = 1usize << 10; // 1024 points
        let mut x_vec = Vec::with_capacity(sample_len);
        for _ in 0..sample_len { x_vec.push(rng.random_range(min_input..=max_input)); }

        // Calculate y_vec = phi(x)
        let y_vec: Vec<MyInt> = x_vec.iter().map(|&x| table.phi(x)).collect();

        // Call gen_lookup_inputs
        let (target_alpha, target_beta, target_gamma, target_aux, table_aux) = table.gen_lookup_inputs(&x_vec);

        // Compare with generic constructor results to verify correctness (moved from implementation to test)
        let (target_auxiliary_exp, table_auxiliary_exp) = lookup_auxiliary_builder(&target_alpha, &table.alpha);
        assert_eq!(target_auxiliary_exp, target_aux, "target auxiliary mismatch");
        assert_eq!(table_auxiliary_exp, table_aux, "table auxiliary mismatch");

        // target_gamma should equal y_vec
        assert_eq!(target_gamma, y_vec, "target_gamma does not equal y_vec");

        // Verify each x falls in [alpha, beta) interval and gamma = phi(alpha)
        for i in 0..x_vec.len() {
            let x = x_vec[i];
            let alpha = target_alpha[i];
            let beta = target_beta[i];
            let gamma = target_gamma[i];
            assert!(alpha <= x && x <= beta, "x not in interval i={} x={} alpha={} beta={}", i, x, alpha, beta);
            assert_eq!(table.phi(alpha), gamma, "gamma does not match phi(alpha) at i={}", i);
            assert_eq!(table.phi(beta), gamma, "gamma does not match phi(beta) at i={}", i);
        }

        // Basic consistency: table_aux count sum equals x_vec length
        let total: usize = table_aux.iter().map(|&v| v as usize).sum();
        assert_eq!(total, x_vec.len(), "table_aux count sum error");

        // target_aux monotonic non-decreasing property (by occurrence order of same value): extract subsequence for each value and check for 0,1,2,...
        use std::collections::HashMap;
        let mut map: HashMap<MyInt, Vec<MyInt>> = HashMap::new();
    for (i, &alpha) in target_alpha.iter().enumerate() { map.entry(alpha).or_default().push(target_aux[i]); }
        for (_k, v) in map.iter() {
            for (idx, occ) in v.iter().enumerate() { assert_eq!(*occ as usize, idx, "occurrence sequence incorrect"); }
        }
    }
    /*
    #[test]
    fn test_table_phi_via_lookup_range_16() {
        // Full traversal of interval [- (2^{42}-1), 2^{42}-1], no sampling.
        const OBW: usize = 16;
        const IBW: usize = 42; // => max_input = 2^{42}-1
        const INCREMENT_BW: usize = 25;
        let table = ActivationTable::new(OBW, IBW, INCREMENT_BW);
        println!("Table built!");

        let max_input: MyInt = ((1i64 << (IBW as u32 - 1)) - 1) as MyInt;
        let min_input: MyInt = -max_input;

        let t_lookup_start = Instant::now();
        for i in 0..(1usize << 30) {
            let x = (i as MyInt) * (1 << 11) as MyInt;
            table.phi_via_lookup(x);
        }
        let dur_lookup = t_lookup_start.elapsed();

        let t_direct_start = Instant::now();
        for i in 0..(1usize << 41) {
            let x = (i as MyInt) * (1 << 0) as MyInt;
            phi(x, OBW);
        }
        let dur_direct = t_direct_start.elapsed();


        println!(
            "Full traverse: N={} direct={:.6}s lookup={:.6}s speedup={:.2}x",
            (max_input - min_input + 1),
            dur_direct.as_secs_f64(),
            dur_lookup.as_secs_f64(),
            if dur_lookup.as_nanos()>0 { dur_direct.as_secs_f64()/dur_lookup.as_secs_f64() } else { 0.0 }
        );

        for x in (min_input..=max_input).step_by(1 << 12) {
            let d = phi(x, OBW);
            let v = table.phi_via_lookup(x);
            assert_eq!(d, v, "Mismatch at x={} direct={} via={}", x, d, v);
        }
    }
     */


    /* // The following only works for MyInt = i64
    #[test]
    fn test_table_build_table_16_42() {
        // The resulting lookup table length is 2^20
        let output_bw = 16usize;
        let input_bw = 42usize; 
        let increment_bw = 25usize;

        let max_output = 1i64 << (output_bw as u64 - 1) as MyInt - 1;
        assert! ( max_output * max_output * 1024 + (max_output + 1) * max_output < (1i64 << (input_bw - 1) as MyInt) as MyInt);

        let (alpha, beta, gamma) = build_table(output_bw, input_bw, increment_bw);


        for i in 0..alpha.len() {

            let alpha_val = alpha[i];
            let beta_val = beta[i];
            let gamma_val = gamma[i];

            assert_eq!(phi(alpha_val, output_bw), gamma_val, "phi(alpha) should equal phi(beta)");
            assert_eq!(phi(beta_val - 1 as MyInt, output_bw), gamma_val, "phi(beta) should equal phi(gamma)");
            // This is important because the lookup table should be [1..2^increment_bw]
            assert!(beta_val - alpha_val <= (1i64 << increment_bw) as MyInt, "beta - alpha should be no more than 2^increment_bw");
            assert!(beta_val > alpha_val, "beta should be greater than alpha");
        }

        println!("build_table({}, {}, {}) OK (placeholder)", output_bw, input_bw, increment_bw);
    }
    */

    #[test]
    fn test_table_build_table_8_25() {
        // The resulting lookup table length is 2^10
        let output_bw = 8usize;
        let input_bw = 25usize; 
        let increment_bw = 13usize;

        assert! ( 127 * 127 * 1024 + 128 * 127 < (1i64 << (input_bw - 1) as MyInt) as MyInt);
        let (alpha, beta, gamma) = build_table(output_bw, input_bw, increment_bw);

        for i in 0..alpha.len() {
            let alpha_val = alpha[i];
            let beta_val = beta[i];
            let gamma_val = gamma[i];

            assert_eq!(phi(alpha_val, output_bw), gamma_val, "phi(alpha) should equal phi(beta)");
            assert_eq!(phi(beta_val - 1 as MyInt, output_bw), gamma_val, "phi(beta) should equal phi(gamma)");
            // This is important because the lookup table should be [1..2^increment_bw]
            assert!(beta_val - alpha_val <= (1i64 << increment_bw) as MyInt, "beta - alpha should be no more than 2^increment_bw");
            assert!(beta_val > alpha_val, "beta should be greater than alpha");
        }

        println!("build_table({}, {}, {}) OK (placeholder)", output_bw, input_bw, increment_bw);
    }

    #[test]
    fn test_table_lookup_random_theta_identity() {
        use ark_bls12_381::Fr;
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        use std::collections::HashSet;

        const TABLE_LEN: usize = 257; 
        const TARGET_LEN: usize = 800; 
        let mut rng = StdRng::seed_from_u64(42);

        // Generate unique random integer set and sort as ordered table
        let mut set: HashSet<MyInt> = HashSet::new();
        while set.len() < TABLE_LEN {
            let val = rng.random_range(-50_000..50_000); // wider range
            set.insert(val);
        }
        let mut table: Vec<MyInt> = set.into_iter().collect();
        table.sort();

        // Randomly sample from table (allowing repetition) to construct target
        let mut target: Vec<MyInt> = Vec::with_capacity(TARGET_LEN);
        for _ in 0..TARGET_LEN {
            let idx = rng.random_range(0..table.len());
            target.push(table[idx]);
        }

        let (target_aux, table_aux) = lookup_auxiliary_builder(&target, &table);

        let total_count: usize = table_aux.iter().map(|&x| x as usize).sum();
        assert_eq!(total_count, target.len(), "table_aux count sum should equal target size");

        // target_aux first occurrence should be 0
        for (i, &t) in target_aux.iter().enumerate() { if t == 0 { continue; } else { assert!(t < i as MyInt); } }

        // Randomly select multiple z values to verify Theta1 * Theta2 = Theta3 * Theta4
        for _trial in 0..5 {
            // Random non-zero z (if zero the identity still holds, here we avoid degenerate cases)
            let mut z_u64 = rng.random::<u64>();
            if z_u64 == 0 { z_u64 = 1; }
            let z = Fr::from(z_u64);
            let (theta1, theta3) = compute_target_theta(&target, &target_aux, z);
            let (theta2, theta4) = compute_table_theta(&table, &table_aux, z);
            assert_eq!(theta1 * theta2, theta3 * theta4, "Grand product identity failed for random z");
        }
    }

    #[test]
    fn test_table_activation_gen_lookup_theta_identity() {
        use ark_bls12_381::Fr;
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;

        let table = ActivationTable::new();
        let (min_x, max_x) = table.get_input_range();

        // Randomly sample x_vec
        let mut rng = StdRng::seed_from_u64(7777);
        let m = 1500usize;
        let mut x_vec = Vec::with_capacity(m);
        for _ in 0..m { x_vec.push(rng.random_range(min_x..=max_x)); }

        // Generate lookup inputs
        let (t_alpha, t_beta, t_gamma, t_aux, tbl_aux) = table.gen_lookup_inputs(&x_vec);

        let t_aux_usize: Vec<usize> = t_aux.iter().map(|&v| v as usize).collect();
        let tbl_aux_usize: Vec<usize> = tbl_aux.iter().map(|&v| v as usize).collect();
        for _ in 0..4 { // 4 个 z
            let mut z_raw = rng.random::<u64>();
            if z_raw == 0 { z_raw = 5; }
            let z = Fr::from(z_raw);
            let (theta1, theta2, theta3, theta4) = table.gen_lookup_theta(&t_alpha, &t_beta, &t_gamma, &t_aux_usize, &tbl_aux_usize, z);
            assert_eq!(theta1 * theta2, theta3 * theta4, "Extended grand product identity failed");
        }
    }
    
}