//! Compute the n-vector xi by using log_2 n random challenges.
//! 
//! Compute the n-vector psi by using the n-vector xi and a random s.
//!  
//! 
use ark_ff::PrimeField;

use rayon::prelude::*;

/// Compute the n-vector xi by using log_2 n random challenges.
/// 
pub fn xi_from_challenges<F: PrimeField>(challenges: &Vec<F>)
-> Vec<F> 
{
    
    let log_n = challenges.len();

    let mut xi = vec![F::one()];

    let mut xi_right: Vec<F>;

    for j in 0..log_n {
        xi_right = xi.par_iter().map(
            |a| 
            *a * challenges[log_n - j - 1]
        ).collect::<Vec<F>>();
        xi.append(&mut xi_right);
    }

    xi 
}

/// Compute the inner product of two tensor-structured-vectors.
/// 
pub fn xi_ip_from_challenges<F:PrimeField>(
    challenges1: &Vec<F>,
    challenges2: &Vec<F>,
) -> F {
    let len1 = challenges1.len();
    let len2 = challenges2.len();
    let len = std::cmp::min(len1, len2);
    let challenges1 = &challenges1[len1-len..len1].to_vec();
    let challenges2 = &challenges2[len2-len..len2].to_vec();

    let mut result = F::one();

    for i in 0..len {
        let cur = challenges1[i] * challenges2[i];
        // println!("cur: {}", cur);
        result = result.mul(F::one().add(cur));
    }

    result
}
