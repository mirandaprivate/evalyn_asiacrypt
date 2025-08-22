#![deny(trivial_numeric_casts, variant_size_differences)]
//! Compute the n-vector xi by using log_2 n random challenges.
//! 
//! Compute the n-vector psi by using the n-vector xi and a random s.
//!  
//! 
use ark_ec::pairing::Pairing;
// use ark_ff::PrimeField;
use ark_std::{
    One, Zero,
    ops::{Add, Mul},
};


/// Compute the n-vector xi by using log_2 n random challenges.
/// 
pub fn xi_from_challenges<E: Pairing>(challenges: &Vec<E::ScalarField>)
-> Vec<E::ScalarField> {
    
    let log_n = challenges.len();

    let mut xi = vec![E::ScalarField::one()];

    for j in 0..log_n {
        let mut xi_left = xi.clone();
        let mut xi_right = xi.iter().map(
            |a| 
            *a * challenges[log_n - j - 1]
        ).collect::<Vec<E::ScalarField>>();
        xi_left.append(&mut xi_right);
        xi = xi_left;
    }

    xi 
}

/// Compute the inner product of two tensor-structured-vectors.
/// 
pub fn xi_ip_from_challenges<E: Pairing>(
    challenges1: &Vec<E::ScalarField>,
    challenges2: &Vec<E::ScalarField>,
) -> E::ScalarField {
    let len1 = challenges1.len();
    let len2 = challenges2.len();
    let len = std::cmp::min(len1, len2);
    let challenges1 = &challenges1[len1-len..len1].to_vec();
    let challenges2 = &challenges2[len2-len..len2].to_vec();

    let mut result = E::ScalarField::one();

    for i in 0..len {
        let cur = challenges1[i] * challenges2[i];
        result = result.mul(E::ScalarField::one().add(cur));
    }

    result
}

// /// Reduce a n-vector to a scalar by using log_2 n random challenges.
// /// 
// pub fn reduce_from_challenges<T>(
//     challenges: &Vec<ZpElement>, vector: &Vec<T>
// ) -> T 
//     where
//     T: 'static + Clone + Copy + Send + Sync 
//         + Mul<ZpElement, Output =T> + Add + Zero,
// {    

//     let mut xi = xi_from_challenges(challenges);

//     let l = std::cmp::min(xi.len(), vector.len());

//     if xi.len() > vector.len() {
//         xi = xi[0..l].to_vec();
//     }

//     let vector_slice = vector[0..l].to_vec();

//     dirac::inner_product(&vector_slice, &xi) 
// }

/// Compute phi(s) in logarithmic time.
/// 
pub fn phi_s<E:Pairing>(
    s: E::ScalarField,
    challenges: &Vec<E::ScalarField>,
) -> E::ScalarField {
    let log_n = challenges.len();
    

    let s_pow_vec = std::iter::successors(
        Some(s),
        |&x| Some(x * x),
    ).take(log_n).collect::<Vec<E::ScalarField>>();


    let product = challenges.iter().rev()
        .zip(s_pow_vec.iter())
        .map(
            |(&a, &b)| 
            E::ScalarField::one() + a * b
        ).fold(E::ScalarField::one(), |acc, x| acc * x);

    product
}


/// Compute the coefficients of the quotient polynomial with a given point
pub fn quotient_poly<E:Pairing> (
    vec: &Vec<E::ScalarField>,
    point: E::ScalarField,
) -> Vec<E::ScalarField> {
   
    let mut result_rev = Vec::new();
    result_rev.push(E::ScalarField::zero());

    let n = vec.len();
    for i in 1..n {

        let cur = result_rev[i-1] * point + vec[n-i];
        result_rev.push(cur);
    }

    result_rev.into_iter().rev().collect()
}


/// Compute phi(s) directly. For unit test purpose only.
/// 
pub fn phi_s_direct<E:Pairing>(
    s: E::ScalarField, 
    coeffs: &Vec<E::ScalarField>, 
) -> E::ScalarField {
    
    let len = coeffs.len();
    let powers_of_s = std::iter::successors(
        Some(E::ScalarField::one()), 
        |&x| Some(x * s)
    ).take(len).collect::<Vec<E::ScalarField>>();
    
    super::utils::inner_product::<E>(&coeffs, &powers_of_s)
}

// /// Compute phi(s) by using the reduce method. For unit test purpose only.
// /// 
// #[cfg(test)]
// fn phi_s_reduce(
//     s: ZpElement, 
//     challenges: &Vec<ZpElement>, 
//     exp_start: usize, 
//     exp_step: usize,
// ) -> ZpElement {
    
//     let log_n = challenges.len() as usize;
//     let n = 2usize.pow(log_n as u32);
 
//     let start = s.pow(exp_start as u64);
//     let step = s.pow(exp_step as u64);

//     let s_vec: Vec<ZpElement> = std::iter::successors(
//         Some(start), 
//         |&x| Some(x * step)
//     ).take(n).collect();

//     let mut s_vec_current = s_vec.clone();

//     for i in 0..log_n {
//         let s_vec_left 
//             = s_vec_current[0..n/2usize.pow(i as u32 + 1)].to_vec();
//         let s_vec_right 
//             = s_vec_current[n/2usize.pow(i as u32 + 1)..].to_vec();
//         s_vec_current = s_vec_left.iter().zip(
//             s_vec_right.iter()
//             ).map(|(&a, &b)| 
//                 a + b * challenges[i]
//             ).collect::<Vec<ZpElement>>();
//     }

//     s_vec_current[0]
// }



#[cfg(test)]
mod tests{

    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::UniformRand;
  
    #[test]
    fn test_quotient()
    {
        let rng = &mut ark_std::test_rng();
        type Fr = <Bls12_381 as Pairing>::ScalarField;
        let point = Fr::rand(rng);

        let log_n = 5;
        let n = 2usize.pow(log_n);
        
        let vec: Vec<Fr> = (0..n).map(|_| 
            Fr::rand(rng)
        ).collect::<Vec<Fr>>();

        let quotient =
        quotient_poly::<Bls12_381>(&vec, point);

        let s = Fr::rand(rng);
        let eval_quotient =
        phi_s_direct::<Bls12_381>(s, &quotient);

        let eval_poly_point =
        phi_s_direct::<Bls12_381>(point, &vec);
        let eval_poly_s =
        phi_s_direct::<Bls12_381>(s, &vec);
        
        assert_eq!(
            eval_quotient.mul(point-s),
            eval_poly_point - eval_poly_s
        );
    }

}

//     #[test]
//     fn test_xi(){
//         let log_n = 10;

//         let challenges = (0..log_n).map(|_| 
//             ZpElement::rand()
//         ).collect::<Vec<ZpElement>>();

//         let s = ZpElement::rand();

//         let timer_direct = Instant::now();
//         let phi_s_direct = phi_s_direct(
//             s, &challenges, 3, 2);
//         println!(
//             " ** Compute phi_s direct time: {:?}", timer_direct.elapsed()
//         );
        
//         let timer_opt = Instant::now();
//         let phi_s_opt = phi_s(
//             s, &challenges, 3, 2);
//         println!(
//             " ** Compute phi_s opt time: {:?}", timer_opt.elapsed()
//         );
        
//         let timer_reduce = Instant::now();
//         let phi_s_reduce = phi_s_reduce(
//             s, &challenges, 3, 2);
//         println!(
//             " ** Compute phi_s reduce time: {:?}", timer_reduce.elapsed()
//         );

//         assert_eq!(phi_s_direct, phi_s_reduce);
//         assert_eq!(phi_s_reduce, phi_s_opt);

//         println!(" * Assert Equal across three methods ");
        
//         let n = 2usize.pow(log_n as u32);
//         let vector_g1 = (0..n).map(|_| 
//             G1Element::from(
//                 rand::thread_rng().gen::<u64>()
//             )
//         ).collect::<Vec<G1Element>>();

//         let timer_g1_reduce = Instant::now();
//         let g1_reduce = reduce_from_challenges(
//             &challenges, &vector_g1);
//         println!(" ** Compute g1 reduce opt time: {:?}", 
//                 timer_g1_reduce.elapsed());

//         let timer_g1_direct = Instant::now();
//         let g1_reduce_direct = reduce_from_challenges_g1_direct(
//             &challenges, &vector_g1);
//         println!(" ** Compute g1 reduce direct time: {:?}", 
//                 timer_g1_direct.elapsed());


//         assert_eq!(g1_reduce_direct, g1_reduce);

//         println!(" * Assert Equal between two methods ");

//         let s_hat = ZpElement::rand();

//         let s_hat_vec: Vec<ZpElement> = std::iter::successors(
//             Some(ZpElement::from(1 as u64)), 
//             |&x| Some(x * s_hat)
//         ).take(n).collect();


//         let xi = xi_from_challenges(&challenges);
//         let psi = psi_from_xi(&xi, s);

//         let xi_s = phi_s(
//             s, &challenges, 1, 1);
//         let xi_s_hat = phi_s(
//             s_hat, &challenges, 1, 1);
//         let psi_s_hat = dirac::inner_product(
//             &psi, &s_hat_vec);
//         assert_eq!((s-s_hat) * psi_s_hat, xi_s - xi_s_hat );
  
//         println!(" * Assert correct computation of psi ");

//     }

// }