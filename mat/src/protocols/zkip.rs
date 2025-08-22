//! Implement the zk inner product protocol
//!
use rayon::prelude::*;

use ark_ec::PrimeGroup;
use ark_ff::PrimeField;

use crate::data_structures::ZkSRS;

use crate::utils::zktr::{ZkTranSeq, TranSeq, TranElem};
use crate::utils::linear;
// use crate::utils::xi;

use super::scalars::ZkMulScalar;


pub struct ZkLiteBullet<F, G>
where 
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_com: G,
    pub length: usize,
}

impl<F, G> ZkLiteBullet<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(
        v_com_value: G,
        length_value: usize,
    ) -> Self {
        Self {
            v_com: v_com_value,
            length: length_value,
        }
    }

    pub fn reduce_prover(
        &self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        vec_a: &Vec<F>,
        vec_b: &Vec<F>,
        tilde_v: F,
    ) -> (G, G, F, F, Vec<F>, Vec<F>)
    {
    println!("LiteBullet.reduce_prover: pushing v_com={:?}, length={}", self.v_com, self.length);
        zk_trans_seq.push_com(self.v_com);
        zk_trans_seq.push_size(self.length);
  
        if (self.length & (self.length - 1)) != 0 {
            panic!("Length is not a power of 2 when proving IpGt");
        }

        let n = self.length;
        let log_n = (n as u64).ilog2() as usize;

        let mut vec_a_current = vec_a;
        
        let mut vec_b_current = vec_b;
         
        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

        let mut lhs_tilde = tilde_v;
        let mut lhs_com = self.v_com.clone();

        let mut vec_a_value;
        let mut vec_b_value;


        for j in 0..log_n {
            let current_len = n / 2usize.pow(j as u32);
            
            let l_tr = 
                linear::inner_product_slice(&vec_a_current[..current_len/2], &vec_b_current[current_len/2..]);
            let r_tr = 
                linear::inner_product_slice(&vec_a_current[current_len/2..], &vec_b_current[..current_len/2]);

            let (l_com,l_tilde) = zk_trans_seq
            .push_gen_blinding(l_tr);
            let (r_com,r_tilde) = zk_trans_seq
            .push_gen_blinding(r_tr);

            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();

            challenges.push(x_j);
            challenges_inv.push(x_j_inv);

            lhs_tilde = lhs_tilde +  l_tilde * x_j + r_tilde * x_j_inv;
            lhs_com = lhs_com + l_com.mul(&x_j) + r_com.mul(&x_j_inv);

            vec_a_value = (0..current_len/2)
            .into_par_iter()
            .map(|i| vec_a_current[i] + vec_a_current[current_len/2 + i] * x_j_inv)
            .collect();

            vec_a_current = &vec_a_value;

            vec_b_value = (0..current_len/2)
            .into_par_iter()
            .map(|i| vec_b_current[i] + vec_b_current[current_len/2 + i] * x_j)
            .collect();

            vec_b_current = &vec_b_value;

            

        }

        let a_reduce = vec_a_current[0];
        let b_reduce = vec_b_current[0];

       

        // let xi_a = xi::xi_from_challenges::<F>(&challenges_inv);
        // assert_eq!(linear::inner_product(&vec_a, &xi_a), a_reduce);
        // println!(" * Check a_reduce in litebullet passed");
        // let xi_b = xi::xi_from_challenges::<F>(&challenges);
        // assert_eq!(linear::inner_product(&vec_b, &xi_b), b_reduce);
        // println!(" * Check b_reduce in litebullet passed");

        let (a_reduce_blind, a_reduce_tilde) =
            zk_trans_seq.push_gen_blinding(a_reduce);

        let (b_reduce_blind, b_reduce_tilde) =
            zk_trans_seq.push_gen_blinding(b_reduce);

        // let a_reduce_check = srs.com_base.mul(&a_reduce)
        // + srs.blind_base.mul(&a_reduce_tilde);
        // assert_eq!(a_reduce_check, a_reduce_blind);
        // println!(" * Check a_reduce_blind in litebullet passed");
        
        let zk_mul = ZkMulScalar::new(
            srs,
            lhs_com,
            a_reduce_blind,
            b_reduce_blind,
        );

        zk_mul.prove(
            zk_trans_seq,
            a_reduce,
            b_reduce,
            lhs_tilde,
            a_reduce_tilde,
            b_reduce_tilde,
        );

        (
            a_reduce_blind,
            b_reduce_blind,
            a_reduce_tilde,
            b_reduce_tilde,
            challenges_inv,
            challenges,
        )

    }

    pub fn verify_as_subprotocol(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, Vec<F>, Vec<F>) {


        let mut a_reduce_blind = G::zero();
        let mut b_reduce_blind = G::zero();
        
        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

    let mut flag = true; // default true; set to false only when errors are detected

        let pointer_old = trans_seq.pointer;
        
    println!("LiteBullet verify: pointer={}, expected commitments {:?} and size {}", 
                 pointer_old, self.v_com, self.length);
        
    // Ensure we don't access transcript out of bounds
        if pointer_old + 1 >= trans_seq.data.len() {
            println!("!! Invalid transcript when verifying LiteBullet: pointer out of range {} (need at least {} elements)",
                     pointer_old, pointer_old + 2);
            return (false, a_reduce_blind, b_reduce_blind, challenges_inv, challenges);
        }
        
        let first_elem = trans_seq.data[pointer_old].clone();
        let second_elem = trans_seq.data[pointer_old + 1].clone();
        
    println!("LiteBullet verify: transcript first elem {:?}, second elem {:?}", 
                 first_elem, second_elem);
        
    // Specify types explicitly to resolve type inference
        let expected_v_com = TranElem::<F, G>::Group(self.v_com);
        let expected_size = TranElem::<F, G>::Size(self.length);
        
        if (expected_v_com.clone(), expected_size.clone()) != (first_elem.clone(), second_elem.clone()) {
            println!("!! Invalid public input when verifying LiteBullet");
            println!("Expected: {:?}, got: {:?}", expected_v_com, first_elem);
            println!("Expected: {:?}, got: {:?}", expected_size, second_elem);
            flag = false;
        } 


        let n = self.length;
        let log_n = (n as u64).ilog2() as usize;
        
    println!("LiteBullet verify: length={}, log_n={}", n, log_n);

    // Pre-calculate required pointer movement and check bounds
        let expected_pointer = pointer_old + 3 * log_n + 4;
        if expected_pointer > trans_seq.data.len() {
            println!("!! Invalid transcript when verifying LiteBullet: expected pointer({}) out of range({})",
                     expected_pointer, trans_seq.data.len());
            flag = false;
        }
        
    // If earlier checks already failed, return early
        if !flag {
            return (flag, a_reduce_blind, b_reduce_blind, challenges_inv, challenges);
        }

    // Only set pointer after checks pass
        trans_seq.pointer = pointer_old + 2;
        
        let mut current_pointer = trans_seq.pointer;
        let mut lhs: G = self.v_com;
        

        for i in 0..log_n {
            println!("LiteBullet verify: iter {}/{}, current_pointer={}", i+1, log_n, current_pointer);
            
            // Check there are enough elements
            if current_pointer + 2 >= trans_seq.data.len() {
                println!("!! Invalid transcript when verifying LiteBullet: pointer {} out of range({})",
                         current_pointer + 2, trans_seq.data.len());
                flag = false;
                break;
            }

            if let (
                TranElem::Group(l_tr),
                TranElem::Group(r_tr),
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[current_pointer].clone(),
                trans_seq.data[current_pointer + 1].clone(),
                trans_seq.data[current_pointer + 2].clone(),
            ) {
                let x_j_inv = x_j.inverse().unwrap();
                lhs = lhs + l_tr.mul(x_j) + r_tr.mul(x_j_inv);
                challenges.push(x_j);
                challenges_inv.push(x_j_inv);
                println!("LiteBullet verify: successfully read challenge values (l_tr, r_tr, x_j)"); 

            } else {
                println!("!! Invalid transcript when verifying LiteBullet: wrong element types");
                println!("Element at {}: {:?}", current_pointer, trans_seq.data[current_pointer]);
                println!("Element at {}: {:?}", current_pointer+1, trans_seq.data[current_pointer+1]);
                println!("Element at {}: {:?}", current_pointer+2, trans_seq.data[current_pointer+2]);
                flag = false;
                break;
            }

            current_pointer += 3;
        }
        
        // If prior verification failed, return early
        if !flag {
            return (flag, a_reduce_blind, b_reduce_blind, challenges_inv, challenges);
        }
        
        // Check there are enough elements
        if current_pointer + 1 >= trans_seq.data.len() {
            println!("!! Invalid transcript when verifying LiteBullet: final pointer {} out of range({})",
                     current_pointer + 1, trans_seq.data.len());
            flag = false;
            return (flag, a_reduce_blind, b_reduce_blind, challenges_inv, challenges);
        }

        if let (
            TranElem::Group(a_reduce_com),
            TranElem::Group(b_reduce_com),
        ) = (
            trans_seq.data[current_pointer].clone(),
            trans_seq.data[current_pointer+1].clone(),
        ) {
            println!("LiteBullet verify: successfully read a_reduce_com and b_reduce_com");
            trans_seq.pointer = current_pointer + 2;

            let zk_mul = ZkMulScalar::new(
                srs,
                lhs,
                a_reduce_com,
                b_reduce_com,
            );
    
            println!("LiteBullet verify: calling ZkMulScalar.verify_as_subprotocol");
            let check = zk_mul.verify_as_subprotocol(
                trans_seq,
            );
            
            if !check {
                println!("!! ZkMulScalar verification failed");
                flag = false;
            } else {
                println!("LiteBullet verify: ZkMulScalar verification succeeded");
            }
            
            a_reduce_blind = a_reduce_com;
            b_reduce_blind = b_reduce_com;

        } else {
            println!("!! Invalid transcript when verifying LiteBullet: expected a_reduce_com and b_reduce_com mismatch");
            println!("Element at {}: {:?}", current_pointer, trans_seq.data[current_pointer]);
            println!("Element at {}: {:?}", current_pointer+1, trans_seq.data[current_pointer+1]);
            flag = false;
        }
        
        println!("LiteBullet verify end: flag={}, challenges_inv.len()={}, challenges.len()={}",
                 flag, challenges_inv.len(), challenges.len());
    
        (flag, a_reduce_blind, b_reduce_blind, challenges_inv, challenges)
        
    }

    pub fn verify(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>
    ) -> bool {

        if trans_seq.check_fiat_shamir() == false {
            println!("!! Fiat shamir check failed when verifying LiteBullet");
            return false;
        }

        return self.verify_as_subprotocol(srs, trans_seq).0;
    }

}


#[cfg(test)]
mod tests {
    
    use super::*;
    // use crate::utils::xi;

    #[test]
    fn test_litebullet() {

        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;
        use ark_poly_commit::smart_pc::SmartPC;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;
        

        let rng = &mut ark_std::rand::thread_rng();


        let logn = 6 as usize;
        let logsqrtn = logn/2;
        let n = 1 << logn;
        let a_vec= vec![(0..n).map(|_| 
            <E as Pairing>::ScalarField::rand(rng)
        ).collect()];
        let b_vec = vec![(0..n).map(|_| 
            Fr::rand(rng)
        ).collect()];

        let pp =
        SmartPC::<E>::setup(logn/2, rng).unwrap();

        let base = pp.g_0;
        let blind_base = pp.tilde_g;

        let zksrs =
        &ZkSRS::new(base, blind_base);


        let a_tilde =
        <E as Pairing>::ScalarField::rand(rng);
        let b_tiide =
        <E as Pairing>::ScalarField::rand(rng);
        
        let mat_a =
        linear::reshape_mat_cm_keep_projection(&a_vec);
        let mat_b =
        linear::reshape_mat_cm_keep_projection(&b_vec);

        let com_a =
            SmartPC::<E>::commit_full(&pp, &mat_a, a_tilde).unwrap();
        
        let com_b =
            SmartPC::<E>::commit_full(&pp, &mat_b, b_tiide).unwrap();

        let c = linear::inner_product(&a_vec[0], &b_vec[0]);

        let c_tilde =
        <E as Pairing>::ScalarField::rand(rng);

        let c_com = base.mul(c) + blind_base.mul(c_tilde);

        let ip =
        ZkLiteBullet::<F, G>::new(
            c_com,
            n,
        );

        let zk_trans_seq = &mut ZkTranSeq::new(zksrs);

        let (
            _,
            _,
            tilde_hat_a,
            tilde_hat_b,
            _,
            _
        ) = ip.reduce_prover(
            zksrs,
            zk_trans_seq,
            &a_vec[0],
            &b_vec[0],
            c_tilde,
        );

        let trans_seq =
        &mut zk_trans_seq.publish_trans();

        let (
            flag,
            hat_a_com,
            hat_b_com,
            xa,
            xb,
        ) = ip.verify_as_subprotocol(
            zksrs,
            trans_seq,
        );

        assert_eq!(flag, true);

        // let xi_a =
        // xi::xi_from_challenges::<F>(&xa);
        // let v_a =
        // linear::inner_product(&a_vec[0], &xi_a);
        // assert_eq!(base.mul(v_a) + blind_base.mul(tilde_hat_a), hat_a_com);

        let xal = xa[0..logsqrtn].to_vec();
        let xar = xa[logsqrtn..logn].to_vec();
        let xbl = xb[0..logsqrtn].to_vec();
        let xbr = xb[logsqrtn..logn].to_vec();
        
        let hat_h = pp.h_hat;
        let va_gt =
        E::pairing(hat_a_com, hat_h);
        let vb_gt =
        E::pairing(hat_b_com, hat_h);
        
        let pc_proof_a = SmartPC::<E>::open(
            &pp,
            &mat_a, 
            &xal,
            &xar,
            va_gt,
            com_a.0,
            &com_a.1,
            tilde_hat_a, 
            a_tilde,
        ).unwrap();

        let pc_proof_b = SmartPC::<E>::open(
            &pp,
            &mat_b, 
            &xbl,
            &xbr,
            vb_gt,
            com_b.0,
            &com_b.1,
            tilde_hat_b, 
            b_tiide,
        ).unwrap();

        let check_a = SmartPC::<E>::verify(
            &pp,
            com_a.0,
            va_gt,
            &xal,
            &xar,
            &pc_proof_a,
        ).unwrap();

        assert_eq!(check_a, true);

        let check_b = SmartPC::<E>::verify(
            &pp,
            com_b.0,
            vb_gt,
            &xbl,
            &xbr,
            &pc_proof_b,
        ).unwrap();

        assert_eq!(check_b, true);
     
        println!(" * Verification of LiteBullet passed");

        assert_eq!(trans_seq.pointer, trans_seq.data.len());

    }

    
}

