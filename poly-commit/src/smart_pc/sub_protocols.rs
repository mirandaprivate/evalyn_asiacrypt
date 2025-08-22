#![deny(trivial_numeric_casts, variant_size_differences)]
/// This module contains utility functions for the smart polynomial commitment scheme.
use ark_ec::{pairing::{Pairing, PairingOutput},
};
use ark_ff::UniformRand;
// use ark_serialize::CanonicalSerialize;
use ark_std::ops::{Add, Mul};

use super::UniversalParams;
use super::xi;
use super::FiatShamir;
use super::utils;

/// Prove the Schnorr proof for the commitment.
pub fn schnorr1_prove<E:Pairing>(
    pp: &UniversalParams<E>,
    com: PairingOutput<E>,
    witness: E::ScalarField,
    fs: &mut FiatShamir,
) -> (PairingOutput<E>, E::ScalarField)
{
    fs.push::<PairingOutput<E>>(&com);

    let rng = &mut ark_std::test_rng();   
    let rand = E::ScalarField::rand(rng);

    let u_tilde = pp.tilde_u;

    let tr_1 = u_tilde.mul(rand);

    let challenge = fs.gen_challenge::<E>();

    let z = rand + challenge * witness;

    let tr_2 = z;
    
    (tr_1, tr_2)
}

/// Verify the Schnorr proof for the tilde commitment.
pub fn schnorr1_verify<E:Pairing>(
    pp: &UniversalParams<E>,
    com: PairingOutput<E>,
    tr: (PairingOutput<E>, E::ScalarField),
    fs: &mut FiatShamir,
) -> bool 
{
    fs.push::<PairingOutput<E>>(&com);

    let u_tilde = pp.tilde_u;
    
    let (tr_1, tr_2) = tr;

    let challenge = fs.gen_challenge::<E>();

    u_tilde.mul(tr_2) == tr_1.add(com.mul(challenge))
}

/// Prove the Schnorr proof for the commitment.
pub fn schnorr2_prove<E:Pairing>(
    pp: &UniversalParams<E>,
    base: PairingOutput<E>,
    com: PairingOutput<E>,
    witness: E::ScalarField,
    hiding_factor: E::ScalarField,
    fs: &mut FiatShamir,
) -> (PairingOutput<E>, E::ScalarField, E::ScalarField)
{
    fs.push::<PairingOutput<E>>(&com);

    let rng = &mut ark_std::test_rng();   
    let rand1 = E::ScalarField::rand(rng);
    let rand2 = E::ScalarField::rand(rng);

    let u_tilde = pp.tilde_u;

    let tr_1 =
    base.mul(rand1) + u_tilde.mul(rand2);

    let challenge = fs.gen_challenge::<E>();

    let z1 = rand1 + challenge * witness;
    let z2 = rand2 + challenge * hiding_factor;

    (tr_1, z1, z2)
}

/// Verify the Schnorr proof for the tilde commitment.
pub fn schnorr2_verify<E:Pairing>(
    pp: &UniversalParams<E>,
    base: PairingOutput<E>,
    com: PairingOutput<E>,
    tr: (PairingOutput<E>, E::ScalarField, E::ScalarField),
    fs: &mut FiatShamir,
) -> bool 
{
    fs.push::<PairingOutput<E>>(&com);

    let u_tilde = pp.tilde_u;
    
    let (tr_1, z1, z2) = tr;

    let challenge = fs.gen_challenge::<E>();

    base.mul(z1).add(u_tilde.mul(z2)) == tr_1.add(com.mul(challenge))
}

/// Prove V_G is correctly computed
/// 
pub fn pip_g1_prove<E: Pairing>(
    pp: &UniversalParams<E>,
    challenges: &Vec<E::ScalarField>,
    fs: &mut FiatShamir,
) -> (<E as Pairing>::G1, <E as Pairing>::G1)
{
    let s = fs.gen_challenge::<E>();
    let xi = xi::xi_from_challenges::<E>(challenges);
   
    let qotient_poly = xi::quotient_poly::<E>(&xi, s);

    let v_prime = utils::msm_g1::<E>(&pp.vec_nu_g, &xi);
    let w = utils::msm_g1::<E>(&pp.vec_g, &qotient_poly);
    
    (v_prime, w)
}

/// Prove V_G is correctly computed
/// 
pub fn pip_g1_verify<E: Pairing>(
    pp: &UniversalParams<E>,
    challenges: &Vec<E::ScalarField>,
    v: E::G1,
    v_prime: E::G1,
    w: E::G1,
    fs: &mut FiatShamir,
) -> bool
{
    let s = fs.gen_challenge::<E>();   
    let phi_s = xi::phi_s::<E>(s, &challenges);

    let g = pp.g_hat;
    let h = pp.h_hat;
    let nu_h = pp.nu_h;
    let hat_s_h = pp.s_h;

    let p1 = E::pairing(v, nu_h);
    let p2 = E::pairing(v_prime, h);
    let p3 = E::pairing(w, h.mul(&s) - hat_s_h);
    let p4 = E::pairing(g.mul(&phi_s) - v, h);

    p1 == p2 && p3 == p4
}


/// Prove V_H is correctly computed
/// 
pub fn pip_g2_prove<E: Pairing>(
    pp: &UniversalParams<E>,
    challenges: &Vec<E::ScalarField>,
    fs: &mut FiatShamir,
) -> (<E as Pairing>::G2, <E as Pairing>::G1)
{
    let s = fs.gen_challenge::<E>();
    let xi = xi::xi_from_challenges::<E>(&challenges);
    let qotient_poly = xi::quotient_poly::<E>(&xi, s);

    let v_prime = utils::msm_g2::<E>(&pp.vec_nu_h, &xi);
    let w = utils::msm_g1::<E>(&pp.vec_g_prime, &qotient_poly);
    
    (v_prime, w)
}

/// Prove V_H is correctly computed
/// 
pub fn pip_g2_verify<E: Pairing>(
    pp: &UniversalParams<E>,
    challenges: &Vec<E::ScalarField>,
    v: E::G2,
    v_prime: E::G2,
    w: E::G1,
    fs: &mut FiatShamir,
) -> bool
{
    let s = fs.gen_challenge::<E>();   
    let phi_s = xi::phi_s::<E>(s, &challenges);

    let g = pp.g_hat;
    let h = pp.h_hat;
    let nu_g = pp.nu_g;
    let hat_sq_h = pp.sq_h;

    let p1 = E::pairing(nu_g, v);
    let p2 = E::pairing(g, v_prime);
    let p3 = E::pairing(w, h.mul(&s) - hat_sq_h);
    let p4 = E::pairing(g, h.mul(&phi_s) - v);

    p1 == p2 && p3 == p4
}



#[cfg(test)]
mod tests{

    use super::*;
    use crate::smart_pc::SmartPC;
    use ark_bls12_381::Bls12_381;

    #[test]
    fn test_subprotocol_smart()
    {
        let rng = &mut ark_std::test_rng();
        type Fr = <Bls12_381 as Pairing>::ScalarField;
        
        let fs = &mut FiatShamir::new();

        let log_n = 5;

        let xs: Vec<Fr> = (0..log_n).map(|_| 
            Fr::rand(rng)
        ).collect::<Vec<Fr>>();

        let xi: Vec<Fr> = xi::xi_from_challenges::<Bls12_381>(&xs);
        
        let pp =
        SmartPC::<Bls12_381>::setup(log_n as usize, rng).unwrap();

        let v = utils::msm_g2::<Bls12_381>(&pp.vec_h, &xi);

        fs.push(&v);

        let (v_prime, w) =
        pip_g2_prove::<Bls12_381>(&pp, &xs, fs);

        let fs_verify = &mut FiatShamir::new();
        fs_verify.push(&v);
        let result = pip_g2_verify(&pp, &xs, v, v_prime, w, fs_verify);

        assert_eq!(result, true);

        let a = Fr::rand(rng);
        let tilde_a = Fr::rand(rng);
        let com_a = pp.u.mul(a) + pp.tilde_u.mul(tilde_a);

        let (tr_1, tr_2, tr_3) =
        schnorr2_prove::<Bls12_381>(&pp, pp.u, com_a, a, tilde_a, fs);
        
        let result_schnoor =
        schnorr2_verify(&pp, pp.u, com_a, (tr_1, tr_2, tr_3), fs_verify);

        assert_eq!(result_schnoor, true);

        let b = Fr::rand(rng);
        let com_b = pp.tilde_u.mul(b);
        let (tr_1, tr_2) =
        schnorr1_prove(&pp, com_b, b, fs);

        let result_schnoor1 =
        schnorr1_verify(&pp, com_b, (tr_1, tr_2), fs_verify);

        assert_eq!(result_schnoor1, true);

    }
}
