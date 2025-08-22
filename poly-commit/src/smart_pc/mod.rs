use std::time::Instant;

// use core::num;

use crate::Error;
use ark_ec::pairing::{Pairing, PairingOutput};
    // scalar_mul::ScalarMul, AffineRepr, CurveGroup, VariableBaseMSM
// };
use ark_ff::{
    Zero,
    One,
    Field,
    UniformRand,
};
// use ark_poly::DenseUVPolynomial;
use ark_std::{
    // format,
    marker::PhantomData,
    rand::RngCore,
    ops::{Add, Mul},
    vec::Vec,
};

use crate::MyInt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ark_serialize::{CanonicalSerialize, Compress};

/// Data structures for the SMART PC
pub mod data_structures;
pub use data_structures::*;

/// utilitiy functions
pub mod utils;

pub mod fiat_shamir;
pub use fiat_shamir::*;
use xi::{xi_from_challenges, xi_ip_from_challenges};
use utils::{
    inner_product, msm_g1, inner_pairing_product, prepare_base_short_g1, msm_g1_short_myint,
    boolean_msm_g1, proj_left, proj_right, scalar_mul_vec_zp, add_vec_zp, scalar_mul_vec_g1,
    add_vec_g1, scalar_mul_vec_g2, add_vec_g2, convert_myint_to_scalar_mat,
    convert_boolean_to_scalar_mat, test_utils, is_zero_vec,
    proj_left_myint, proj_right_myint,
};

/// subprotocols for the SMART PC
pub mod sub_protocols;
// use sub_protocols::*;

pub mod xi;
/// SMART_PC is an implementation of modified Dory
pub struct SmartPC<E: Pairing> {
    _engine: PhantomData<E>,
}

impl<E> SmartPC<E>
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{
    /// Constructs public parameters when given as input the maximum degree `degree`
    /// for the polynomial commitment scheme.
    ///
    pub fn setup<R: RngCore>(
        qlog: usize,
        rng: &mut R,
    ) -> Result<UniversalParams<E>, Error> {
        if qlog < 1 {
            return Err(Error::DegreeIsZero);
        }

        let setup_time =
        start_timer!(||
            format!("Smart-PC::Setup with dimension 2^{}, 2^{} coefficients",
            qlog,
            2 * qlog)
        );

        let q = 1 << qlog;
        
        let nu = E::ScalarField::rand(rng);
        let hat_s = E::ScalarField::rand(rng);
        let g_hat = E::G1::rand(rng);
        let h_hat = E::G2::rand(rng);


        // powers_of_s = [1, s, ..., s^(q-1)], len = q
        let mut powers_of_s = vec![E::ScalarField::one()];
        let mut cur = hat_s;
        for _ in 0..(q-1) {
            powers_of_s.push(cur);
            cur *= &hat_s;
        }

        let hat_sq = cur;

        assert_eq!(hat_sq, hat_s.pow([q as u64]));

        let mut powers_of_sq = vec![E::ScalarField::one()];
        cur = hat_sq;
        for _ in 0..(q-1) {
            powers_of_sq.push(cur);
            cur *= &hat_sq;
        }

        #[cfg(feature = "parallel")]
        let nu_powers_of_s = powers_of_s.clone()
        .into_par_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let nu_powers_of_s = powers_of_s.clone()
        .into_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        
        #[cfg(feature = "parallel")]
        let nu_powers_of_sq = powers_of_sq.clone()
        .into_par_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let nu_powers_of_sq = powers_of_sq.clone()
        .into_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        
        let s_h = h_hat.mul(hat_s);
        let sq_h = h_hat.mul(hat_sq);
        let nu_g = g_hat.mul(nu);
        let nu_h = h_hat.mul(nu);
        let hat_s_inv = hat_s.inverse().unwrap();
        let g_0 = g_hat.mul(hat_s_inv);
        let u = E::pairing(g_0, h_hat);
        let tilde_g =
            g_hat.mul(hat_s_inv *  hat_s_inv);
        let tilde_u = E::pairing(
            tilde_g, h_hat);
        
        #[cfg(feature = "parallel")]
        let vec_g = powers_of_s
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let vec_g = powers_of_s
        .into_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(feature = "parallel")]
        let vec_h = powers_of_sq.clone()
        .into_par_iter()
        .map(|x| h_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let vec_h = powers_of_sq.clone()
        .into_iter()
        .map(|x| h_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(feature = "parallel")]
        let vec_g_prime = powers_of_sq
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let vec_g_prime = powers_of_sq
        .into_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(feature = "parallel")]
        let vec_nu_g = nu_powers_of_s
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        
        #[cfg(not(feature = "parallel"))]
        let vec_nu_g = nu_powers_of_s
        .into_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        let vec_nu_h = nu_powers_of_sq
        .into_par_iter()
        .map(|x| h_hat.mul(x))
        .collect::<Vec<_>>();
        
        let pp = UniversalParams {
            q,
            num_vars: q,
            g_hat,
            h_hat,
            s_h,
            sq_h,
            nu_g,
            nu_h,
            g_0,
            tilde_g,
            u,
            tilde_u,
            vec_g,
            vec_h,
            vec_g_prime,
            vec_nu_g,
            vec_nu_h,
        };
        end_timer!(setup_time);
        Ok(pp)
    }


    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        // DEBUG: print generator pool sizes
        println!("[SmartPC::commit_full] vec_g_len={}, vec_h_len={}, input n={}, m={}", pp.vec_g.len(), pp.vec_h.len(), mat.len(), mat[0].len());
        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to {:?}-vector:",
            n * m
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = msm_g1::<E>(
                &vec_g, &col);
            tier_one_vec.push(commit_col);
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    
    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_myint_via_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        // DEBUG: print generator pool sizes
        // println!("[SmartPC::commit_myint_via_full] vec_g_len=2^{}, vec_h_len=2^{}, input n=2^{}, m=2^{}", pp.vec_g.len().ilog2(), pp.vec_h.len().ilog2(), mat.len().ilog2(), mat[0].len().ilog2());
        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to {:?}-vector:",
            n * m
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();


        let timer = Instant::now();
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            if is_zero_vec(&mat[i]) {
                tier_one_vec.push(E::G1::zero());
                continue;
            } else {
                let col = &mat[i].clone().into_iter()
                    .map(|x| E::ScalarField::from(x))
                    .collect::<Vec<_>>();
                let commit_col = msm_g1::<E>(
                    &vec_g, &col);
                tier_one_vec.push(commit_col);
            }
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }


    

    /// Commits after transforming into a square matrix
    pub fn commit_square_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> { 
        let square = convert_to_square::<E>(mat);
        Self::commit_full(pp, &square, hiding_factor)
    }

    /// Commits after transforming into a square matrix
    pub fn commit_square_myint(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> { 
        let square = convert_to_square_myint(mat);
        println!("[SmartPC::commit_square_myint] vec_g_len=2^{}, vec_h_len=2^{}, input n=2^{}, m=2^{}", pp.vec_g.len().ilog2(), pp.vec_h.len().ilog2(), square.len().ilog2(), square[0].len().ilog2());
        Self::commit_myint_via_full(pp, &square, hiding_factor)
    }


    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_short(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        hiding_factor: E::ScalarField,
        k: usize,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {

        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to {:?}-vector:",
            n * m
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let prepare_g =
            prepare_base_short_g1::<E>(&vec_g, k);
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = msm_g1_short_myint::<E>(
                &prepare_g, &col, k);
            tier_one_vec.push(commit_col);
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time in short bit: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    /// Outputs a commitment to a boolean matrix
    /// and intermediate tier-one commitment
    pub fn commit_boolean(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<bool>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {

        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to boolean vector with {:?} coefficients:",
            m * n
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = boolean_msm_g1::<E>(
                &vec_g, &col);
            tier_one_vec.push(commit_col);
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time in boolean: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn eval(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
    ) -> (PairingOutput<E>, E::ScalarField) {
        
        let rng = &mut ark_std::test_rng();
        
        let v_tilde = E::ScalarField::rand(rng);

        let l_vec = xi_from_challenges::<E>(&xl);
        let r_vec = xi_from_challenges::<E>(&xr);

        let la = proj_left::<E>(mat, &l_vec);
        let v = inner_product::<E>(&la, &r_vec);

        let com_v = pp.u.mul(&v) + pp.tilde_u.mul(&v_tilde);
        
        (com_v, v_tilde)
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open_square (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        let square = convert_to_square::<E>(mat);
        let (_m, n) = shape_to_square_shape((mat[0].len(), mat.len()));
        let xxxx = [xr.as_slice(), xl.as_slice()].concat();
        let log_n_new = (n as u64).ilog2() as usize;

        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::open(pp, &square, &xl_new, &xr_new, v_com, mat_com, tier1, v_tilde, mat_tilde)
    }

        /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open_square_myint (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        let square = convert_to_square_myint(mat);
        let (_m, n) = shape_to_square_shape((mat[0].len(), mat.len()));
        let xxxx = [xr.as_slice(), xl.as_slice()].concat();
        let log_n_new = (n as u64).ilog2() as usize;

        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::open_myint(pp, &square, &xl_new, &xr_new, v_com, mat_com, tier1, v_tilde, mat_tilde)
    }


    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        println!("[SmartPC::open] vec_g_len={}, vec_h_len={}, m={}, n={}", pp.vec_g.len(), pp.vec_h.len(), mat[0].len(), mat.len());
        let mut fs = FiatShamir::new();
        fs.push(&mat_com);
        fs.push(&v_com);

        let n = mat.len();
        let m = mat[0].len();
        start_timer!(|| format!(
            "Opening a {:?}-vector:",
            m * n
        ));

        let rng = &mut ark_std::test_rng();


        let x = fs.gen_challenge::<E>();

        let log_m = (m as u64).ilog2() as usize;
        let log_n = (n as u64).ilog2() as usize;

        let mut vec_l_tilde: Vec<PairingOutput<E>> = Vec::new();
        let mut vec_r_tilde: Vec<PairingOutput<E>>= Vec::new();

        let vec_l_hiding_factor: Vec<E::ScalarField> =
        (0..(log_m+log_n)).map(|_| E::ScalarField::rand(rng)).collect();
        let vec_r_hiding_factor: Vec<E::ScalarField> =
        (0..(log_m+log_n)).map(|_| E::ScalarField::rand(rng)).collect();

        let u_0 = E::pairing(pp.g_0, pp.h_hat);
        let u_tilde = pp.tilde_u;
        let l_vec: Vec<E::ScalarField> = xi_from_challenges::<E>(&xl);
        let r_vec: Vec<E::ScalarField> = xi_from_challenges::<E>(&xr);

        let mut capital_a_current = tier1[0..n].to_vec();
        let mut h_vec_current = pp.vec_h[0..n].to_vec();
        let mut r_current = r_vec[0..n].to_vec();
        
        let mut challenges_n: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_n: Vec<E::ScalarField> = Vec::new();

        let la: Vec<E::ScalarField> = proj_left::<E>(mat, &l_vec);

        let mut v_current = la.to_vec();

        for j in 0..log_n {

            // println!("Within proj proving iteration");


            let current_len = n / 2usize.pow(j as u32);
            

            let v_left = 
                v_current[0..current_len/2].to_vec();
            let v_right = 
                v_current[current_len/2..current_len].to_vec();

            let capital_a_left = 
                capital_a_current[0..current_len/2].to_vec();
            let capital_a_right = 
                capital_a_current[current_len/2..current_len].to_vec();
            
            let r_left = 
                r_current[0..current_len/2].to_vec();
            let r_right = 
                r_current[current_len/2..current_len].to_vec();
            

            let h_left = 
                h_vec_current[0..current_len/2].to_vec();
            let h_right = 
                h_vec_current[current_len/2..current_len].to_vec();

            let l_tr = 
                inner_pairing_product(&capital_a_left, &h_right).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_left, &r_right));
            let r_tr = 
                inner_pairing_product(&capital_a_right, &h_left).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_right, &r_left));

            let l_tr_tilde = l_tr + u_tilde.mul(&vec_l_hiding_factor[j]);
            let r_tr_tilde = r_tr + u_tilde.mul(&vec_r_hiding_factor[j]);

            vec_l_tilde.push(l_tr_tilde);
            vec_r_tilde.push(r_tr_tilde);

            fs.push(&l_tr_tilde);
            fs.push(&r_tr_tilde);
            
            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);

            v_current = add_vec_zp::<E>(
                &v_left,
                &scalar_mul_vec_zp::<E>(
                    &v_right, &x_j_inv),
            );

            capital_a_current = add_vec_g1::<E>(
                &capital_a_left,
                &scalar_mul_vec_g1::<E>(
                    &capital_a_right, &x_j_inv),
            );

            h_vec_current = add_vec_g2::<E>(
                &h_left,
                &scalar_mul_vec_g2::<E>(
                    &h_right, &x_j),
            );

            r_current = add_vec_zp::<E>(
                &r_left,
                &scalar_mul_vec_zp::<E>(
                    &r_right, &x_j),
            );

        }


        // let timer = Instant::now();

        let xi_n_inv = xi_from_challenges::<E>(&challenges_inv_n);
        
        let a_xi_inv = proj_right::<E>(mat, &xi_n_inv);

        // println!(" * Time for ket_zp: {:?}", timer.elapsed());

        let h_reduce = h_vec_current[0];
        let r_reduce = r_current[0];

        let mut a_current: Vec<E::ScalarField> = a_xi_inv.to_vec();
        
        let mut g_vec_current = pp.vec_g[0..m].to_vec();
        let mut l_current = l_vec[0..m].to_vec();
        
        let mut challenges_m: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_m: Vec<E::ScalarField> = Vec::new();
        

        for j in 0..log_m {

            // println!("Within scalar_proj proving iteration");


            let current_len = m / 2usize.pow(j as u32);
            
            let a_left = 
                a_current[0..current_len/2].to_vec();
            let a_right = 
                a_current[current_len/2..current_len].to_vec();
            
            let l_left = 
                l_current[0..current_len/2].to_vec();
            let l_right = 
                l_current[current_len/2..current_len].to_vec();
            

            let g_left = 
                g_vec_current[0..current_len/2].to_vec();
            let g_right = 
                g_vec_current[current_len/2..current_len].to_vec();

            let l_tr = 
                E::pairing(msm_g1::<E>(&g_right, &a_left), h_reduce).mul(&x)
                + u_0.mul(&r_reduce.mul(&inner_product::<E>(&a_left, &l_right)));
            let r_tr = 
                E::pairing(msm_g1::<E>(&g_left, &a_right), h_reduce).mul(&x)
                + u_0.mul(&r_reduce.mul(inner_product::<E>(&a_right, &l_left)));

            let l_tr_tilde = l_tr + u_tilde.mul(&vec_l_hiding_factor[j+log_n]);
            let r_tr_tilde = r_tr + u_tilde.mul(&vec_r_hiding_factor[j+log_n]);

            vec_l_tilde.push(l_tr_tilde);
            vec_r_tilde.push(r_tr_tilde);

            fs.push(&l_tr_tilde);
            fs.push(&r_tr_tilde);
            
            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);

            a_current = add_vec_zp::<E>(
                &a_left,
                &scalar_mul_vec_zp::<E>(
                    &a_right, &x_j_inv),
            );

            g_vec_current = add_vec_g1::<E>(
                &g_left,
                &scalar_mul_vec_g1::<E>(
                    &g_right, &x_j),
            );

            l_current = add_vec_zp::<E>(
                &l_left,
                &scalar_mul_vec_zp::<E>(
                    &l_right, &x_j),
            );

        }

        let a_reduce = a_current[0];
        let g_reduce = g_vec_current[0];
        
        // /////////////////////////////////////////////////////////////
        // Add Zero-Knowledge from now on
        // /////////////////////////////////////////////////////////////
        

        let xi_l =
        xi_ip_from_challenges::<E>(&xl, &challenges_m);
        let xi_r =
        xi_ip_from_challenges::<E>(&xr, &challenges_n);
        
            
        let base_rhs =  
            u_0.mul(&(xi_l * xi_r))
            + E::pairing(g_reduce.mul(&x), h_reduce);

        let rhs_tilde = E::ScalarField::rand(rng);

        let rhs_com = 
            base_rhs.mul(&a_reduce) + u_tilde.mul(&rhs_tilde);

        fs.push(&g_reduce);
        fs.push(&h_reduce);    
        fs.push(&rhs_com);
        
        
        let mut lhs_tilde = v_tilde + x * mat_tilde;

        // assert_eq!(current_index, 0);

        for j in 0..log_n {
            let l_tilde = vec_l_hiding_factor[j];
            let r_tilde = vec_r_hiding_factor[j];
            let x_j = challenges_n[j];
            let x_j_inv = challenges_inv_n[j];
            lhs_tilde = lhs_tilde + l_tilde * x_j + r_tilde * x_j_inv;
        }  

        for j in 0..log_m {
            let l_tilde = vec_l_hiding_factor[j+log_n];
            let r_tilde = vec_r_hiding_factor[j+log_n];
            let x_j = challenges_m[j];
            let x_j_inv = challenges_inv_m[j];
            lhs_tilde = lhs_tilde + l_tilde * x_j + r_tilde * x_j_inv;
        }  

        let eq_tilde = lhs_tilde - rhs_tilde;
    
        let eq_tilde_com = u_tilde.mul(&eq_tilde);

        let (v_g_prime, w_g) = 
        sub_protocols::pip_g1_prove::<E>(
            pp, 
            &challenges_m, 
            &mut fs,
        );

        let (v_h_prime, w_h) =
        sub_protocols::pip_g2_prove::<E>(
            pp, 
            &challenges_n, 
            &mut fs,
        );

        let (tr2, z1, z2) =
        sub_protocols::schnorr2_prove(
            pp,
            base_rhs,
            rhs_com,
            a_reduce,
            rhs_tilde,
            &mut fs,
        );


        let (tr1, z11) =
        sub_protocols::schnorr1_prove(
            pp,
            eq_tilde_com, 
            eq_tilde, 
            &mut fs,
        ); 



        let proof = Trans::<E> {
            vec_l_tilde: vec_l_tilde,
            vec_r_tilde: vec_r_tilde,
            com_rhs_tilde: rhs_com,
            v_g: g_reduce,
            v_h: h_reduce,
            v_g_prime: v_g_prime,
            v_h_prime: v_h_prime,
            w_g: w_g,
            w_h: w_h,
            schnorr_1_f: tr1,
            schnorr_1_z: z11,
            schnorr_2_f: tr2,
            schnorr_2_z_1: z1,
            schnorr_2_z_2: z2
        };

        Ok(proof)

    }
    
    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open_myint (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        println!("[SmartPC::open] vec_g_len={}, vec_h_len={}, m={}, n={}", pp.vec_g.len(), pp.vec_h.len(), mat[0].len(), mat.len());
        let mut fs = FiatShamir::new();
        fs.push(&mat_com);
        fs.push(&v_com);

        let n = mat.len();
        let m = mat[0].len();
        start_timer!(|| format!(
            "Opening a {:?}-vector:",
            m * n
        ));

        let rng = &mut ark_std::test_rng();


        let x = fs.gen_challenge::<E>();

        let log_m = (m as u64).ilog2() as usize;
        let log_n = (n as u64).ilog2() as usize;

        let mut vec_l_tilde: Vec<PairingOutput<E>> = Vec::new();
        let mut vec_r_tilde: Vec<PairingOutput<E>>= Vec::new();

        let vec_l_hiding_factor: Vec<E::ScalarField> =
        (0..(log_m+log_n)).map(|_| E::ScalarField::rand(rng)).collect();
        let vec_r_hiding_factor: Vec<E::ScalarField> =
        (0..(log_m+log_n)).map(|_| E::ScalarField::rand(rng)).collect();

        let u_0 = E::pairing(pp.g_0, pp.h_hat);
        let u_tilde = pp.tilde_u;
        let l_vec: Vec<E::ScalarField> = xi_from_challenges::<E>(&xl);
        let r_vec: Vec<E::ScalarField> = xi_from_challenges::<E>(&xr);

        let mut capital_a_current = tier1[0..n].to_vec();
        let mut h_vec_current = pp.vec_h[0..n].to_vec();
        let mut r_current = r_vec[0..n].to_vec();
        
        let mut challenges_n: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_n: Vec<E::ScalarField> = Vec::new();

        let la: Vec<E::ScalarField> = proj_left_myint::<E>(mat, &l_vec);

        let mut v_current = la.to_vec();

        for j in 0..log_n {

            // println!("Within proj proving iteration");


            let current_len = n / 2usize.pow(j as u32);
            

            let v_left = 
                v_current[0..current_len/2].to_vec();
            let v_right = 
                v_current[current_len/2..current_len].to_vec();

            let capital_a_left = 
                capital_a_current[0..current_len/2].to_vec();
            let capital_a_right = 
                capital_a_current[current_len/2..current_len].to_vec();
            
            let r_left = 
                r_current[0..current_len/2].to_vec();
            let r_right = 
                r_current[current_len/2..current_len].to_vec();
            

            let h_left = 
                h_vec_current[0..current_len/2].to_vec();
            let h_right = 
                h_vec_current[current_len/2..current_len].to_vec();

            let l_tr = 
                inner_pairing_product(&capital_a_left, &h_right).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_left, &r_right));
            let r_tr = 
                inner_pairing_product(&capital_a_right, &h_left).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_right, &r_left));

            let l_tr_tilde = l_tr + u_tilde.mul(&vec_l_hiding_factor[j]);
            let r_tr_tilde = r_tr + u_tilde.mul(&vec_r_hiding_factor[j]);

            vec_l_tilde.push(l_tr_tilde);
            vec_r_tilde.push(r_tr_tilde);

            fs.push(&l_tr_tilde);
            fs.push(&r_tr_tilde);
            
            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);

            v_current = add_vec_zp::<E>(
                &v_left,
                &scalar_mul_vec_zp::<E>(
                    &v_right, &x_j_inv),
            );

            capital_a_current = add_vec_g1::<E>(
                &capital_a_left,
                &scalar_mul_vec_g1::<E>(
                    &capital_a_right, &x_j_inv),
            );

            h_vec_current = add_vec_g2::<E>(
                &h_left,
                &scalar_mul_vec_g2::<E>(
                    &h_right, &x_j),
            );

            r_current = add_vec_zp::<E>(
                &r_left,
                &scalar_mul_vec_zp::<E>(
                    &r_right, &x_j),
            );

        }


        // let timer = Instant::now();

        let xi_n_inv = xi_from_challenges::<E>(&challenges_inv_n);
        
        let a_xi_inv = proj_right_myint::<E>(mat, &xi_n_inv);

        // println!(" * Time for ket_zp: {:?}", timer.elapsed());

        let h_reduce = h_vec_current[0];
        let r_reduce = r_current[0];

        let mut a_current: Vec<E::ScalarField> = a_xi_inv.to_vec();
        
        let mut g_vec_current = pp.vec_g[0..m].to_vec();
        let mut l_current = l_vec[0..m].to_vec();
        
        let mut challenges_m: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_m: Vec<E::ScalarField> = Vec::new();
        

        for j in 0..log_m {

            // println!("Within scalar_proj proving iteration");


            let current_len = m / 2usize.pow(j as u32);
            
            let a_left = 
                a_current[0..current_len/2].to_vec();
            let a_right = 
                a_current[current_len/2..current_len].to_vec();
            
            let l_left = 
                l_current[0..current_len/2].to_vec();
            let l_right = 
                l_current[current_len/2..current_len].to_vec();
            

            let g_left = 
                g_vec_current[0..current_len/2].to_vec();
            let g_right = 
                g_vec_current[current_len/2..current_len].to_vec();

            let l_tr = 
                E::pairing(msm_g1::<E>(&g_right, &a_left), h_reduce).mul(&x)
                + u_0.mul(&r_reduce.mul(&inner_product::<E>(&a_left, &l_right)));
            let r_tr = 
                E::pairing(msm_g1::<E>(&g_left, &a_right), h_reduce).mul(&x)
                + u_0.mul(&r_reduce.mul(inner_product::<E>(&a_right, &l_left)));

            let l_tr_tilde = l_tr + u_tilde.mul(&vec_l_hiding_factor[j+log_n]);
            let r_tr_tilde = r_tr + u_tilde.mul(&vec_r_hiding_factor[j+log_n]);

            vec_l_tilde.push(l_tr_tilde);
            vec_r_tilde.push(r_tr_tilde);

            fs.push(&l_tr_tilde);
            fs.push(&r_tr_tilde);
            
            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);

            a_current = add_vec_zp::<E>(
                &a_left,
                &scalar_mul_vec_zp::<E>(
                    &a_right, &x_j_inv),
            );

            g_vec_current = add_vec_g1::<E>(
                &g_left,
                &scalar_mul_vec_g1::<E>(
                    &g_right, &x_j),
            );

            l_current = add_vec_zp::<E>(
                &l_left,
                &scalar_mul_vec_zp::<E>(
                    &l_right, &x_j),
            );

        }

        let a_reduce = a_current[0];
        let g_reduce = g_vec_current[0];
        
        // /////////////////////////////////////////////////////////////
        // Add Zero-Knowledge from now on
        // /////////////////////////////////////////////////////////////
        

        let xi_l =
        xi_ip_from_challenges::<E>(&xl, &challenges_m);
        let xi_r =
        xi_ip_from_challenges::<E>(&xr, &challenges_n);
        
            
        let base_rhs =  
            u_0.mul(&(xi_l * xi_r))
            + E::pairing(g_reduce.mul(&x), h_reduce);

        let rhs_tilde = E::ScalarField::rand(rng);

        let rhs_com = 
            base_rhs.mul(&a_reduce) + u_tilde.mul(&rhs_tilde);

        fs.push(&g_reduce);
        fs.push(&h_reduce);    
        fs.push(&rhs_com);
        
        
        let mut lhs_tilde = v_tilde + x * mat_tilde;

        // assert_eq!(current_index, 0);

        for j in 0..log_n {
            let l_tilde = vec_l_hiding_factor[j];
            let r_tilde = vec_r_hiding_factor[j];
            let x_j = challenges_n[j];
            let x_j_inv = challenges_inv_n[j];
            lhs_tilde = lhs_tilde + l_tilde * x_j + r_tilde * x_j_inv;
        }  

        for j in 0..log_m {
            let l_tilde = vec_l_hiding_factor[j+log_n];
            let r_tilde = vec_r_hiding_factor[j+log_n];
            let x_j = challenges_m[j];
            let x_j_inv = challenges_inv_m[j];
            lhs_tilde = lhs_tilde + l_tilde * x_j + r_tilde * x_j_inv;
        }  

        let eq_tilde = lhs_tilde - rhs_tilde;
    
        let eq_tilde_com = u_tilde.mul(&eq_tilde);

        let (v_g_prime, w_g) = 
        sub_protocols::pip_g1_prove::<E>(
            pp, 
            &challenges_m, 
            &mut fs,
        );

        let (v_h_prime, w_h) =
        sub_protocols::pip_g2_prove::<E>(
            pp, 
            &challenges_n, 
            &mut fs,
        );

        let (tr2, z1, z2) =
        sub_protocols::schnorr2_prove(
            pp,
            base_rhs,
            rhs_com,
            a_reduce,
            rhs_tilde,
            &mut fs,
        );


        let (tr1, z11) =
        sub_protocols::schnorr1_prove(
            pp,
            eq_tilde_com, 
            eq_tilde, 
            &mut fs,
        ); 



        let proof = Trans::<E> {
            vec_l_tilde: vec_l_tilde,
            vec_r_tilde: vec_r_tilde,
            com_rhs_tilde: rhs_com,
            v_g: g_reduce,
            v_h: h_reduce,
            v_g_prime: v_g_prime,
            v_h_prime: v_h_prime,
            w_g: w_g,
            w_h: w_h,
            schnorr_1_f: tr1,
            schnorr_1_z: z11,
            schnorr_2_f: tr2,
            schnorr_2_z_1: z1,
            schnorr_2_z_2: z2
        };

        Ok(proof)

    }

    /// Verify as a square matrix
    pub fn verify_square(
        pp: &UniversalParams<E>,
        mat_com: PairingOutput<E>,
        v_com: PairingOutput<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &Trans<E>,
    ) -> Result<bool, Error> {
        let m = (1 << xl.len()) as usize;
        let n = (1 << xr.len()) as usize;

        let (_m_new, n_new) = shape_to_square_shape((m, n));
        let log_n_new = (n_new as u64).ilog2() as usize;

        // Reconstruct the original challenge vector from xl and xr
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(xr); // right challenges first
        xxxx.extend_from_slice(xl); // then left challenges
        
        // Reorganize for the new square shape
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::verify(pp, mat_com, v_com, &xl_new, &xr_new, proof)
    }


    /// Verifies that `value` is the evaluation at `point` of the polynomial
    /// committed inside `comm`.
    pub fn verify(
        pp: &UniversalParams<E>,
        mat_com: PairingOutput<E>,
        v_com: PairingOutput<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &Trans<E>,
    ) -> Result<bool, Error> {
        let check_time = start_timer!(|| "Checking evaluation");

        let mut fs = FiatShamir::new();
        fs.push(&mat_com);
        fs.push(&v_com);

        let log_m = xl.len();
        let log_n = xr.len();

        let u_0 = pp.u;

        let x = fs.gen_challenge::<E>();
        
        let mut lhs: PairingOutput<E> = 
            v_com.add(mat_com.mul(&x));

        let vec_l_tilde = &proof.vec_l_tilde;
        let vec_r_tilde = &proof.vec_r_tilde;
        
        let mut challenges_n: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_n: Vec<E::ScalarField> = Vec::new();

        for j in 0..log_n {

            let l_tr = vec_l_tilde[j];
            let r_tr = vec_r_tilde[j];
            fs.push(&l_tr);
            fs.push(&r_tr);

            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();
            lhs = lhs + l_tr.mul(&x_j) + r_tr.mul(&x_j_inv);
            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);

        }

        let mut challenges_m: Vec<E::ScalarField> = Vec::new();
        let mut challenges_inv_m: Vec<E::ScalarField> = Vec::new();

        for j in 0..log_m {

            let l_tr = vec_l_tilde[log_n + j];
            let r_tr = vec_r_tilde[log_n + j];
            fs.push(&l_tr);
            fs.push(&r_tr);

            let x_j = fs.gen_challenge::<E>();
            let x_j_inv = x_j.inverse().unwrap();
            lhs = lhs + l_tr.mul(&x_j) + r_tr.mul(&x_j_inv);
            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);
        }


        let xi_l =
        xi_ip_from_challenges::<E>(&xl, &challenges_m);
        let xi_r =
        xi_ip_from_challenges::<E>(&xr, &challenges_n);
        
        let base_rhs =  
            u_0.mul(&(xi_l * xi_r))
            + E::pairing(proof.v_g.mul(&x), proof.v_h);
        let rhs_blind = proof.com_rhs_tilde;
        let eq_tilde_com = lhs - rhs_blind;
        
        let v_g = proof.v_g;
        let v_h = proof.v_h;

        fs.push(&v_g);
        fs.push(&v_h);    
        fs.push(&rhs_blind);
        

        let check1 = 
        sub_protocols::pip_g1_verify::<E>(
            pp, 
            &challenges_m, 
            v_g,
            proof.v_g_prime,
            proof.w_g,
            &mut fs,
        );

        let check2 =
        sub_protocols::pip_g2_verify::<E>(
            pp, 
            &challenges_n,
            v_h,
            proof.v_h_prime,
            proof.w_h, 
            &mut fs,
        );

        let check3 =
        sub_protocols::schnorr2_verify(
            pp,
            base_rhs,
            rhs_blind,
            (proof.schnorr_2_f,proof.schnorr_2_z_1,proof.schnorr_2_z_2),
            &mut fs,
        );

        let check4 =
        sub_protocols::schnorr1_verify(
            pp,
            eq_tilde_com, 
            (proof.schnorr_1_f,proof.schnorr_1_z), 
            &mut fs,
        ); 

        
        // println!(" check 1 {:?}", check1);
        // println!(" check 2 {:?}", check2);
        // println!(" check 3 {:?}", check3);
        // println!(" check 4 {:?}", check4);
        let result =  check1 && check2 && check3 && check4;

        end_timer!(check_time, || format!("Result: {}", result));
        Ok(result)
    }

}




/// Experiment for SMART PC
pub fn experiment_smart_template<E>(num_vars: usize)
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{

    let rng = &mut ark_std::test_rng();

    let start_setup = Instant::now();
    let pp = SmartPC::<E>::setup(num_vars/2, rng).unwrap();

    let srs_size = pp.serialized_size(Compress::Yes);

    println!("SRS size: {:?} B", srs_size);

    let setup_time = start_setup.elapsed().as_secs_f64();
    println!("Setup time: {:?} s", setup_time);

    let n: usize = 1 << (num_vars/2);
    let k = 8;
    #[cfg(feature = "parallel")]
    let mat: Vec<Vec<MyInt>> = (0..n).into_par_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        (0..n).into_iter().map(|_|{
            rng.gen_range(-128..128)
        }).collect()
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let mat: Vec<Vec<MyInt>> = (0..n).into_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        (0..n).into_iter().map(|_|{
            rng.gen_range(-128..128)
        }).collect()
    }).collect();

    let mat_scalar =
    convert_myint_to_scalar_mat::<E>(&mat);


    let hiding_factor = E::ScalarField::rand(rng);

    let start_commit = Instant::now();
    let comm_full =
        SmartPC::<E>::commit_short(&pp, &mat, hiding_factor, k).unwrap();
    let comm = comm_full.0;
    let commit_time = start_commit.elapsed().as_secs_f64();
    println!("Commit time: {:?} s", commit_time);


    let mut commit_writer = Vec::new();
    comm.serialize_compressed(&mut commit_writer).unwrap();
    let commit_size = commit_writer.len();
    println!("Commit size: {:?} B", commit_size);


    let xl =
    (0..num_vars/2).map(|_| E::ScalarField::rand(rng)).collect();
    let xr =
    (0..num_vars/2).map(|_| E::ScalarField::rand(rng)).collect();
    let (v_com, v_tilde) =
    SmartPC::<E>::eval(&pp, &mat_scalar, &xl, &xr);

    let start_open = Instant::now();
    let proof = SmartPC::<E>::open(
        &pp, &mat_scalar, &xl, &xr,
        v_com, comm_full.0, &comm_full.1, v_tilde, hiding_factor,
    );
    let proof = proof.unwrap();
    let open_time = start_open.elapsed().as_secs_f64();
    println!("Open time: {:?} s", open_time);

    let verify_time = Instant::now();
    let check = SmartPC::<E>::verify(
        &pp,
        comm,
        v_com,
        &xl,
        &xr,
        &proof,
    );
    let check = check.unwrap();
    let verify_time = verify_time.elapsed().as_secs_f64()*1000.0;
    // println!("Check: {:?}", check);
    println!("Verify time: {:?} ms", verify_time);

    let mut writer = Vec::new();
    proof.serialize_compressed(&mut writer).unwrap();
    let proof_size = writer.len();
    println!("Proof size: {:?} B", proof_size);

    println!(
        "SMART-PC  verified: {:?}",
        check
    );

    // Verify boolean matrix
    let boolean_mat = mat.iter().map(|row|{
        row.iter().map(|x| *x > 0).collect()
    }).collect();
    let boolean_mat_scalar =
    convert_boolean_to_scalar_mat::<E>(&boolean_mat);
    let start_commit_boolean = Instant::now();
    let comm_boolean =
        SmartPC::<E>::commit_boolean(&pp, &boolean_mat, hiding_factor).unwrap();
    let commit_time_boolean = start_commit_boolean.elapsed().as_secs_f64();
    println!("Commit time (boolean): {:?} s", commit_time_boolean);
    let (v_com, v_tilde) =
    SmartPC::<E>::eval(&pp, &boolean_mat_scalar, &xl, &xr);

    // let myint_mat_from_boolean: Vec<Vec<MyInt>> =
    // boolean_mat.iter().map(|row|{
    //     row.iter().map(|x| if *x {1} else {0}).collect()
    // }).collect();
    // let comm_check =
    // SmartPC::<E>::commit_short(&pp, &myint_mat_from_boolean, hiding_factor, k).unwrap();
    // assert_eq!(comm_boolean, comm_check);

    let proof = SmartPC::<E>::open(
        &pp, &boolean_mat_scalar, &xl, &xr,
        v_com, comm_boolean.0, &comm_boolean.1, v_tilde, hiding_factor,
    );
    let proof = proof.unwrap();

    let check = SmartPC::<E>::verify(
        &pp,
        comm_boolean.0,
        v_com,
        &xl,
        &xr,
        &proof,
    );
    let check = check.unwrap();
    println!(
        "SMART-PC (boolean)  verified: {:?}",
        check
    );
}

/// Test existing utils function
pub fn test_smart<E:Pairing>() {
    test_utils::<E>();
    // let mut rng = ark_std::test_rng();
    // let degree = 10;
    // let num_vars = 5;
    // let num_samples = 1;
    // let num_queries = 1;
    // let num_coefficients
}


/// Reshape a col major matrix into an (almost) square col major matrix
/// such that the projection keep unchanged
/// 
pub fn convert_to_square<E: Pairing>(
    a: &Vec<Vec<E::ScalarField>>,
) -> Vec<Vec<E::ScalarField>>
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1 << log_n_new;
    let m_new = m * n / n_new;

    // (0..n_new).into_iter().map(|j| {
    //     (0..m_new).into_par_iter().map(|i| {
    //         a[(i * n_new + j) % n][(i * n_new + j) / n]
    //     }).collect()
    // }).collect()

    (0..n_new).into_iter().map(|j| {       // j = new col
        (0..m_new).into_par_iter().map(|i| { // i = new row
            // Flattened index 'k' based on new matrix position (column-major)
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()
}

/// Reshape a col major matrix into a (almost) square col major matrix
/// such that the projection keep unchanged
/// 
pub fn convert_to_square_myint(
    a: &Vec<Vec<MyInt>>,
) -> Vec<Vec<MyInt>>
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1 << log_n_new;
    let m_new = m * n / n_new;

    (0..n_new).into_iter().map(|j| {       // j = new col
        (0..m_new).into_par_iter().map(|i| { // i = new row
            // Flattened index 'k' based on new matrix position (column-major)
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()
}

/// Compute the new shape after converting into a square matrix
pub fn shape_to_square_shape((m, n): (usize, usize)) -> (usize, usize) {
    let log_n_new = ((m * n).ilog2()/2) as usize;
    let n_new = 1 << log_n_new;
    let m_new = m * n / n_new;
    (m_new, n_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr as BlsFr};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_commit_square_matrix_workflow() {
        let mut rng = test_rng();
        
        // Test parameters
        let qlog = 12; // q = 4096
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();
        
        // Create a test matrix as a long vector (simulating concatenated matrices)
        let test_data = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(4u64)];
        let test_matrix = vec![test_data]; // 1×4 matrix
        
        println!("Original matrix shape: {}×{}", test_matrix.len(), test_matrix[0].len());
        
        
        // Generate random hiding factors
        let hiding_factor = BlsFr::rand(&mut rng);
        
        // Test commit_square_full
        let (commitment, tier1) = SmartPC::<Bls12_381>::commit_square_full(&pp, &test_matrix, hiding_factor).unwrap();
        println!("✓ Commitment succeeded");
            
        // Raw challenges sized for original (m_raw=1, n_raw=4) layout: we treat columns-length as rows after conversion
        // Original logical shape before square conversion: m_raw = test_matrix.len() = 1 (columns), n_raw = test_matrix[0].len() = 4 (rows)
        let m_raw = test_matrix[0].len(); // we stored data as single column-major row vector
        let n_raw = test_matrix.len();
        let log_rows = (m_raw as u64).ilog2() as usize; // will become part of left
        let log_cols = (n_raw as u64).ilog2() as usize; // zero here
        let xl_raw: Vec<BlsFr> = (0..log_rows).map(|_| BlsFr::rand(&mut rng)).collect();
        let xr_raw: Vec<BlsFr> = (0..log_cols).map(|_| BlsFr::rand(&mut rng)).collect();
        println!("Raw challenge vector lengths: xl_raw={}, xr_raw={}", xl_raw.len(), xr_raw.len());

        // Reproduce open_square transformation locally so evaluation aligns with commitment/open
        // 1. Convert to square
        let square = convert_to_square::<Bls12_381>(&test_matrix);
        // 2. Determine new right dimension bits (same formula as open_square)
        let (_m_new, n_new) = shape_to_square_shape((test_matrix[0].len(), test_matrix.len()));
        let log_n_new = (n_new as u64).ilog2() as usize;
        // 3. Concatenate original xr_raw || xl_raw then split
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&xr_raw); 
        xxxx.extend_from_slice(&xl_raw);
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();
        println!("Transformed challenge lengths: xl_new={}, xr_new={}", xl_new.len(), xr_new.len());

        // 4. Evaluate square matrix at transformed point using existing eval
        let (v_com, v_tilde) = SmartPC::<Bls12_381>::eval(&pp, &test_matrix, &xl_raw, &xr_raw);
        let (v_com_new, v_tilde_new) = SmartPC::<Bls12_381>::eval(&pp, &square, &xl_new, &xr_new);

        assert_eq!(v_com - v_com_new, pp.tilde_u.mul(v_tilde - v_tilde_new), "Evaluation mismatch after square conversion");

        // Test open_square - this should work with the original test_matrix
        let open_result = SmartPC::<Bls12_381>::open_square(
            &pp,
            &test_matrix, // original non-square (open_square converts internally)
            &xl_raw,
            &xr_raw,
            v_com,
            commitment,
            &tier1,
            v_tilde,
            hiding_factor,
        );
        
        match open_result {
            Ok(proof) => {
                println!("✓ Opening succeeded");
                
                // Test verify_square
                let verify_result = SmartPC::<Bls12_381>::verify_square(
                    &pp,
                    commitment,
                    v_com,
                    &xl_raw,
                    &xr_raw,
                    &proof,
                );
                
                match verify_result {
                    Ok(true) => println!("✓ Verification succeeded"),
                    Ok(false) => panic!("✗ Verification failed"),
                    Err(e) => panic!("✗ Verification error: {:?}", e),
                }
            },
            Err(e) => panic!("✗ Opening failed: {:?}", e),
        }
    }

    #[test]
    fn test_commit_square_challenge_vector_reorganization() {
        // Test the challenge vector reorganization logic in verify_square
        let original_xl = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64)]; // 3 bits -> 8 rows
        let original_xr = vec![]; // 0 bits -> 1 column
        
        let m = 1 << original_xl.len(); // 8
        let n = 1 << original_xr.len(); // 1
        println!("Original shape: {}×{}", m, n);
        
        let (m_new, n_new) = shape_to_square_shape((m, n));
        println!("New shape: {}×{}", m_new, n_new);
        
        // Should be: 8×1 -> total=8, log=3, so 4×2 (log_right=1, log_left=2)
        assert_eq!(m_new, 4);
        assert_eq!(n_new, 2);
        
        let log_n_new = (n_new as u64).ilog2() as usize;
        assert_eq!(log_n_new, 1);
        
        // Reconstruct challenge vector
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&original_xr); // empty
        xxxx.extend_from_slice(&original_xl); // [1, 2, 3]
        
        let xl_new = xxxx[log_n_new..].to_vec(); // should be [2, 3]
        let xr_new = xxxx[..log_n_new].to_vec(); // should be [1]
        
        println!("Original: xl={:?}, xr={:?}", original_xl, original_xr);
        println!("Reorganized: xl_new={:?}, xr_new={:?}", xl_new, xr_new);
        
        assert_eq!(xl_new.len(), 2);
        assert_eq!(xr_new.len(), 1);
        assert_eq!(xl_new, vec![BlsFr::from(2u64), BlsFr::from(3u64)]);
        assert_eq!(xr_new, vec![BlsFr::from(1u64)]);
    }

    #[test]
    fn test_commit_square_convert_to_square() {
        // Test the convert_to_square function
        let test_matrix = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]; // 1×8 matrix
        
        let square = convert_to_square_myint(&test_matrix);
        
        println!("Original: 1×8 -> Square: {}×{}", square.len(), square[0].len());
        
        // 8 total elements -> 4×2 or 2×4, should choose more balanced
        // log_total = 3, log_right = 3/2 = 1, so 4×2
        assert_eq!(square.len(), 2); // columns
        assert_eq!(square[0].len(), 4); // rows
        
        println!("Square matrix:");
        for i in 0..square[0].len() {
            for j in 0..square.len() {
                print!("{} ", square[j][i]);
            }
            println!();
        }
    }
}
