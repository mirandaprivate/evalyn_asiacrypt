//! Defines the data structures used in SMART-PC
//! use crate::*;
//! AdditiveGroup, AffineRepr
//!
// use ark_ff::ToConstraintField;
#![deny(missing_docs, trivial_numeric_casts, variant_size_differences)]
use ark_ff::Zero;
use ark_ec::pairing::{Pairing,PairingOutput};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress,
    SerializationError, Valid,
    Validate,
};
use ark_std::io::{Read, Write};


// `UniversalParams` are the universal parameters for the SMART-PC scheme.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
/// Universal structured reference string (SRS) parameters containing all group bases
/// and vectors required for commitments and openings in the SMART-PC scheme.
pub struct UniversalParams<E: Pairing> {
    /// Maximum supported matrix dimension / polynomial size upper bound.
    pub q: usize,
    /// Number of variables (e.g. multilinear variables or vector length).
    pub num_vars: usize,
    /// Generator (or normalized base) in G1.
    pub g_hat: E::G1,
    /// Generator (or normalized base) in G2.
    pub h_hat: E::G2,
    /// Randomization element s^h in G2.
    pub s_h: E::G2,
    /// Randomization element (s^q)^h in G2.
    pub sq_h: E::G2,
    /// Auxiliary random base in G1 (for encoding / hiding).
    pub nu_g: E::G1,
    /// Auxiliary random base in G2.
    pub nu_h: E::G2,
    /// Public base g_0 in G1.
    pub g_0: E::G1,
    /// Variant base \~g in G1 (used for hiding / separation).
    pub tilde_g: E::G1,
    /// Base element u = e(g, h) in GT.
    pub u: PairingOutput<E>,
    /// Independent GT base \~u used for randomness.
    pub tilde_u: PairingOutput<E>,
    /// Vector of G1 bases for primary commitments.
    pub vec_g: Vec<E::G1>,
    /// Vector of G2 bases for primary commitments.
    pub vec_h: Vec<E::G2>,
    /// G1 bases used in opening phase (g').
    pub vec_g_prime: Vec<E::G1>,
    /// Additional G1 bases for randomization / zero-knowledge (nu_g).
    pub vec_nu_g: Vec<E::G1>,
    /// Additional G2 bases for randomization / zero-knowledge (nu_h).
    pub vec_nu_h: Vec<E::G2>,
}



impl<E: Pairing> Valid for UniversalParams<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.q.check()?;
        self.num_vars.check()?;
        self.g_hat.check()?;
        self.h_hat.check()?;
        self.s_h.check()?;
        self.sq_h.check()?;
        self.nu_g.check()?;
        self.nu_h.check()?;
        self.g_0.check()?;
        self.tilde_g.check()?;
        self.u.check()?;
        self.tilde_u.check()?;
        self.vec_g.check()?;
        self.vec_h.check()?;
        self.vec_g_prime.check()?;
        self.vec_nu_g.check()?;
        self.vec_nu_h.check()?;
        Ok(())
    }
}


impl<E: Pairing> CanonicalSerialize for UniversalParams<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.q.serialize_with_mode(&mut writer, compress)?;
        self.num_vars.serialize_with_mode(&mut writer, compress)?;
        self.g_hat.serialize_with_mode(&mut writer, compress)?;
        self.h_hat.serialize_with_mode(&mut writer, compress)?;
        self.s_h.serialize_with_mode(&mut writer, compress)?;
        self.sq_h.serialize_with_mode(&mut writer, compress)?;
        self.nu_g.serialize_with_mode(&mut writer, compress)?;
        self.nu_h.serialize_with_mode(&mut writer, compress)?;
        self.g_0.serialize_with_mode(&mut writer, compress)?;
        self.tilde_g.serialize_with_mode(&mut writer, compress)?;
        self.u.serialize_with_mode(&mut writer, compress)?;
        self.tilde_u.serialize_with_mode(&mut writer, compress)?;
        self.vec_g.serialize_with_mode(&mut writer, compress)?;
        self.vec_h.serialize_with_mode(&mut writer, compress)?;
        self.vec_g_prime.serialize_with_mode(&mut writer, compress)?;
        self.vec_nu_g.serialize_with_mode(&mut writer, compress)?;
        self.vec_nu_h.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.q.serialized_size(compress)
            + self.num_vars.serialized_size(compress)
            + self.g_hat.serialized_size(compress)
            + self.h_hat.serialized_size(compress)
            + self.s_h.serialized_size(compress)
            + self.sq_h.serialized_size(compress)
            + self.nu_g.serialized_size(compress)
            + self.nu_h.serialized_size(compress)
            + self.g_0.serialized_size(compress)
            + self.tilde_g.serialized_size(compress)
            + self.u.serialized_size(compress)
            + self.tilde_u.serialized_size(compress)
            + self.vec_g.serialized_size(compress)
            + self.vec_h.serialized_size(compress)
            + self.vec_g_prime.serialized_size(compress)
            + self.vec_nu_g.serialized_size(compress)
            + self.vec_nu_h.serialized_size(compress)
    }
}

impl<E: Pairing> CanonicalDeserialize for UniversalParams<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let q = usize::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let num_vars = usize::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let g_hat = E::G1::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let h_hat = E::G2::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let s_h = E::G2::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let sq_h = E::G2::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let nu_g = E::G1::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let nu_h = E::G2::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let g_0 = E::G1::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let tilde_g = E::G1::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let u = PairingOutput::<E>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let tilde_u = PairingOutput::<E>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let vec_g = Vec::<E::G1>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let vec_h = Vec::<E::G2>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let vec_g_prime = Vec::<E::G1>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let vec_nu_g = Vec::<E::G1>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;
        let vec_nu_h = Vec::<E::G2>::deserialize_with_mode(
            &mut reader, compress, Validate::No)?;

        let result = Self {
            q,
            num_vars,
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
        if let Validate::Yes = validate {
            result.check()?;
        }

        Ok(result)
    }
}


// `Trans` are the transcript for the SMART-PC scheme.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
/// Transcript capturing intermediate Bulletproof and Schnorr values after Fiat-Shamir.
pub struct Trans<E: Pairing> {
    /// Bulletproofs L vector (with hiding factors).
    pub vec_l_tilde: Vec<PairingOutput<E>>,
    /// Bulletproofs R vector (with hiding factors).
    pub vec_r_tilde: Vec<PairingOutput<E>>,
    /// Hiding commitment to hat_a.
    pub com_rhs_tilde: PairingOutput<E>,
    /// Commitment component V_G in G1.
    pub v_g: E::G1,
    /// Commitment component V_H in G2.
    pub v_h: E::G2,
    /// Extended component V_G' in G1.
    pub v_g_prime: E::G1,
    /// Extended component V_H' in G2.
    pub v_h_prime: E::G2,
    /// Intermediate commitment W_G in G1.
    pub w_g: E::G1,
    /// Intermediate commitment W_H in G1.
    pub w_h: E::G1,
    /// First Schnorr proof group element f.
    pub schnorr_1_f: PairingOutput<E>,
    /// First Schnorr proof scalar z.
    pub schnorr_1_z: E::ScalarField,
    /// Second Schnorr proof group element f.
    pub schnorr_2_f: PairingOutput<E>,
    /// Second Schnorr proof scalar z1.
    pub schnorr_2_z_1: E::ScalarField,
    /// Second Schnorr proof scalar z2.
    pub schnorr_2_z_2: E::ScalarField,
}

impl<E: Pairing> Trans<E> {
    /// Create an empty transcript (vectors cleared, group elements set to default, scalars zero).
    pub fn new() -> Self {
        Self {
            vec_l_tilde: Vec::new(),
            vec_r_tilde: Vec::new(),
            com_rhs_tilde: PairingOutput::<E>::default(),
            v_g: <E::G1 as Default>::default(),
            v_h: <E::G2 as Default>::default(),
            v_g_prime: <E::G1 as Default>::default(),
            v_h_prime: <E::G2 as Default>::default(),
            w_g: <E::G1 as Default>::default(),
            w_h: <E::G1 as Default>::default(),
            schnorr_1_f: PairingOutput::<E>::default(),
            schnorr_1_z: E::ScalarField::zero(),
            schnorr_2_f: PairingOutput::<E>::default(),
            schnorr_2_z_1: E::ScalarField::zero(),
            schnorr_2_z_2: E::ScalarField::zero(),
        }
    }
}

impl<E: Pairing> Valid for Trans<E> {


    fn check(&self) -> Result<(), SerializationError> {
        self.vec_l_tilde.check()?;
        self.vec_r_tilde.check()?;
        self.com_rhs_tilde.check()?;
        self.v_g.check()?;
        self.v_h.check()?;
        self.v_g_prime.check()?;
        self.v_h_prime.check()?;
        self.w_g.check()?;
        self.w_h.check()?;
        self.schnorr_1_f.check()?;
        self.schnorr_1_z.check()?;
        self.schnorr_2_f.check()?;
        self.schnorr_2_z_1.check()?;
        self.schnorr_2_z_2.check()?;
        Ok(())
    }
}

impl<E: Pairing> CanonicalSerialize for Trans<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.vec_l_tilde.serialize_with_mode(&mut writer, compress)?;
        self.vec_r_tilde.serialize_with_mode(&mut writer, compress)?;
        self.com_rhs_tilde.serialize_with_mode(&mut writer, compress)?;
        self.v_g.serialize_with_mode(&mut writer, compress)?;
        self.v_h.serialize_with_mode(&mut writer, compress)?;
        self.v_g_prime.serialize_with_mode(&mut writer, compress)?;
        self.v_h_prime.serialize_with_mode(&mut writer, compress)?;
        self.w_g.serialize_with_mode(&mut writer, compress)?;
        self.w_h.serialize_with_mode(&mut writer, compress)?;
        self.schnorr_1_f.serialize_with_mode(&mut writer, compress)?;
        self.schnorr_1_z.serialize_with_mode(&mut writer, compress)?;
        self.schnorr_2_f.serialize_with_mode(&mut writer, compress)?;
        self.schnorr_2_z_1.serialize_with_mode(&mut writer, compress)?;
        self.schnorr_2_z_2.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.vec_l_tilde.serialized_size(compress)
            + self.vec_r_tilde.serialized_size(compress)
            + self.com_rhs_tilde.serialized_size(compress)
            + self.v_g.serialized_size(compress)
            + self.v_h.serialized_size(compress)
            + self.v_g_prime.serialized_size(compress)
            + self.v_h_prime.serialized_size(compress)
            + self.w_g.serialized_size(compress)
            + self.w_h.serialized_size(compress)
            + self.schnorr_1_f.serialized_size(compress)
            + self.schnorr_1_z.serialized_size(compress)
            + self.schnorr_2_f.serialized_size(compress)
            + self.schnorr_2_z_1.serialized_size(compress)
            + self.schnorr_2_z_2.serialized_size(compress)
    }
}