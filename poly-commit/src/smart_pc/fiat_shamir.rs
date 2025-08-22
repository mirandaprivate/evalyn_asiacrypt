#![deny(trivial_numeric_casts, variant_size_differences)]
//! Utility functions for the adaptive Fiat-Shamir transformation
//! 
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Sha256, Digest};

/// Generate random field element
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
)]
pub struct FiatShamir {
    /// The hash function used by the Fiat-Shamir transform
    pub hasher: Sha256,
}

impl FiatShamir {
    /// Create a new Fiat-Shamir transform
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"poly-commit");
        Self { hasher}
    }

    /// Update the state of the Fiat-Shamir transform
    pub fn push <T>(&mut self, data: &T)
    where
        T: CanonicalSerialize,
    {
        let mut data_bytes = Vec::new();
        data.serialize_compressed(&mut data_bytes).unwrap();
        self.hasher.update(data_bytes);
    }

    /// Get the current state of the Fiat-Shamir transform
    pub fn gen_challenge <E:Pairing> (&mut self) -> <E as Pairing>::ScalarField {
        let hash_value: [u8; 32] = self.hasher.clone().finalize().into();
        let challenge =
        E::ScalarField::from_le_bytes_mod_order(&hash_value);

        self.push::<E::ScalarField>(&challenge);

        challenge
    }

    /// Get the current state of the Fiat-Shamir transform
    pub fn get_state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

}