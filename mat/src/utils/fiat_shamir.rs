//! Utility functions for the adaptive Fiat-Shamir transformation
//! 
use ark_ff::PrimeField;
use ark_ff::field_hashers::{DefaultFieldHasher,HashToField};
use ark_serialize::CanonicalSerialize;
use sha2::{Sha256, Digest};

// Generate random field element
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
)]
pub struct FiatShamir {
    // The hash function used by the Fiat-Shamir transform
    pub hasher: Sha256,
}

impl FiatShamir {
    // Create a new Fiat-Shamir transform
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"zk-smart");
        Self { hasher}
    }

    // Update the state of the Fiat-Shamir transform
    pub fn push <T>(&mut self, data: &T)
    where
        T: CanonicalSerialize,
    {
        let mut data_bytes = Vec::new();
        data.serialize_compressed(&mut data_bytes).unwrap();
        self.hasher.update(data_bytes);
    }

    // Get the current state of the Fiat-Shamir transform
    pub fn gen_challenge <Fp: PrimeField> (&mut self)
    -> Fp {
        let hash_value: [u8; 32] = self.hasher.clone().finalize().into();

        let h2f = 
        <DefaultFieldHasher<Sha256> as HashToField<Fp>>::new(b"zk-smart");

        let values: [Fp; 1] = h2f.hash_to_field(&hash_value);
        let mut challenge = values[0];

        while challenge == Fp::zero() {
            self.hasher.update(b"zk-smart");
            let hash_value: [u8; 32] = self.hasher.clone().finalize().into();
            let values: [Fp; 1] = h2f.hash_to_field(&hash_value);
            challenge = values[0];
        }

        self.push::<Fp>(&challenge);

        challenge
    }

    // Get the current state of the Fiat-Shamir transform
    pub fn get_state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

}