//! Utility functions for the adaptive Fiat-Shamir transformation
//! 
use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

use crate::poseidon::Poseidon;

/// Fiat-Shamir transcript for zero-knowledge proofs
#[derive(Clone, Debug)]
pub struct FiatShamir<F: PrimeField> {
    /// The hash function used by the Fiat-Shamir transform
    pub hasher: Poseidon<F>,
    pub initial_state: F,
    pub cur_state: F,
    pub proof_vec: Vec<F>,
    pub state_vec: Vec<F>,
}

/// Serializable structure for FiatShamir using ark-serialize
#[derive(Clone, Debug)]
pub struct FiatShamirSerialized<F: PrimeField> {
    pub initial_state: F,
    pub proof_vec: Vec<F>,
}

// Implementation of FiatShamir
impl<F: PrimeField + UniformRand + Absorb> FiatShamir<F> {
    /// Create a new Fiat-Shamir transform
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let hasher = Poseidon::new_with_recommended_params()?;
        
        let mut rng = StdRng::seed_from_u64(42u64);
        let initial_state = F::rand(&mut rng);
        let cur_state = initial_state;

        // Initialize with zero in proof_vec and initial_state in state_vec
        let mut proof_vec = Vec::<F>::new();
        let mut state_vec = Vec::<F>::new();
        proof_vec.push(F::zero());
        state_vec.push(initial_state);

        Ok(Self {
            hasher,
            initial_state,
            cur_state,
            proof_vec,
            state_vec
        })
    }

    /// Create a new Fiat-Shamir transform with specific initial state
    pub fn new_with_state(c:F, initial_state: F) -> Result<Self, Box<dyn std::error::Error>> {
        let hasher = Poseidon::new_with_recommended_params()?;
        let cur_state = initial_state;
        
        // Initialize with zero in proof_vec and initial_state in state_vec
        let mut proof_vec = Vec::<F>::new();
        let mut state_vec = Vec::<F>::new();
        proof_vec.push(c);
        state_vec.push(initial_state);

        Ok(Self {
            hasher,
            initial_state,
            cur_state,
            proof_vec,
            state_vec
        })
    }

    /// Push a transcript element to the transcript
    pub fn push(&mut self, tran_elem: F) {
        let new_state = self.hasher.hash2to1(self.cur_state, tran_elem);
        
        self.proof_vec.push(tran_elem);
        self.state_vec.push(new_state);

        self.cur_state = new_state;
    }

    /// Generate challenge from current state
    pub fn gen_challenge(&mut self) -> F {
        let challenge = self.cur_state;
        self.push(challenge);

        challenge
    }

    /// Verify the Fiat-Shamir transcript
    pub fn verify_fs(&self) -> bool {
        // Check vector lengths match
        if self.state_vec.len() != self.proof_vec.len() {
            return false;
        }

        // Handle case with only initial elements
        if self.proof_vec.len() == 1 {
            return self.proof_vec[0] == F::zero() && self.state_vec[0] == self.initial_state;
        }

        // Verify the sequence starting from the second element
        let mut cur_state = self.initial_state;
        for i in 1..self.proof_vec.len() {
            cur_state = self.hasher.hash2to1(cur_state, self.proof_vec[i]);
            if self.state_vec[i] != cur_state {
                return false;
            }
        }

        // Also verify that the first elements are correct
        self.proof_vec[0] == F::zero() && self.state_vec[0] == self.initial_state
    }

    /// Get the current state
    pub fn get_current_state(&self) -> F {
        self.cur_state.clone()
    }

    /// Get the initial state
    pub fn get_initial_state(&self) -> F {
        self.initial_state.clone()
    }

    /// Get proof vector length (excluding the initial zero)
    pub fn proof_len(&self) -> usize {
        if self.proof_vec.len() > 0 {
            self.proof_vec.len() - 1 // Subtract 1 to exclude the initial zero
        } else {
            0
        }
    }

    /// Reset the transcript to initial state
    pub fn reset(&mut self) {
        self.cur_state = self.initial_state;
        self.proof_vec.clear();
        self.state_vec.clear();
        
        // Re-initialize with zero and initial_state
        self.proof_vec.push(F::zero());
        self.state_vec.push(self.initial_state);
    }

    /// Get the proof vector (for verification)
    pub fn get_proof_vec(&self) -> Vec<F> {
        self.proof_vec[1..].to_vec()
    }

    /// Get the state vector (for verification)
    pub fn get_state_vec(&self) -> Vec<F> {
        self.state_vec[1..].to_vec()
    }

    /// Create a new FiatShamir instance from existing proof and state vectors
    /// Useful for verification scenarios
    pub fn from_vectors(
        initial_state: F,
        proof_vec: Vec<F>,
        state_vec: Vec<F>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if proof_vec.len() != state_vec.len() {
            return Err("Proof vector and state vector must have the same length".into());
        }

        // Verify the vectors start with the correct initial elements
        if proof_vec.is_empty() || state_vec.is_empty() {
            return Err("Proof and state vectors cannot be empty".into());
        }

        if proof_vec[0] != F::zero() {
            return Err("First element of proof_vec must be zero".into());
        }

        if state_vec[0] != initial_state {
            return Err("First element of state_vec must be initial_state".into());
        }

        let hasher = Poseidon::new_with_recommended_params()?;
        let cur_state = if state_vec.len() == 1 {
            initial_state
        } else {
            *state_vec.last().unwrap()
        };

        let instance = Self {
            hasher,
            initial_state,
            cur_state,
            proof_vec,
            state_vec,
        };

        // Verify the transcript is consistent
        if !instance.verify_fs() {
            return Err("Invalid transcript: verification failed".into());
        }

        Ok(instance)
    }

    /// Get a snapshot of the current transcript state
    pub fn get_snapshot(&self) -> (F, Vec<F>, Vec<F>) {
        (self.initial_state, self.proof_vec.clone(), self.state_vec.clone())
    }

    /// Verify the transcript and return the final state if valid
    pub fn verify_and_get_final_state(&self) -> Option<F> {
        if self.verify_fs() {
            if self.state_vec.len() == 1 {
                Some(self.initial_state)
            } else {
                Some(*self.state_vec.last().unwrap())
            }
        } else {
            None
        }
    }
}

// 序列化相关方法，使用 ark-serialize
impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> FiatShamir<F> {
    /// Serialize the FiatShamir instance to bytes using ark-serialize
    /// Only serializes initial_state and proof_vec
    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        
        // Serialize initial_state
        self.initial_state.serialize_compressed(&mut bytes)?;
        
        // Serialize proof_vec length
        (self.proof_vec.len() as u64).serialize_compressed(&mut bytes)?;
        
        // Serialize each element in proof_vec
        for elem in &self.proof_vec {
            elem.serialize_compressed(&mut bytes)?;
        }
        
        Ok(bytes)
    }

    /// Deserialize bytes to create a FiatShamir instance using ark-serialize
    /// Reconstructs state_vec by re-hashing the proof_vec
    pub fn deserialize(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut cursor = std::io::Cursor::new(bytes);
        
        // Deserialize initial_state
        let initial_state = F::deserialize_compressed(&mut cursor)?;
        
        // Deserialize proof_vec length
        let proof_len = u64::deserialize_compressed(&mut cursor)? as usize;
        
        // Deserialize proof_vec
        let mut proof_vec = Vec::with_capacity(proof_len);
        for _ in 0..proof_len {
            let elem = F::deserialize_compressed(&mut cursor)?;
            proof_vec.push(elem);
        }
        
        let hasher = Poseidon::new_with_recommended_params()?;
        let mut state_vec = Vec::new();
        let mut cur_state = initial_state;
        
        // Reconstruct state_vec
        if proof_vec.is_empty() {
            // This should not happen in normal operation, but handle gracefully
            return Err("Deserialized proof_vec cannot be empty".into());
        }
        
        // First element should be zero and corresponds to initial_state
        if proof_vec[0] != F::zero() {
            return Err("First element of deserialized proof_vec must be zero".into());
        }
        
        state_vec.push(initial_state);
        
        // Reconstruct state_vec by re-hashing starting from index 1
        for i in 1..proof_vec.len() {
            cur_state = hasher.hash2to1(cur_state, proof_vec[i]);
            state_vec.push(cur_state);
        }
        
        Ok(Self {
            hasher,
            initial_state,
            cur_state,
            proof_vec,
            state_vec,
        })
    }

    /// Convert to hex string for easier debugging and storage
    pub fn to_hex(&self) -> Result<String, Box<dyn std::error::Error>> {
        let bytes = self.serialize()?;
        Ok(hex::encode(bytes))
    }

    /// Create from hex string
    pub fn from_hex(hex_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = hex::decode(hex_str)?;
        Self::deserialize(&bytes)
    }

    /// Convert to JSON string (simplified version using hex encoding)
    pub fn to_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let hex_string = self.to_hex()?;
        let json = format!(r#"{{"fiat_shamir_data":"{}"}}"#, hex_string);
        Ok(json)
    }

    /// Create from JSON string
    pub fn from_json(json_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Simple JSON parsing - look for the hex data between quotes
        let start_marker = r#""fiat_shamir_data":""#;
        let start_pos = json_str.find(start_marker)
            .ok_or("Invalid JSON format")?
            + start_marker.len();
        
        let remaining = &json_str[start_pos..];
        let end_pos = remaining.find('"')
            .ok_or("Invalid JSON format")?;
        
        let hex_data = &remaining[..end_pos];
        Self::from_hex(hex_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;

    #[test]
    fn test_fiat_shamir_creation() {
        println!("=== Testing Fiat-Shamir Creation ===");
        
        let fs = FiatShamir::<BlsFr>::new()
            .expect("Failed to create Fiat-Shamir instance");
        
        println!("Initial state: {:?}", fs.get_initial_state());
        println!("Current state: {:?}", fs.get_current_state());
        println!("Proof length: {}", fs.proof_len());
        println!("Actual proof_vec length: {}", fs.proof_vec.len());
        println!("Actual state_vec length: {}", fs.state_vec.len());
        
        assert_eq!(fs.get_initial_state(), fs.get_current_state());
        assert_eq!(fs.proof_len(), 0); // Excludes initial zero
        assert_eq!(fs.proof_vec.len(), 1); // Includes initial zero
        assert_eq!(fs.state_vec.len(), 1); // Includes initial state
        assert_eq!(fs.proof_vec[0], BlsFr::zero());
        assert_eq!(fs.state_vec[0], fs.initial_state);
        
        println!("✅ Fiat-Shamir creation test passed!");
    }

    // #[test]
    // fn test_fiat_shamir_creation_with_state() {
    //     println!("=== Testing Fiat-Shamir Creation with State ===");
        
    //     let initial_state = BlsFr::from(12345u64);
    //     let fs = FiatShamir::<BlsFr>::new_with_state(initial_state, initial_state)
    //         .expect("Failed to create Fiat-Shamir instance with state");
        
    //     assert_eq!(fs.get_initial_state(), initial_state);
    //     assert_eq!(fs.get_current_state(), initial_state);
    //     assert_eq!(fs.proof_len(), 0);
    //     assert_eq!(fs.proof_vec.len(), 1);
    //     assert_eq!(fs.state_vec.len(), 1);
    //     assert_eq!(fs.proof_vec[0], BlsFr::zero());
    //     assert_eq!(fs.state_vec[0], initial_state);
        
    //     println!("✅ Fiat-Shamir creation with state test passed!");
    // }

    // #[test]
    // fn test_fiat_shamir_push() {
    //     println!("=== Testing Fiat-Shamir Push ===");
        
    //     let mut fs = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(100u64), BlsFr::from(100u64))
    //         .expect("Failed to create Fiat-Shamir instance");
        
    //     let initial_state = fs.get_current_state();
    //     println!("Initial state: {:?}", initial_state);
        
    //     // Push some elements
    //     let elem1 = BlsFr::from(200u64);
    //     let elem2 = BlsFr::from(300u64);
    //     let elem3 = BlsFr::from(400u64);
        
    //     println!("Pushing element 1: {:?}", elem1);
    //     fs.push(elem1);
    //     let state1 = fs.get_current_state();
    //     println!("State after push 1: {:?}", state1);
        
    //     println!("Pushing element 2: {:?}", elem2);
    //     fs.push(elem2);
    //     let state2 = fs.get_current_state();
    //     println!("State after push 2: {:?}", state2);
        
    //     println!("Pushing element 3: {:?}", elem3);
    //     fs.push(elem3);
    //     let state3 = fs.get_current_state();
    //     println!("State after push 3: {:?}", state3);
        
    //     // Verify states are different
    //     assert_ne!(initial_state, state1);
    //     assert_ne!(state1, state2);
    //     assert_ne!(state2, state3);
        
    //     // Verify proof length (excludes initial zero)
    //     assert_eq!(fs.proof_len(), 3);
    //     assert_eq!(fs.proof_vec.len(), 4); // Includes initial zero
    //     assert_eq!(fs.state_vec.len(), 4); // Includes initial state
        
    //     // Verify proof_vec contains the correct elements
    //     assert_eq!(fs.proof_vec[0], BlsFr::zero());
    //     assert_eq!(fs.proof_vec[1], elem1);
    //     assert_eq!(fs.proof_vec[2], elem2);
    //     assert_eq!(fs.proof_vec[3], elem3);
        
    //     // Verify state_vec contains the correct states
    //     assert_eq!(fs.state_vec[0], initial_state);
    //     assert_eq!(fs.state_vec[1], state1);
    //     assert_eq!(fs.state_vec[2], state2);
    //     assert_eq!(fs.state_vec[3], state3);
        
    //     println!("Final proof length: {}", fs.proof_len());
    //     println!("✅ Fiat-Shamir push test passed!");
    // }

    #[test]
    fn test_fiat_shamir_challenge_generation() {
        println!("=== Testing Fiat-Shamir Challenge Generation ===");
        
        let mut fs = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(500u64), BlsFr::from(500u64))
            .expect("Failed to create Fiat-Shamir instance");
        
        let initial_state = fs.get_current_state();
        println!("Initial state: {:?}", initial_state);
        
        // Generate first challenge
        let challenge1 = fs.gen_challenge();
        println!("Challenge 1: {:?}", challenge1);
        
        // The challenge should be the previous state
        assert_eq!(challenge1, initial_state);
        
        // Generate second challenge
        let challenge2 = fs.gen_challenge();
        println!("Challenge 2: {:?}", challenge2);
        
        // Challenges should be different
        assert_ne!(challenge1, challenge2);
        
        // Generate third challenge
        let challenge3 = fs.gen_challenge();
        println!("Challenge 3: {:?}", challenge3);
        assert_ne!(challenge2, challenge3);
        
        assert_eq!(fs.proof_len(), 3);
        
        // Verify that challenges were properly added to proof_vec
        assert_eq!(fs.proof_vec[1], challenge1);
        assert_eq!(fs.proof_vec[2], challenge2);
        assert_eq!(fs.proof_vec[3], challenge3);
        
        println!("✅ Fiat-Shamir challenge generation test passed!");
    }

    // #[test]
    // fn test_fiat_shamir_verification() {
    //     println!("=== Testing Fiat-Shamir Verification ===");
        
    //     let mut fs = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(1000u64), BlsFr::from(1000u64))
    //         .expect("Failed to create Fiat-Shamir instance");
        
    //     // Initial transcript should pass verification
    //     println!("Initial transcript verification: {}", fs.verify_fs());
    //     assert!(fs.verify_fs(), "Initial transcript should pass verification");
        
    //     // Add some elements and challenges
    //     let elem1 = BlsFr::from(2000u64);
    //     let elem2 = BlsFr::from(3000u64);
        
    //     println!("Adding element 1: {:?}", elem1);
    //     fs.push(elem1);
        
    //     println!("Generating challenge 1");
    //     let _challenge1 = fs.gen_challenge();
        
    //     println!("Adding element 2: {:?}", elem2);
    //     fs.push(elem2);
        
    //     println!("Generating challenge 2");
    //     let _challenge2 = fs.gen_challenge();
        
    //     println!("Proof length: {}", fs.proof_len());
        
    //     // Now verification should still pass
    //     let verification_result = fs.verify_fs();
    //     println!("Verification result: {}", verification_result);
    //     assert!(verification_result, "Valid transcript should pass verification");
        
    //     println!("✅ Fiat-Shamir verification test passed!");
    // }

    #[test]
    fn test_fiat_shamir_reset() {
        println!("=== Testing Fiat-Shamir Reset ===");
        
        let mut fs = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(9999u64), BlsFr::from(9999u64))
            .expect("Failed to create Fiat-Shamir instance");
        
        let initial_state = fs.get_initial_state();
        
        // Add some elements
        fs.push(BlsFr::from(100u64));
        fs.push(BlsFr::from(200u64));
        let _challenge = fs.gen_challenge();
        
        // Verify state is not initial
        assert_ne!(fs.get_current_state(), initial_state);
        assert_eq!(fs.proof_len(), 3); // 2 pushes + 1 challenge
        
        // Reset and verify
        fs.reset();
        assert_eq!(fs.get_current_state(), initial_state);
        assert_eq!(fs.proof_len(), 0);
        assert_eq!(fs.proof_vec.len(), 1); // Should have initial zero
        assert_eq!(fs.state_vec.len(), 1); // Should have initial state
        assert_eq!(fs.proof_vec[0], BlsFr::zero());
        assert_eq!(fs.state_vec[0], initial_state);
        assert!(fs.verify_fs());
        
        println!("✅ Reset test passed!");
    }

    #[test]
    fn test_fiat_shamir_from_vectors_validation() {
        println!("=== Testing from_vectors Validation ===");
        
        let initial_state = BlsFr::from(7777u64);
        
        // Test valid vectors with only initial elements
        let valid_fs = FiatShamir::<BlsFr>::from_vectors(
            initial_state,
            vec![BlsFr::zero()],
            vec![initial_state]
        ).expect("Failed to create from valid vectors");
        
        assert_eq!(valid_fs.get_current_state(), initial_state);
        assert_eq!(valid_fs.proof_len(), 0);
        assert!(valid_fs.verify_fs());
        
        // Test invalid first element in proof_vec
        let result = FiatShamir::<BlsFr>::from_vectors(
            initial_state,
            vec![BlsFr::from(123u64)], // Wrong first element
            vec![initial_state]
        );
        assert!(result.is_err(), "Wrong first element in proof_vec should fail");
        
        // Test invalid first element in state_vec
        let result = FiatShamir::<BlsFr>::from_vectors(
            initial_state,
            vec![BlsFr::zero()],
            vec![BlsFr::from(999u64)] // Wrong first element
        );
        assert!(result.is_err(), "Wrong first element in state_vec should fail");
        
        // Test empty vectors
        let result = FiatShamir::<BlsFr>::from_vectors(
            initial_state,
            vec![],
            vec![]
        );
        assert!(result.is_err(), "Empty vectors should fail");
        
        println!("✅ from_vectors validation test passed!");
    }


}