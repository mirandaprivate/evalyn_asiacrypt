//! Utility functions for the adaptive Fiat-Shamir transformation
//! 
use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

// Import FiatShamir from the correct path based on error message suggestion
use crate::FiatShamir;

#[derive(Debug, Clone)]
pub enum TransElem<F: PrimeField> {
    Challenge(F),
    Response(F),
} 

#[derive(Debug, Clone)]
pub struct Transcript<F: PrimeField> {
    pub trans_seq: Vec<TransElem<F>>,
    pub pointer: usize,
    pub fs: FiatShamir<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> Transcript<F> {
    /// Create a new transcript
    pub fn new(root:F) -> Self {
        let fs = FiatShamir::new_with_state(root, root).unwrap();
        let mut trans_seq_value = Vec::new();
        trans_seq_value.push(TransElem::Response(root)); // Initialize with a dummy Response

        Self {
            trans_seq: trans_seq_value,
            pointer: 1,
            fs,
        }
    }

    pub fn default() -> Self {
        Self::new(F::zero())
    }

    /// Generate a challenge from the transcript
    pub fn gen_challenge(&mut self) -> F {
        let challenge = self.fs.gen_challenge();
        self.trans_seq.push(TransElem::Challenge(challenge));
        self.pointer += 1;
        challenge
    }

    pub fn push_response(&mut self, response: F) {
        self.fs.push(response);
        self.trans_seq.push(TransElem::Response(response));
        self.pointer += 1;
    }

    pub fn reset_pointer(&mut self) {
        self.pointer = 1;
    }

    pub fn get_trans_seq(&self) -> Vec<F> {
        let mut seq = Vec::new();
        for elem in &self.trans_seq {
            match elem {
                TransElem::Challenge(ch) => seq.push(*ch),
                TransElem::Response(resp) => seq.push(*resp),
            }
        }
        seq
    }

    pub fn get_fs_proof_vec(&self) -> Vec<F> {
        self.fs.get_proof_vec().to_vec().clone() // Convert from &Vec<F> to Vec<F>
    }

    pub fn get_fs_state_vec(&self) -> Vec<F> {
        self.fs.get_state_vec().to_vec().clone() // Convert from &Vec<F> to Vec<F>
    }

    pub fn publish_trans(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new(); // Initialize the bytes vector
        let trans_seq = self.get_trans_seq();

        // Serialize each field element individually using the correct method
        for field_elem in trans_seq {
            field_elem.serialize_uncompressed(&mut bytes)?;
        }
        
        println!("Serialized transcript size: {:?} bytes", bytes.len());
        Ok(bytes)
    }

    pub fn get_proof_size_in_bytes(&self) -> usize {
        let bytes = self.publish_trans().unwrap();
        bytes.len()
    }

    pub fn get_at_position(&self, pointer: usize) -> F {
        match &self.trans_seq[pointer] {
            TransElem::Challenge(ch) => *ch,
            TransElem::Response(resp) => *resp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;

    #[test]
    fn test_transcript_creation() {
        let transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        assert_eq!(transcript.trans_seq.len(), 1);
        assert_eq!(transcript.pointer, 1);
        
        // Check pointer consistency
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Transcript creation test passed!");
    }

    #[test]
    fn test_transcript_gen_challenge() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        let challenge = transcript.gen_challenge();
        
        assert_eq!(transcript.trans_seq.len(), 2);
        assert_eq!(transcript.pointer, 2);
        
        // Check that the challenge was stored correctly
        match &transcript.trans_seq[1] {
            TransElem::Challenge(stored_challenge) => assert_eq!(*stored_challenge, challenge),
            _ => panic!("Expected Challenge element"),
        }
        
        // Check pointer consistency
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Generate challenge test passed!");
    }

    #[test]
    fn test_transcript_push_response() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        let response = BlsFr::from(123u64);
        
        transcript.push_response(response);
        assert_eq!(transcript.trans_seq.len(), 2);
        assert_eq!(transcript.pointer, 2);

        // Check that the response was stored correctly
        match &transcript.trans_seq[1] {
            TransElem::Response(stored_response) => assert_eq!(*stored_response, response),
            _ => panic!("Expected Response element"),
        }
        
        // Check pointer consistency
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Push response test passed!");
    }

    #[test]
    fn test_transcript_mixed_transcript_sequence() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Check consistency at start
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Generate a challenge
        let challenge = transcript.gen_challenge();
        
        // Check consistency after challenge
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Add a response
        let response = BlsFr::from(200u64);
        transcript.push_response(response);
        
        // Check consistency after response
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Final checks
        assert_eq!(transcript.trans_seq.len(), 3);
        assert_eq!(transcript.pointer, 3);
        
        match &transcript.trans_seq[1] {
            TransElem::Challenge(ch) => assert_eq!(*ch, challenge),
            _ => panic!("Expected Challenge at index 1"),
        }
        
        match &transcript.trans_seq[2] {
            TransElem::Response(resp) => assert_eq!(*resp, response),
            _ => panic!("Expected Response at index 2"),
        }
        
        // Final pointer consistency check
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Mixed transcript sequence test passed!");
    }

    #[test]
    fn test_transcript_multiple_challenges() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        let challenge1 = transcript.gen_challenge();
        // Check consistency after first challenge
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        let challenge2 = transcript.gen_challenge();
        // Check consistency after second challenge
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        let challenge3 = transcript.gen_challenge();
        // Check consistency after third challenge
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Challenges should be different (with very high probability)
        assert_ne!(challenge1, challenge2);
        assert_ne!(challenge2, challenge3);
        assert_ne!(challenge1, challenge3);
        
        assert_eq!(transcript.trans_seq.len(), 4);
        assert_eq!(transcript.pointer, 4);
        
        println!("Multiple challenges test passed!");
    }

    #[test]
    fn test_transcript_multiple_responses() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        let response1 = BlsFr::from(111u64);
        transcript.push_response(response1);
        // Check consistency after first response
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        let response2 = BlsFr::from(222u64);
        transcript.push_response(response2);
        // Check consistency after second response
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        let response3 = BlsFr::from(333u64);
        transcript.push_response(response3);
        // Check consistency after third response
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        assert_eq!(transcript.trans_seq.len(), 4);
        assert_eq!(transcript.pointer, 4);
        
        println!("Multiple responses test passed!");
    }

    #[test]
    fn test_transcript_get_proof_vec() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Add some elements to build a proof
        transcript.gen_challenge();
        transcript.push_response(BlsFr::from(2u64));
        
        let proof_vec = transcript.get_fs_proof_vec();
        
        // Check final pointer consistency
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Proof vector length: {}", proof_vec.len());
        println!("Get proof vector test passed!");
    }

    #[test]
    fn test_transcript_get_state_vec() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Add some elements to build state
        transcript.gen_challenge();
        transcript.push_response(BlsFr::from(20u64));
        
        let state_vec = transcript.get_fs_state_vec();
        
        // Check final pointer consistency
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("State vector length: {}", state_vec.len());
        println!("Get state vector test passed!");
    }

    #[test]
    fn test_transcript_publish_trans() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Add some elements
        let _challenge = transcript.gen_challenge();
        transcript.push_response(BlsFr::from(84u64));
        
        let serialized_result = transcript.publish_trans();
        
        // Check pointer consistency before serialization
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        match serialized_result {
            Ok(bytes) => {
                assert!(!bytes.is_empty(), "Serialized bytes should not be empty");
                println!("Serialized transcript to {} bytes", bytes.len());
            }
            Err(e) => panic!("Serialization failed: {}", e),
        }
        
        println!("Publish transcript test passed!");
    }

    #[test]
    fn test_transcript_large_transcript() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Add many elements to test performance and correctness
        for i in 0..100 {
            // Check consistency
            assert_eq!(transcript.pointer, transcript.trans_seq.len());
            
            if i % 10 == 0 {
                transcript.gen_challenge();
                // Check consistency after each challenge
                assert_eq!(transcript.pointer, transcript.trans_seq.len());
            }
            if i % 5 == 0 {
                transcript.push_response(BlsFr::from((i * 2) as u64));
                // Check consistency after each response
                assert_eq!(transcript.pointer, transcript.trans_seq.len());
            }
        }
        
        assert!(transcript.trans_seq.len() > 0);
        
        // Final consistency checks
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Test that we can still get proof and state vectors
        let proof_vec = transcript.get_fs_proof_vec();
        let state_vec = transcript.get_fs_state_vec();
        
        println!("Large transcript test passed with {} elements", transcript.trans_seq.len());
        println!("Proof vector: {} elements, State vector: {} elements", 
                 proof_vec.len(), state_vec.len());
        println!("Final pointers - transcript.pointer: {}", transcript.pointer);
    }

    #[test]
    fn test_transcript_empty_transcript_operations() {
        let transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Test operations on empty transcript
        let proof_vec = transcript.get_fs_proof_vec();
        let state_vec = transcript.get_fs_state_vec();
        
        // Check pointer consistency for empty transcript
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        println!("Empty transcript - Proof vector: {} elements, State vector: {} elements", 
                 proof_vec.len(), state_vec.len());
        
        // Test serialization of empty transcript
        let serialized_result = transcript.publish_trans();
        assert!(serialized_result.is_ok(), "Empty transcript serialization should succeed");
        
        println!("Empty transcript operations test passed!");
    }

    #[test]
    fn test_transcript_pointer_consistency_comprehensive() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        // Test various combinations of operations and check consistency at each step
        let operations = vec![
            ("challenge", BlsFr::zero()), // Value doesn't matter for challenge
            ("response", BlsFr::from(2u64)),
            ("challenge", BlsFr::zero()),
            ("response", BlsFr::from(5u64)),
            ("response", BlsFr::from(6u64)),
            ("challenge", BlsFr::zero()),
        ];
        
        for (i, (op_type, value)) in operations.iter().enumerate() {
            match *op_type {
                "challenge" => { transcript.gen_challenge(); },
                "response" => transcript.push_response(*value),
                _ => panic!("Unknown operation type"),
            }
            
            // Check consistency after each operation
            assert_eq!(transcript.pointer, transcript.trans_seq.len(), 
                      "Pointer mismatch after operation {} ({})", i, op_type);
            
            println!("After operation {}: pointer={}, trans_seq.len()={}", 
                     i, transcript.pointer, transcript.trans_seq.len());
        }
        
        println!("Comprehensive pointer consistency test passed!");
    }

    #[test]
    fn test_transcript_challenge_randomness_with_consistency() {
        let mut transcript: Transcript<BlsFr> = Transcript::new(BlsFr::from(0u64));
        
        let challenge1 = transcript.gen_challenge();
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        let challenge2 = transcript.gen_challenge();
        assert_eq!(transcript.pointer, transcript.trans_seq.len());
        
        // Challenges should be different due to transcript state
        assert_ne!(challenge1, challenge2, "Challenges should be different");
        
        println!("Challenge randomness with consistency test passed!");
    }
}

