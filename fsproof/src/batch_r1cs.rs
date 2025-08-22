//! Batch constraints proof using Fiat-Shamir transform
//! 
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;

use mat::utils::matdef::DenseMatFieldCM;

use crate::poseidon::Poseidon;
use crate::fs_trans::FiatShamir;

/// Batch R1CS constraints proof using Fiat-Shamir transform
#[derive(Clone, Debug)]
pub struct BatchConstraints<F: PrimeField> {
    /// The Fiat-Shamir transcript
    pub fs_trans: FiatShamir<F>,
    /// R1CS constraint matrices in dense (column-major) form
    /// Shape: (#constraints, #variables)
    pub a_r1cs: DenseMatFieldCM<F>,
    pub b_r1cs: DenseMatFieldCM<F>,
    pub c_r1cs: DenseMatFieldCM<F>,
    /// Witness matrix (column major)
    pub witness_mat: DenseMatFieldCM<F>,
    pub ready: (bool, bool)
}

impl<F: PrimeField + Absorb + Send + Sync> BatchConstraints<F> {
    /// Create a new batch constraints proof
    pub fn new() -> Self {
        let fs_trans = FiatShamir::new().expect("Failed to create FiatShamir");

        Self {
            fs_trans,
            a_r1cs: DenseMatFieldCM::new(0, 0),
            b_r1cs: DenseMatFieldCM::new(0, 0),
            c_r1cs: DenseMatFieldCM::new(0, 0),
            witness_mat: DenseMatFieldCM::new(0, 0),
            ready: (false, false),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.ready == (true, true)
    }

    pub fn prepare_from_transcript(&mut self, fs_trans: &FiatShamir<F>) {
        self.fs_trans = fs_trans.clone();
        self.gen_r1cs_constraints();
        let fs_trans_clone = self.fs_trans.clone();
        if let Err(e) = self.gen_witness_from_fs_trans(&fs_trans_clone) {
            println!("Warning: Failed to generate witness: {}", e);
        }
        self.ready = (true, true);
    }

    pub fn gen_r1cs_constraints(&mut self) -> (DenseMatFieldCM<F>, DenseMatFieldCM<F>, DenseMatFieldCM<F>) {
        // Create a poseidon instance for R1CS generation
        let poseidon = match Poseidon::<F>::new_with_recommended_params() {
            Ok(p) => p,
            Err(e) => {
                println!("Failed to create Poseidon: {}", e);
                // Return empty matrices on error
                return (
                    DenseMatFieldCM::new(0, 0),
                    DenseMatFieldCM::new(0, 0),
                    DenseMatFieldCM::new(0, 0)
                );
            }
        };

        // Get R1CS constraint matrices from Poseidon
        let (a_sparse, b_sparse, c_sparse) = poseidon.gen_r1cs_sparse_mats_padded().unwrap();

        // Convert sparse mats to dense column-major matrices via built-in to_dense
        self.a_r1cs = a_sparse.to_dense();
        self.b_r1cs = b_sparse.to_dense();
        self.c_r1cs = c_sparse.to_dense();

        self.ready.0 = true;

        (self.a_r1cs.clone(), self.b_r1cs.clone(), self.c_r1cs.clone())
    }

    /// Set the Fiat-Shamir transcript and generate corresponding R1CS witnesses
    pub fn gen_witness_from_fs_trans(&mut self, fs_trans: &FiatShamir<F>) -> Result<(), Box<dyn std::error::Error>> {
        self.fs_trans = fs_trans.clone();

        // Create a poseidon instance for witness generation
        let poseidon = Poseidon::<F>::new_with_recommended_params()?;
     
        // Generate witness matrix from the Fiat-Shamir transcript
        let mut witness_mat_data = Vec::new();

        let proof_vec = self.fs_trans.get_proof_vec().to_vec();
        let state_vec = self.fs_trans.get_state_vec().to_vec();

        let mut cur_a = self.fs_trans.get_initial_state();
        let mut cur_b;
        let mut cur_c;

        // Generate witness for each step in the transcript
        for i in 0..proof_vec.len() {
            cur_b = proof_vec[i];
            cur_c = state_vec[i];

            // Verify that the hash computation is correct
            let computed_hash = poseidon.hash2to1(cur_a, cur_b);
            if computed_hash != cur_c {
                return Err(format!("Hash verification failed at step {}: expected {}, got {}", 
                                   i, cur_c, computed_hash).into());
            }

            // Get witness for this hash computation
            let cur_w = poseidon.gen_r1cs_witness_padded(cur_a, cur_b, cur_c)?;

            witness_mat_data.push(cur_w);
            cur_a = cur_c;
        }

        if witness_mat_data.is_empty() {
            // Handle empty transcript case
            self.witness_mat = DenseMatFieldCM::new(0, 0);
        } else {
            self.witness_mat = DenseMatFieldCM::from_data(witness_mat_data);
        }

        self.ready.1 = true;

        Ok(())
    }

    /// Check that all R1CS constraints are satisfied: (Aw) ∘ (Bw) = Cw
    ///
    pub fn check_constraints(&self) -> bool {
        if self.witness_mat.data.is_empty() {
            // Empty witness matrix - this is valid for empty transcripts
            println!("✅ Empty witness matrix - no constraints to verify");
            return true;
        }

    let num_constraints = self.a_r1cs.shape.0;
    let num_variables = self.a_r1cs.shape.1;
        let num_instances = self.witness_mat.shape.1;

        // Dimension checks
    if self.b_r1cs.shape.0 != num_constraints || self.c_r1cs.shape.0 != num_constraints {
            println!("❌ Inconsistent constraint matrix rows: A={}, B={}, C={}", 
                     num_constraints, self.b_r1cs.shape.0, self.c_r1cs.shape.0);
            return false;
        }

    if self.b_r1cs.shape.1 != num_variables || self.c_r1cs.shape.1 != num_variables {
            println!("❌ Inconsistent constraint matrix cols: A={}, B={}, C={}", 
                     num_variables, self.b_r1cs.shape.1, self.c_r1cs.shape.1);
            return false;
        }

        if self.witness_mat.shape.0 != num_variables {
            println!("❌ Witness matrix dimension mismatch: expected {} variables, got {}", 
                     num_variables, self.witness_mat.shape.0);
            return false;
        }

        println!("Checking R1CS constraints...");
        println!("- Constraints: {}", num_constraints);
        println!("- Variables: {}", num_variables);
        println!("- Instances: {}", num_instances);

    // Step 1: Compute AW (dense A × dense witness matrix W)
    println!("Computing AW (dense × dense)...");
    let aw = self.a_r1cs.par_mul(&self.witness_mat);

    // Step 2: Compute BW (dense B × dense witness matrix W)
    println!("Computing BW (dense × dense)...");
    let bw = self.b_r1cs.par_mul(&self.witness_mat);

    // Step 3: Compute CW (dense C × dense witness matrix W)
    println!("Computing CW (dense × dense)...");
    let cw = self.c_r1cs.par_mul(&self.witness_mat);

        // Step 4: Compute (AW) ∘ (BW) (Hadamard product)
        println!("Computing (AW) ∘ (BW)...");
    let aw_hadamard_bw = aw.par_hadamard(&bw);

        // Step 5: Check if (AW) ∘ (BW) = CW
        println!("Verifying constraints...");
        let mut total_failed = 0;

        for instance_idx in 0..num_instances {
            for constraint_idx in 0..num_constraints {
                if aw_hadamard_bw.data[instance_idx][constraint_idx] != cw.data[instance_idx][constraint_idx] {
                    if total_failed < 10 { // Limit output to avoid spam
                        println!("❌ Constraint {} failed for instance {}: {} ≠ {}", 
                                 constraint_idx, instance_idx,
                                 aw_hadamard_bw.data[instance_idx][constraint_idx], 
                                 cw.data[instance_idx][constraint_idx]);
                    }
                    total_failed += 1;
                }
            }
        }

        if total_failed == 0 {
            println!("✅ All R1CS constraints satisfied!");
            true
        } else {
            println!("❌ {} constraint(s) failed!", total_failed);
            if total_failed > 10 {
                println!("   (Only showing first 10 failures)");
            }
            false
        }
    }

    /// Verify the entire batch: both transcript consistency and R1CS constraints
    pub fn verify_batch(&self) -> bool {
        // First verify the Fiat-Shamir transcript
        if !self.fs_trans.verify_fs() {
            println!("❌ Fiat-Shamir transcript verification failed!");
            return false;
        }
        println!("✅ Fiat-Shamir transcript verification passed!");
        
        // Then verify R1CS constraints
        if !self.check_constraints() {
            println!("❌ R1CS constraint verification failed!");
            return false;
        }

        println!("✅ Batch verification passed!");
        true
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize { self.a_r1cs.shape.0 }

    /// Get the number of variables per constraint
    pub fn num_variables(&self) -> usize { self.a_r1cs.shape.1 }

    /// Get the number of instances (transcript length)
    pub fn num_instances(&self) -> usize {
        self.witness_mat.shape.1
    }

    /// Get a summary of the batch proof
    pub fn get_summary(&self) -> String {
        format!(
            "BatchConstraints Summary:\n\
             - Transcript length: {} steps\n\
             - Number of constraints: {}\n\
             - Number of variables: {}\n\
             - Number of instances: {}",
            self.fs_trans.proof_len(),
            self.num_constraints(),
            self.num_variables(),
            self.num_instances()
        )
    }

    // (Removed sparse-specific helper functions; dense operations handled via par_* methods)
}

// (Removed sparse_to_dense; using SparseFieldMat::to_dense())

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_std::UniformRand;

    #[test]
    fn test_batch_constraints_empty_transcript() {
        println!("=== Testing BatchConstraints with Empty Transcript ===");
        
        // Create empty Fiat-Shamir transcript
        let fs_trans = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(12345u64), BlsFr::from(12345u64))
            .expect("Failed to create FiatShamir");
        
        // Verify empty transcript is valid
        println!("Empty transcript verification: {}", fs_trans.verify_fs());
        assert!(fs_trans.verify_fs());
        
        let mut batch = BatchConstraints::<BlsFr>::new();
        
        // Set the empty transcript
        batch.gen_witness_from_fs_trans(&fs_trans)
            .expect("Failed to set FiatShamir transcript");
        
        // Should pass verification (empty case)
        assert!(batch.check_constraints());
        
        println!("✅ Empty transcript test passed!");
    }

    #[test]
    fn test_batch_constraints_simple_transcript() {
        println!("=== Testing BatchConstraints with Simple Transcript ===");
        
        // Create and build a simple Fiat-Shamir transcript
        let mut fs_trans = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(9999u64), BlsFr::from(9999u64))
            .expect("Failed to create FiatShamir");
        
        // Build transcript with some operations
        println!("Building transcript...");
        fs_trans.push(BlsFr::from(100u64));
        fs_trans.push(BlsFr::from(200u64));
        let _challenge1 = fs_trans.gen_challenge();
        fs_trans.push(BlsFr::from(300u64));
        let _challenge2 = fs_trans.gen_challenge();
        
        println!("Transcript length: {}", fs_trans.proof_len());
        println!("Transcript verification: {}", fs_trans.verify_fs());
        assert!(fs_trans.verify_fs());
        
        let mut batch = BatchConstraints::<BlsFr>::new();
        
        // Set the transcript and generate constraints
        batch.gen_witness_from_fs_trans(&fs_trans)
            .expect("Failed to set FiatShamir transcript");
        
        // Verify the batch
        assert!(batch.check_constraints());
        
        println!("✅ Simple transcript test passed!");
    }

    #[test]
    fn test_batch_constraints_complex_transcript() {
        println!("=== Testing BatchConstraints with Complex Transcript ===");
        
        // Create a more complex transcript
        let mut fs_trans = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(54321u64), BlsFr::from(54321u64))
            .expect("Failed to create FiatShamir");
        
        let mut rng = ark_std::rand::thread_rng();
        
        println!("Building complex transcript...");
        
        // Interleave pushes and challenges
        for i in 0..5 {
            // Push some random elements
            fs_trans.push(BlsFr::rand(&mut rng));
            fs_trans.push(BlsFr::from((i * 1000) as u64));
            
            // Generate challenges
            let _challenge = fs_trans.gen_challenge();
            
            if i % 2 == 0 {
                fs_trans.push(BlsFr::from((i * 500) as u64));
                let _challenge2 = fs_trans.gen_challenge();
            }
        }
        
        println!("Final transcript length: {}", fs_trans.proof_len());
        assert!(fs_trans.verify_fs());
        
        let mut batch = BatchConstraints::<BlsFr>::new();
        
        // Set the transcript
        batch.gen_witness_from_fs_trans(&fs_trans)
            .expect("Failed to set FiatShamir transcript");
        
        // Verify the batch
        assert!(batch.check_constraints());
        
        println!("✅ Complex transcript test passed!");
    }

    #[test]
    fn test_batch_constraints_large_transcript() {
        println!("=== Testing BatchConstraints with Large Transcript ===");
        
        let mut fs_trans = FiatShamir::<BlsFr>::new_with_state(BlsFr::from(77777u64), BlsFr::from(77777u64))
            .expect("Failed to create FiatShamir");
        
        let mut rng = ark_std::rand::thread_rng();
        
        // Build a large transcript
        for i in 0..50 {
            fs_trans.push(BlsFr::rand(&mut rng));
            
            if i % 10 == 9 {
                let _challenge = fs_trans.gen_challenge();
            }
        }
        
        // Add some final challenges
        for _ in 0..5 {
            let _challenge = fs_trans.gen_challenge();
        }
        
        println!("Large transcript length: {}", fs_trans.proof_len());
        assert!(fs_trans.verify_fs());
        
        let mut batch = BatchConstraints::<BlsFr>::new();
        batch.gen_witness_from_fs_trans(&fs_trans)
            .expect("Failed to set FiatShamir transcript");
        
        // This might take a moment for large transcripts
        println!("Verifying large batch...");
        assert!(batch.check_constraints());
        
        println!("✅ Large transcript test passed!");
    }

}