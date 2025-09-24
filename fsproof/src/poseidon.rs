//! Poseidon hash function implementation for zero-knowledge proofs
//! 
//! This module provides a Poseidon hash function implementation that can generate
//! R1CS constraints for use in zero-knowledge proof systems.

use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_crypto_primitives::{
    sponge::{
        poseidon::{PoseidonConfig, PoseidonSponge},
        Absorb, CryptographicSponge,
        constraints::CryptographicSpongeVar,
    },
};
use ark_relations::{
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, ConstraintSystem, Matrix},
};
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
};
use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;

use mat::utils::matdef::SparseFieldMat;

#[derive(Clone, Debug)]
pub struct Poseidon<F> where
    F: PrimeField,
{
    poseidon_config: PoseidonConfig<F>,
}

#[derive(Clone)]
pub struct PoseidonCircuit<F: PrimeField> {
    pub secret_a: Option<F>,
    pub secret_b: Option<F>,
    pub public_hash: Option<F>,
    pub poseidon_config: PoseidonConfig<F>,
}

// Implement ConstraintSynthesizer trait for PoseidonCircuit
impl<F: PrimeField + Absorb> ConstraintSynthesizer<F> for PoseidonCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // Allocate secret variables
        let a_var = FpVar::new_witness(cs.clone(), || {
            self.secret_a.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let b_var = FpVar::new_witness(cs.clone(), || {
            self.secret_b.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Allocate public input
        let hash_var = FpVar::new_input(cs.clone(), || {
            self.public_hash.ok_or(SynthesisError::AssignmentMissing)
        })?;
        
        // Use Poseidon gadget
        let mut sponge = PoseidonSpongeVar::new(cs.clone(), &self.poseidon_config);
        let input_vars = vec![a_var, b_var];
        sponge.absorb(&input_vars)?;
        let computed_hash_vec = sponge.squeeze_field_elements(1)?;
        let computed_hash = &computed_hash_vec[0];
        
        // Constraint: computed_hash == public_hash
        computed_hash.enforce_equal(&hash_var)?;
        
        Ok(())
    }
}

impl<F> Poseidon<F> 
where 
    F: PrimeField + UniformRand + Absorb,
{
    /// Create new Poseidon instance
    pub fn new(poseidon_config: PoseidonConfig<F>) -> Self {
        Self { poseidon_config }
    }

    /// Create new instance with recommended parameters
    pub fn new_with_recommended_params() -> Result<Self, Box<dyn std::error::Error>> {
        let poseidon_config = Self::gen_config_with_recommended_params()?;
        Ok(Self::new(poseidon_config))
    }

    /// Generate recommended Poseidon configuration
    fn gen_config_with_recommended_params() -> Result<PoseidonConfig<F>, Box<dyn std::error::Error>> {
        // Use parameters recommended by Poseidon paper
        let full_rounds = 8;
        let partial_rounds = 31;
        let alpha = 5;
        let rate = 2;
        let capacity = 1;
        let state_size = rate + capacity;

        // Create standard Cauchy MDS matrix
        let mut mds = vec![vec![F::zero(); state_size]; state_size];
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for i in 0..state_size {
            x_values.push(F::from((i + 1) as u64));
        }
        for i in 0..state_size {
            y_values.push(F::from((state_size + i + 1) as u64));
        }

        for i in 0..state_size {
            for j in 0..state_size {
                let diff = x_values[i] - y_values[j];
                mds[i][j] = diff.inverse().ok_or("Failed to compute inverse of Cauchy matrix element")?;
            }
        }

        // Create ARK round constants
        use ark_std::rand::rngs::StdRng;
        use ark_std::rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42u64);

        let total_rounds = full_rounds + partial_rounds;
        let mut ark = vec![vec![F::zero(); state_size]; total_rounds];

        for round in 0..total_rounds {
            for state_elem in 0..state_size {
                ark[round][state_elem] = F::rand(&mut rng);
            }
        }

        Ok(PoseidonConfig::new(
            full_rounds,
            partial_rounds,
            alpha,
            mds,
            ark,
            rate,
            capacity,
        ))
    }

    /// Compute Poseidon hash: hash(a, b) = c
    pub fn hash2to1(&self, a: F, b: F) -> F {
        let mut sponge = PoseidonSponge::<F>::new(&self.poseidon_config);
        sponge.absorb(&vec![a, b]); 
        let hash_result = sponge.squeeze_field_elements(1)[0];
        hash_result
    }

    /// Generate R1CS constraint matrices (A, B, C)
    /// Constraint: A * w ∘ B * w = C * w (where ∘ is Hadamard product)
    pub fn gen_r1cs_mats(&self) -> Result<(Matrix<F>, Matrix<F>, Matrix<F>), SynthesisError> {
        // Create constraint system
        let cs = ConstraintSystem::<F>::new_ref();
        
        // Create circuit with dummy values to generate constraint structure
        // Note: Matrix structure is independent of specific values, so we can use arbitrary values
        let dummy_circuit = PoseidonCircuit {
            secret_a: Some(F::zero()),
            secret_b: Some(F::zero()),
            public_hash: Some(F::zero()),
            poseidon_config: self.poseidon_config.clone(),
        };
        
        // Generate constraints
        dummy_circuit.generate_constraints(cs.clone())?;
        cs.finalize();
        
        // Convert to matrices
        let matrices = cs.to_matrices().ok_or(SynthesisError::Unsatisfiable)?;
        
        Ok((matrices.a, matrices.b, matrices.c))
    }

    /// Generate R1CS constraint matrices (A, B, C) in SparseMat format
    /// Constraint: A * w ∘ B * w = C * w (where ∘ is Hadamard product)
    /// Pads matrices to power of 2 dimensions if necessary
    pub fn gen_r1cs_sparse_mats_padded(&self) -> Result<(SparseFieldMat<F>, SparseFieldMat<F>, SparseFieldMat<F>), SynthesisError> {
        // Generate dense matrices first
        let (dense_a, dense_b, dense_c) = self.gen_r1cs_mats()?;
        
        // Get witness vector to determine the actual number of variables
        let test_witness = self.gen_r1cs_witness(F::zero(), F::zero(), F::zero())?;
        
        let num_constraints = dense_a.len();
        let num_variables = test_witness.len(); 
        
        println!("Original dimensions: {} constraints x {} variables (from witness length)", num_constraints, num_variables);
        
        // Calculate padded dimensions (next power of 2)
        let padded_rows = next_power_of_2(num_constraints);
        let padded_cols = next_power_of_2(num_variables);
        
        println!("Padded dimensions: {} x {}", padded_rows, padded_cols);
        
        // Convert to SparseMat format
        let sparse_a = dense_to_sparse(&dense_a, padded_rows, padded_cols)?;
        let sparse_b = dense_to_sparse(&dense_b, padded_rows, padded_cols)?;
        let sparse_c = dense_to_sparse(&dense_c, padded_rows, padded_cols)?;
        
        Ok((sparse_a, sparse_b, sparse_c))
    }

    /// Generate R1CS witness vector w
    /// witness contains all variable assignments: [instance variables, witness variables]
    pub fn gen_r1cs_witness(&self, a: F, b: F, c: F) -> Result<Vec<F>, SynthesisError> {
        // Create constraint system
        let cs = ConstraintSystem::<F>::new_ref();
        
        // Create circuit with concrete values
        let circuit = PoseidonCircuit {
            secret_a: Some(a),
            secret_b: Some(b),
            public_hash: Some(c),
            poseidon_config: self.poseidon_config.clone(),
        };
        
        // Generate constraints
        circuit.generate_constraints(cs.clone())?;
        cs.finalize();
        
        // Get variable assignments (refer to main function implementation)
        let cs_borrow = cs.borrow().unwrap();
        let instance_assignment = &cs_borrow.instance_assignment;
        let witness_assignment = &cs_borrow.witness_assignment;

        // Build complete variable vector: [instance variables, witness variables]
        let mut variable_values = Vec::with_capacity(
            instance_assignment.len() + witness_assignment.len()
        );
        variable_values.extend_from_slice(instance_assignment);
        variable_values.extend_from_slice(witness_assignment);
        
        Ok(variable_values)
    }

    /// Generate R1CS witness vector w (padded to power of 2 length)
    /// witness contains all variable assignments: [instance variables, witness variables]
    pub fn gen_r1cs_witness_padded(&self, a: F, b: F, c: F) -> Result<Vec<F>, SynthesisError> {
        let witness = self.gen_r1cs_witness(a, b, c)?;
        // let original_len = witness.len();
        
        // Pad to next power of 2 
        let padded_length = next_power_of_2(witness.len());
        let mut padded_witness = witness;
        padded_witness.resize(padded_length, F::zero());
        
        // println!("Witness padded from {} to {} elements", original_len, padded_length);
        
        Ok(padded_witness)
    }
}

// Helper functions

/// Calculate next power of 2 greater than or equal to n
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    1 << (64 - (n - 1).leading_zeros())
}

/// Convert dense matrix to SparseMat with better bounds checking
fn dense_to_sparse<F: PrimeField>(
    dense_matrix: &Matrix<F>, 
    padded_rows: usize, 
    padded_cols: usize
) -> Result<SparseFieldMat<F>, SynthesisError> {
    let mut sparse_matrix = SparseFieldMat::<F>::new(padded_rows, padded_cols);
    let mut sparse_data = Vec::new();
    
    // Fill non-zero entries from dense matrix
    for (row_idx, row) in dense_matrix.iter().enumerate() {
        if row_idx >= padded_rows {
            break;
        }
        
        for &(coeff, col_idx) in row.iter() {
           
            if col_idx < padded_cols && !coeff.is_zero() {
                sparse_data.push((row_idx, col_idx, coeff));
            }
        }
    }
    
    sparse_matrix.set_data(sparse_data);
    
    Ok(sparse_matrix)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;
    use ark_std::test_rng;
    use ark_bls12_381::{Fr as BlsFr};

    #[test]
    fn test_poseidon_r1cs_satisfaction() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Generate random inputs
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        
        // Compute correct hash value
        let c = poseidon.hash2to1(a, b);
        
        println!("Testing with:");
        println!("a = {:?}", a);
        println!("b = {:?}", b);
        println!("c = {:?}", c);
        
        // Generate R1CS matrices
        let (matrix_a, matrix_b, matrix_c) = poseidon.gen_r1cs_mats()
            .expect("Failed to generate R1CS matrices");
        
        // Generate witness vector
        let witness = poseidon.gen_r1cs_witness(a, b, c)
            .expect("Failed to generate witness");
        
        println!("Generated {} constraints", matrix_a.len());
        println!("Witness vector length: {}", witness.len());
        
        // Verify R1CS equation: A * w ∘ B * w = C * w
        let num_constraints = matrix_a.len();
        assert_eq!(matrix_b.len(), num_constraints);
        assert_eq!(matrix_c.len(), num_constraints);
        
        for constraint_idx in 0..num_constraints {
            // Compute A[i] * w
            let a_dot_w = matrix_multiply_row(&matrix_a[constraint_idx], &witness);
            
            // Compute B[i] * w  
            let b_dot_w = matrix_multiply_row(&matrix_b[constraint_idx], &witness);
            
            // Compute C[i] * w
            let c_dot_w = matrix_multiply_row(&matrix_c[constraint_idx], &witness);
            
            // Verify equation: (A[i] * w) * (B[i] * w) = C[i] * w
            let left_side = a_dot_w * b_dot_w;
            let right_side = c_dot_w;
            
            assert_eq!(
                left_side, 
                right_side,
                "R1CS constraint {} failed: ({:?}) * ({:?}) != {:?}",
                constraint_idx,
                a_dot_w,
                b_dot_w,
                c_dot_w
            );
        }
        
        println!("✅ All {} R1CS constraints satisfied!", num_constraints);
    }

    #[test]
    fn test_witness_based_dimension_consistency() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Generate test inputs
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        let c = poseidon.hash2to1(a, b);
        
        println!("=== Testing Witness-Based Dimension Consistency ===");
        
        // Step 1: Generate witness to determine dimensions
        let witness = poseidon.gen_r1cs_witness(a, b, c)
            .expect("Failed to generate witness");
        
        println!("Original witness length: {}", witness.len());
        
        // Step 2: Generate sparse matrices using witness length
        let (sparse_a, _sparse_b, _sparse_c) = poseidon.gen_r1cs_sparse_mats_padded()
            .expect("Failed to generate sparse matrices");
        
        // Step 3: Generate padded witness
        let padded_witness = poseidon.gen_r1cs_witness_padded(a, b, c)
            .expect("Failed to generate padded witness");
        
        let (rows_a, cols_a) = sparse_a.get_shape();
        
        println!("Results:");
        println!("  Original witness length: {}", witness.len());
        println!("  Padded witness length: {}", padded_witness.len());
        println!("  Matrix dimensions: {} x {}", rows_a, cols_a);
        
        assert_eq!(padded_witness.len(), cols_a, 
                   "Padded witness length ({}) must equal matrix columns ({})", 
                   padded_witness.len(), cols_a);
        
        assert!(padded_witness.len().is_power_of_two(), "Witness length should be power of 2");
        assert!(cols_a.is_power_of_two(), "Matrix columns should be power of 2");
        assert!(rows_a.is_power_of_two(), "Matrix rows should be power of 2");
        
        println!("✅ Witness-based dimension consistency test passed!");
    }

    #[test]
    fn test_poseidon_sparse_matrix_conversion() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Generate test inputs
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        let c = poseidon.hash2to1(a, b);
        
        println!("=== Testing Sparse Matrix Conversion ===");
        
        // Generate sparse matrices
        let (sparse_a, _sparse_b, _sparse_c) = poseidon.gen_r1cs_sparse_mats_padded()
            .expect("Failed to generate sparse R1CS matrices");
        
        // Generate padded witness
        let padded_witness = poseidon.gen_r1cs_witness_padded(a, b, c)
            .expect("Failed to generate padded witness");
        
        let (rows_a, cols_a) = sparse_a.get_shape();
        
        println!("Sparse matrix A dimensions: {}x{}", rows_a, cols_a);
        println!("Padded witness length: {}", padded_witness.len());
        
        // Verify dimensions are powers of 2
        assert!(rows_a.is_power_of_two(), "Matrix rows should be power of 2");
        assert!(cols_a.is_power_of_two(), "Matrix cols should be power of 2");
        assert!(padded_witness.len().is_power_of_two(), "Witness length should be power of 2");
        
        assert_eq!(padded_witness.len(), cols_a, 
                   "Witness length ({}) must match matrix columns ({})", 
                   padded_witness.len(), cols_a);
        
        println!("✅ Sparse matrix conversion tests passed!");
    }

    #[test]
    fn test_poseidon_sparse_matrix_satisfaction() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Generate test inputs
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        let c = poseidon.hash2to1(a, b);
        
        println!("=== Testing Sparse Matrix R1CS Satisfaction ===");
        
        // Generate sparse matrices and padded witness
        let (sparse_a, sparse_b, sparse_c) = poseidon.gen_r1cs_sparse_mats_padded()
            .expect("Failed to generate sparse R1CS matrices");
        let padded_witness = poseidon.gen_r1cs_witness_padded(a, b, c)
            .expect("Failed to generate padded witness");
        
        let aw = sparse_a.proj_right(&padded_witness);
        let bw = sparse_b.proj_right(&padded_witness);
        let cw = sparse_c.proj_right(&padded_witness);

        let left_side = hadamard_product(&aw, &bw);
        let right_side = cw;

        assert_eq!(left_side, right_side, "Sparse R1CS constraint failed");
        
        println!("✅ All sparse R1CS constraints satisfied!");
    }
    
    #[test]
    fn test_poseidon_hash_consistency() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Test hash computation consistency
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        
        let hash1 = poseidon.hash2to1(a, b);
        let hash2 = poseidon.hash2to1(a, b);
        
        assert_eq!(hash1, hash2, "Hash function should be deterministic");
        
        // Test different inputs produce different outputs
        let c = BlsFr::rand(&mut rng);
        let hash3 = poseidon.hash2to1(a, c);
        
        assert_ne!(hash1, hash3, "Different inputs should produce different hashes");
        
        println!("✅ Hash consistency tests passed!");
    }
    
    #[test]
    fn test_poseidon_r1cs_with_wrong_hash() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        let wrong_c = BlsFr::rand(&mut rng); // Use wrong hash value
        
        // Try to generate witness (should fail or produce witness that doesn't satisfy constraints)
        let result = poseidon.gen_r1cs_witness(a, b, wrong_c);
        
        // If witness generation succeeds, verify it doesn't satisfy constraints
        if let Ok(witness) = result {
            let (matrix_a, matrix_b, matrix_c) = poseidon.gen_r1cs_mats()
                .expect("Failed to generate R1CS matrices");
            
            let mut constraint_satisfied = true;
            
            for constraint_idx in 0..matrix_a.len() {
                let a_dot_w = matrix_multiply_row(&matrix_a[constraint_idx], &witness);
                let b_dot_w = matrix_multiply_row(&matrix_b[constraint_idx], &witness);
                let c_dot_w = matrix_multiply_row(&matrix_c[constraint_idx], &witness);
                
                if a_dot_w * b_dot_w != c_dot_w {
                    constraint_satisfied = false;
                    break;
                }
            }
            
            assert!(!constraint_satisfied, "Wrong hash should not satisfy R1CS constraints");
            println!("✅ Correctly detected invalid witness!");
        } else {
            println!("✅ Witness generation failed as expected for wrong hash!");
        }
    }
    
    #[test]
    fn test_poseidon_witness_vector_element_mapping() {
        let mut rng = test_rng();
        
        // Create Poseidon instance
        let poseidon = Poseidon::<BlsFr>::new_with_recommended_params()
            .expect("Failed to create Poseidon instance");
        
        // Generate test inputs
        let a = BlsFr::rand(&mut rng);
        let b = BlsFr::rand(&mut rng);
        let c = poseidon.hash2to1(a, b);
        
        println!("=== Input Values ===");
        println!("a = {:?}", a);
        println!("b = {:?}", b);
        println!("c = {:?}", c);
        
        // Generate witness vector
        let witness = poseidon.gen_r1cs_witness(a, b, c)
            .expect("Failed to generate witness");
        
        // Create constraint system to get variable information
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        let circuit = PoseidonCircuit {
            secret_a: Some(a),
            secret_b: Some(b),
            public_hash: Some(c),
            poseidon_config: poseidon.poseidon_config.clone(),
        };
        
        circuit.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        cs.finalize();
        
        let cs_borrow = cs.borrow().unwrap();
        let instance_assignment = &cs_borrow.instance_assignment;
        let witness_assignment = &cs_borrow.witness_assignment;
        
        println!("\n=== Instance Variables ===");
        println!("Instance variables count: {}", instance_assignment.len());
        for (i, &value) in instance_assignment.iter().enumerate() {
            println!("instance[{}] = {:?}", i, value);
        }
        
        println!("\n=== Witness Vector Info ===");
        println!("Total witness length: {}", witness.len());
        println!("Instance part: {} elements", instance_assignment.len());
        println!("Witness part: {} elements", witness_assignment.len());
        
        println!("\n=== First few witness elements ===");
        let show_count = std::cmp::min(10, witness.len());
        for i in 0..show_count {
            println!("witness[{}] = {:?}", i, witness[i]);
        }

        if witness.len() > 1 {
            assert_eq!(c, witness[1], "Second element of witness should match the output of hash");
        }
        if witness.len() > 2 {
            assert_eq!(a, witness[2], "Third element of witness should match the first input of hash");
        }
        if witness.len() > 3 {
            assert_eq!(b, witness[3], "Fourth element of witness should match the second input of hash");
        }
        
        println!("\n✅ Values printed successfully!");
    }
    
    // Helper function: compute dot product of matrix row and vector
    fn matrix_multiply_row(row: &[(BlsFr, usize)], vector: &[BlsFr]) -> BlsFr {
        let mut result = BlsFr::zero();
        for &(coeff, index) in row {
            if index < vector.len() {
                result += coeff * vector[index];
            }
        }
        result
    }

    // Locally implemented Hadamard product
    fn hadamard_product<F: PrimeField>(vec1: &[F], vec2: &[F]) -> Vec<F> {
        assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length for Hadamard product");
        
        vec1.iter()
            .zip(vec2.iter())
            .map(|(&a, &b)| a * b)
            .collect()
    }

}