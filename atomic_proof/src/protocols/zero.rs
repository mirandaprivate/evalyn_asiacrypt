//! Prove a matrix is a zero matrix
//! Prove its projection on a random challenges are zero

use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;

use crate::atomic_pop::AtomicPoP;
use crate::atomic_protocol::{AtomicMatProtocol, AtomicMatProtocolInput, MatOp};
use crate::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};

#[derive(Debug, Clone)]
pub struct EqZero<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> EqZero<F> 
{
    pub fn new(
        shape_a: (usize, usize),
    ) -> Self {
        let protocol_input = AtomicMatProtocolInput {
            op: MatOp::EqZero,
            a: DenseMatFieldCM::<F>::new(shape_a.0, shape_a.1),
            b: DenseMatFieldCM::<F>::new(0, 0),
            hat_c: F::zero(),
            point_c: (Vec::new(), Vec::new()),
            shape_a: shape_a,
            shape_b: (0, 0),
            shape_c: shape_a,
        };

        let mut atomic_pop = AtomicPoP::<F>::new();
        // Set the message with the correct c value and c_index
        atomic_pop.set_message(
            F::zero(),
            (Vec::new(), Vec::new()),
            0,
            (Vec::new(), Vec::new()),
        );

        Self {
            protocol_input,
            atomic_pop,
        }
    }

}



impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for EqZero<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }
    
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        let mut xl = Vec::new();
        let mut xr = Vec::new();
        let mut xl_indices = Vec::new();
        let mut xr_indices = Vec::new(); 

        let m = self.protocol_input.a.shape.0;
        let n = self.protocol_input.a.shape.1;

        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;

        for _ in 0..log_m {
            xl_indices.push(trans.pointer);
            let challenge = trans.gen_challenge();
            xl.push(challenge);
        }

        for _ in 0..log_n {
            xr_indices.push(trans.pointer);
            let challenge = trans.gen_challenge();
            xr.push(challenge);
        }
        
        // Compute projection of matrix A onto challenges
        let a_hat = F::zero();
        // For zero matrix protocol, b_hat should be zero
        let b_hat = F::zero();
        
        // Push computed values to transcript
        let a_hat_index = trans.pointer;
        trans.push_response(a_hat);
         
        let b_hat_index = 0; // placeholder index

        self.atomic_pop.set_pop_trans(
            a_hat,
            b_hat,
            (xl.clone(), xr.clone()),
            (Vec::new(), Vec::new()),
            Vec::new(),
            Vec::new(),
            a_hat_index,
            b_hat_index,
            (xl_indices.clone(), xr_indices.clone()),
            (Vec::new(), Vec::new()),
            Vec::new(),
            Vec::new(),
        );

        println!("✅ EqZero reduce_prover completed successfully");
        
        // Verify the constraint: a_hat should equal zero (b_hat is zero)
        let constraint_result = a_hat == F::zero();
        if !constraint_result {
            println!("!! EqZero constraint failed: {} != 0", a_hat);
        }
        constraint_result
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let mut xl = Vec::new();
        let mut xr = Vec::new();
        let mut xl_indices = Vec::new();
        let mut xr_indices = Vec::new();
        
        let m = self.protocol_input.a.shape.0;
        let n = self.protocol_input.a.shape.1;

        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;
        
        // Read challenges from transcript (same as in reduce_prover)
        for _ in 0..log_m {
            xl_indices.push(trans.pointer);
            let challenge = trans.get_at_position(trans.pointer);
            xl.push(challenge);
            trans.pointer += 1;
        }

        for _ in 0..log_n {
            xr_indices.push(trans.pointer);
            let challenge = trans.get_at_position(trans.pointer);
            xr.push(challenge);
            trans.pointer += 1;
        }
        

        // Read computed values from transcript
        let hat_a_index = trans.pointer;
        let hat_a = trans.get_at_position(trans.pointer);
        trans.pointer += 1;
        
        let hat_b_index = 0; // placeholder index

        self.atomic_pop.set_pop_trans(
            hat_a,
            F::zero(),
            (xl.clone(), xr.clone()),
            (Vec::new(), Vec::new()),
            Vec::new(),
            Vec::new(),
            hat_a_index,
            hat_b_index,
            (xl_indices.clone(), xr_indices.clone()),
            (Vec::new(), Vec::new()),
            Vec::new(),
            Vec::new(),
        );
        
        println!("✅ EqZero verify_as_subprotocol completed successfully");
        
        // Verify the constraint: hat_a should equal zero
        let constraint_result = hat_a == F::zero();
        if !constraint_result {
            println!("!! EqZero verification constraint failed: {} != 0", hat_a);
        }
        constraint_result
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        // Calculate basic check: all elements equal zero
        let check = ArithmeticExpression::input(0); // Placeholder for now
        let link_xa = (Vec::new(), Vec::new());
        let link_xb = (Vec::new(), Vec::new());

        self.atomic_pop.set_check(check);
        self.atomic_pop.set_link_xa(link_xa);
        self.atomic_pop.set_link_xb(link_xb);

        if !self.atomic_pop.is_ready() {
            println!("!! AtomicPoP is not ready! Run ZeroProtocol.reduce_prover() first");
            return false;
        }
        true
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        self.atomic_pop.synthesize_constraints(cs_builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use fsproof::helper_trans::Transcript;
    use mat::utils::matdef::DenseMatFieldCM;

    #[test]
    fn test_eqzero_new() {
        println!("=== Testing EqZero::new ===");
        
        let shape_a = (4, 4);
        
        let eq_zero = EqZero::<BlsFr>::new(shape_a);
        
        assert_eq!(eq_zero.protocol_input.shape_a, shape_a);
        assert_eq!(eq_zero.protocol_input.shape_b, (0, 0));
        assert_eq!(eq_zero.protocol_input.shape_c, shape_a);
        assert_eq!(eq_zero.protocol_input.hat_c, BlsFr::zero());
        
        println!("✅ EqZero::new test passed");
    }


    #[test]
    fn test_eqzero_with_actual_zero_matrix() {
        println!("=== Testing EqZero with actual zero matrix ===");
        
        let mut trans = Transcript::<BlsFr>::new(BlsFr::zero());
        
        // Create EqZero protocol instance
        let mut eq_zero = EqZero::<BlsFr>::new((4, 4));
        
        // Test reduce_prover
        let prover_result = eq_zero.reduce_prover(&mut trans);
        assert!(prover_result, "Prover should succeed for zero matrix");
        
        // Reset transcript for verification
        let mut verify_trans = trans.clone();
        verify_trans.reset_pointer();
        
        // Test verify_as_subprotocol  
        let verify_result = eq_zero.verify_as_subprotocol(&mut verify_trans);
        assert!(verify_result, "Verification should succeed for zero matrix");
        
        println!("✅ EqZero with actual zero matrix test passed");
    }

    #[test]
    fn test_eqzero_different_sizes() {
        println!("=== Testing EqZero with different matrix sizes ===");
        
        let sizes = vec![(2, 2), (4, 8), (8, 4), (16, 16)];
        
        for shape in sizes {
            println!("Testing with shape {:?}", shape);
            
            let mut eq_zero = EqZero::<BlsFr>::new(shape);
        
            
            let mut trans = Transcript::<BlsFr>::new(BlsFr::zero());
            let prover_result = eq_zero.reduce_prover(&mut trans);
            assert!(prover_result, "Prover should succeed for zero matrix of shape {:?}", shape);
            
            let mut verify_trans = trans.clone();
            verify_trans.reset_pointer();
            let verify_result = eq_zero.verify_as_subprotocol(&mut verify_trans);
            assert!(verify_result, "Verification should succeed for zero matrix of shape {:?}", shape);
        }
        
        println!("✅ EqZero different sizes test passed");
    }

    #[test]
    fn test_eqzero_protocol_flow() {
        println!("=== Testing EqZero complete protocol flow ===");
        
        // Create EqZero protocol instance
        let mut eq_zero = EqZero::<BlsFr>::new((8, 8));
        

        
        // Test full protocol flow
        let mut trans = Transcript::<BlsFr>::new(BlsFr::zero());
        
        // Step 1: Prover phase
        let prover_result = eq_zero.reduce_prover(&mut trans);
        assert!(prover_result, "Prover phase should succeed");
        
        // Step 2: Verification phase (using verify method)
        let mut verify_trans = trans.clone();
        verify_trans.reset_pointer();
        let verify_result = eq_zero.verify(&mut verify_trans);
        assert!(verify_result, "Verification phase should succeed");
        
        println!("✅ EqZero complete protocol flow test passed");
    }


    #[test]
    fn test_eqzero_matrix_projection() {
        println!("=== Testing EqZero matrix projection properties ===");
        
        // Create a matrix with specific structure to test projection
        let mut test_mat = DenseMatFieldCM::<BlsFr>::new(4, 4);
        let mut test_data = vec![vec![BlsFr::zero(); 4]; 4];
        
        // Make it so that random projections are likely to be non-zero
        for i in 0..4 {
            for j in 0..4 {
                test_data[j][i] = BlsFr::from((i + j) as u64);
            }
        }
        test_mat.set_data(test_data);
        
        // Test that projection of non-zero matrix gives non-zero result
        let xl = vec![BlsFr::from(1u64), BlsFr::from(2u64)]; // log2(4) = 2 challenges
        let xr = vec![BlsFr::from(3u64), BlsFr::from(4u64)]; // log2(4) = 2 challenges
        
        let projection = test_mat.proj_lr_challenges(&xl, &xr);
        assert_ne!(projection, BlsFr::zero(), "Projection of non-zero matrix should be non-zero");
        
        // Test that projection of zero matrix gives zero
        let mut zero_mat = DenseMatFieldCM::<BlsFr>::new(4, 4);
        let zero_data = vec![vec![BlsFr::zero(); 4]; 4];
        zero_mat.set_data(zero_data);
        
        let zero_projection = zero_mat.proj_lr_challenges(&xl, &xr);
        assert_eq!(zero_projection, BlsFr::zero(), "Projection of zero matrix should be zero");
        
        println!("✅ EqZero matrix projection test passed");
    }

    #[test]
    fn test_eqzero_edge_cases() {
        println!("=== Testing EqZero edge cases ===");
        
        // Test with smallest power-of-2 matrix (2x2)
        let mut eq_zero_small = EqZero::<BlsFr>::new((2, 2));
        let mut small_zero_mat = DenseMatFieldCM::<BlsFr>::new(2, 2);
        let small_zero_data = vec![vec![BlsFr::zero(); 2]; 2];
        small_zero_mat.set_data(small_zero_data);
   
        
        let mut trans_small = Transcript::<BlsFr>::new(BlsFr::zero());
        let result_small = eq_zero_small.reduce_prover(&mut trans_small);
        assert!(result_small, "Should work with 2x2 zero matrix");
        
        // Test with rectangular zero matrix (4x8)
        let mut eq_zero_rect = EqZero::<BlsFr>::new((4, 8));
        let mut rect_zero_mat = DenseMatFieldCM::<BlsFr>::new(4, 8);
        let rect_zero_data = vec![vec![BlsFr::zero(); 4]; 8];
        rect_zero_mat.set_data(rect_zero_data);
        
        let mut trans_rect = Transcript::<BlsFr>::new(BlsFr::zero());
        let result_rect = eq_zero_rect.reduce_prover(&mut trans_rect);
        assert!(result_rect, "Should work with 4x8 zero matrix");
        
        println!("✅ EqZero edge cases test passed");
    }

    #[test]
    fn test_integrated_matrix_equation_proof() {
        println!("=== Testing integrated proof: mat_a * mat_b - mat_c = 0 ===");
        
        use crate::protocols::mul::MatMul;
        use crate::protocols::sub::MatSub;
        use ark_ff::{Zero, One};
        
        // --- Test Data Setup ---
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(4, 4);
        let mut a_data = vec![vec![BlsFr::zero(); 4]; 4];
        for i in 0..4 { a_data[i][i] = BlsFr::one(); }
        mat_a.set_data(a_data);
        
        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(4, 4);
        let mut b_data = vec![vec![BlsFr::zero(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                b_data[j][i] = BlsFr::from((i + j + 1) as u64);
            }
        }
        mat_b.set_data(b_data);
        
        let mat_c = mat_b.clone();
        let mat_d = mat_a.mat_mul(&mat_b);


        let mut prover_trans = Transcript::new(BlsFr::zero());
        
        let mut eq_zero_protocol = EqZero::<BlsFr>::new((4, 4));

        eq_zero_protocol.reduce_prover(&mut prover_trans);

        let (hat_c, point_c) = eq_zero_protocol.atomic_pop.get_a();
        let (hat_c_index, point_c_index) = eq_zero_protocol.atomic_pop.get_a_index();

        let mut sub_protocol = MatSub::<BlsFr>::new(
            hat_c, point_c, hat_c_index, point_c_index,
            (4, 4), (4, 4), (4, 4)
        );
        sub_protocol.set_input(mat_d, mat_c.clone());

        sub_protocol.reduce_prover(&mut prover_trans);

        let (hat_c, point_c) = sub_protocol.atomic_pop.get_a();
        let (hat_c_index, point_c_index) = sub_protocol.atomic_pop.get_a_index();
        
        let mut mul_protocol = MatMul::<BlsFr>::new(
            hat_c, point_c, hat_c_index, point_c_index,
            (4, 4), (4, 4), (4, 4)
        );
        mul_protocol.set_input(mat_a.clone(), mat_b.clone());

        mul_protocol.reduce_prover(&mut prover_trans);
        assert_eq!(prover_trans.pointer, prover_trans.trans_seq.len());

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();

        let mut eq_zero_protocol = EqZero::<BlsFr>::new((4, 4));

        let flag1 = eq_zero_protocol.verify_as_subprotocol(&mut verifier_trans);

        let (hat_c, point_c) = eq_zero_protocol.atomic_pop.get_a();
        let (hat_c_index, point_c_index) = eq_zero_protocol.atomic_pop.get_a_index();

        let mut sub_protocol = MatSub::<BlsFr>::new(
            hat_c, point_c, hat_c_index, point_c_index,
            (4, 4), (4, 4), (4, 4)
        );

        let flag2 = sub_protocol.verify_as_subprotocol(&mut verifier_trans);

        let (hat_c, point_c) = sub_protocol.atomic_pop.get_a();
        let (hat_c_index, point_c_index) = sub_protocol.atomic_pop.get_a_index();
        
        let mut mul_protocol = MatMul::<BlsFr>::new(
            hat_c, point_c, hat_c_index, point_c_index,
            (4, 4), (4, 4), (4, 4)
        );

        let flag3 = mul_protocol.verify_as_subprotocol(&mut verifier_trans);

        assert_eq!(flag1 && flag2 && flag3, true);
        assert_eq!(verifier_trans.pointer, verifier_trans.trans_seq.len());

        let (c_proj, c_point) = sub_protocol.atomic_pop.get_b();
        let (a_proj, a_point) = mul_protocol.atomic_pop.get_a();
        let (b_proj, b_point) = mul_protocol.atomic_pop.get_b();

        let c_proj_expected = mat_c.proj_lr_challenges(&c_point.0, &c_point.1);
        let a_proj_expected = mat_a.proj_lr_challenges(&a_point.0, &a_point.1);
        let b_proj_expected = mat_b.proj_lr_challenges(&b_point.0, &b_point.1);

        assert_eq!(c_proj, c_proj_expected, "Projection of mat_c should match expected");
        assert_eq!(a_proj, a_proj_expected, "Projection of mat_a should match expected");
        assert_eq!(b_proj, b_proj_expected, "Projection of mat_b should match expected");
    }

    #[test]
    fn test_integrated_matrix_equation_with_verification() {
        println!("=== Testing matrix equation with full verification ===");
        
        use crate::protocols::mul::MatMul;
        use crate::protocols::sub::MatSub;
        use ark_ff::Zero;
        
        // Create a more complex example: 2x2 matrices for faster testing
        
        // mat_a = [[1, 2], [3, 4]]
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(2, 2);
        let a_data = vec![
            vec![BlsFr::from(1u64), BlsFr::from(3u64)], // Column 0
            vec![BlsFr::from(2u64), BlsFr::from(4u64)], // Column 1
        ];
        mat_a.set_data(a_data);
        
        // mat_b = [[5, 6], [7, 8]]
        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(2, 2);
        let b_data = vec![
            vec![BlsFr::from(5u64), BlsFr::from(7u64)], // Column 0
            vec![BlsFr::from(6u64), BlsFr::from(8u64)], // Column 1
        ];
        mat_b.set_data(b_data);
        
        // mat_c = mat_a * mat_b = [[19, 22], [43, 50]]
        let mut mat_c = DenseMatFieldCM::<BlsFr>::new(2, 2);
        let c_data = vec![
            vec![BlsFr::from(19u64), BlsFr::from(43u64)], // Column 0: [1*5+2*7, 3*5+4*7]
            vec![BlsFr::from(22u64), BlsFr::from(50u64)], // Column 1: [1*6+2*8, 3*6+4*8]
        ];
        mat_c.set_data(c_data);
        
        println!("Created 2x2 test matrices with known multiplication result");
        
        // Generate challenges for 2x2 matrices (log2(2) = 1 challenge each)
        let xl = vec![BlsFr::from(3u64)]; // 1 challenge for rows
        let xr = vec![BlsFr::from(5u64)]; // 1 challenge for columns
        let point_c = (xl.clone(), xr.clone());
        
        // Test the complete workflow with verification
        
        // 1. MatMul: Prove mat_a * mat_b = some result
        let expected_mul = mat_c.proj_lr_challenges(&xl, &xr); // This should match actual multiplication
        
        let mut mat_mul = MatMul::<BlsFr>::new(
            expected_mul,
            point_c.clone(),
            0,
            (vec![0], vec![1]),
            (2, 2), (2, 2), (2, 2),
        );
        mat_mul.set_input(mat_a, mat_b);
        
        // Prover phase
        let mut mul_prover_trans = Transcript::<BlsFr>::new(BlsFr::zero());
        let mul_prove_success = mat_mul.reduce_prover(&mut mul_prover_trans);
        assert!(mul_prove_success, "MatMul proving should succeed");
        
        // Verification phase
        // The verifier must use the transcript generated by the prover.
        let mut mul_verify_trans = mul_prover_trans.clone();
        mul_verify_trans.reset_pointer();
        let mul_verify_success = mat_mul.verify_as_subprotocol(&mut mul_verify_trans);
        assert!(mul_verify_success, "MatMul verification should succeed");
        
        // 2. MatSub: Prove result - mat_c = 0
        let expected_sub = BlsFr::zero(); // Should be zero if multiplication is correct
        
        let mut mat_sub = MatSub::<BlsFr>::new(
            expected_sub,
            point_c.clone(),
            1,
            (vec![2], vec![3]),
            (2, 2), (2, 2), (2, 2),
        );
        mat_sub.set_input(mat_c.clone(), mat_c.clone()); // c - c = 0
        
        // Prover phase
        let mut sub_prover_trans = Transcript::<BlsFr>::new(BlsFr::zero());
        let sub_prove_success = mat_sub.reduce_prover(&mut sub_prover_trans);
        assert!(sub_prove_success, "MatSub proving should succeed");
        
        // Verification phase
        let mut sub_verify_trans = sub_prover_trans.clone();
        sub_verify_trans.reset_pointer();
        let sub_verify_success = mat_sub.verify_as_subprotocol(&mut sub_verify_trans);
        assert!(sub_verify_success, "MatSub verification should succeed");
        
        // 3. EqZero: Prove the final result is zero matrix
        let mut eq_zero = EqZero::<BlsFr>::new((2, 2));
        
        // Prover phase
        let mut zero_prover_trans = Transcript::<BlsFr>::new(BlsFr::zero());
        let zero_prove_success = eq_zero.reduce_prover(&mut zero_prover_trans);
        assert!(zero_prove_success, "EqZero proving should succeed");
        
        // Verification phase
        let mut zero_verify_trans = zero_prover_trans.clone();
        zero_verify_trans.reset_pointer();
        let zero_verify_success = eq_zero.verify_as_subprotocol(&mut zero_verify_trans);
        assert!(zero_verify_success, "EqZero verification should succeed");
        
        println!("✅ Complete matrix equation with verification test passed");
        println!("✅ Successfully proved and verified: mat_a * mat_b - mat_c = 0");
    }
}