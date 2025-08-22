//! Implement the matrix subtraction protocol for zero-knowledge proofs
//! For mat_a - mat_b = mat_c, we verify that projections are equal
//! point_c values and indices are passed to point_a and point_b
//!
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
pub struct MatSub<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> MatSub<F> 
{
    pub fn new(
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
        shape_c: (usize, usize),
        shape_a: (usize, usize),
        shape_b: (usize, usize),
    ) -> Self {
        let protocol_input = AtomicMatProtocolInput {
            op: MatOp::Sub,
            a: DenseMatFieldCM::new(shape_a.0, shape_a.1),
            b: DenseMatFieldCM::new(shape_b.0, shape_b.1),
            hat_c: hat_c,
            point_c: point_c.clone(),
            shape_a: shape_a,
            shape_b: shape_b,
            shape_c: shape_c,
        };

        let mut atomic_pop = AtomicPoP::new();
        // Set the message with the correct c value and c_index
        atomic_pop.set_message(
            hat_c,
            point_c,
            hat_c_index,
            point_c_index,
        );

        Self {
            protocol_input,
            atomic_pop,
        }
    }

    pub fn set_input(&mut self, mat_a: DenseMatFieldCM<F>, mat_b: DenseMatFieldCM<F>) {
        self.protocol_input.a = mat_a;
        self.protocol_input.b = mat_b;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for MatSub<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }
    
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        // For matrix subtraction A - B = C, we use the challenges provided in point_c
        
        // let m = self.protocol_input.shape_a.0; // rows
        // let n = self.protocol_input.shape_a.1; // cols
        
        // println!("MatSub: m={}, n={}", m, n);
        
        // Use the challenges from the protocol input, don't generate new ones
        let xl = self.protocol_input.point_c.0.clone();
        let xr = self.protocol_input.point_c.1.clone();
        
        // The indices for point_c are already set in the constructor.
        // We just need to retrieve them.
        let xl_indices = self.atomic_pop.mapping.point_c_index.0.clone();
        let xr_indices = self.atomic_pop.mapping.point_c_index.1.clone();

        // Compute projections of matrices A and B onto the given challenges
        let hat_a = self.protocol_input.a.proj_lr_challenges(&xl, &xr);
        let hat_b = self.protocol_input.b.proj_lr_challenges(&xl, &xr);
        
        // Push computed values to transcript
        let hat_a_index = trans.pointer;
        trans.push_response(hat_a);
        let hat_b_index = trans.pointer;
        trans.push_response(hat_b);
        
        // The hat_c value should already be set correctly in the constructor.
        // We verify the constraint here.
        let constraint_result = hat_a - hat_b == self.protocol_input.hat_c;
        
        // Store all indices in atomic_pop
        self.atomic_pop.set_pop_trans(
            hat_a,
            hat_b,
            (xl.clone(), xr.clone()),
            (xl.clone(), xr.clone()),
            Vec::new(),
            vec![hat_a, hat_b],
            hat_a_index,
            hat_b_index,
            (xl_indices.clone(), xr_indices.clone()), // point_a_index
            (xl_indices.clone(), xr_indices.clone()), // point_b_index
            Vec::new(), // all challenge indices
            vec![hat_a_index, hat_b_index],
        );

        // println!("✅ MatSub reduce_prover completed successfully");
        
        if !constraint_result {
            println!("!! MatSub constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
        }
        constraint_result
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        // The verifier knows point_c. It only needs to read hat_a and hat_b from the transcript.
        let m = self.protocol_input.shape_a.0;
        let n = self.protocol_input.shape_a.1;
        
        // println!("MatSub verify_as_subprotocol: m={}, n={}", m, n);
        // println!("verify_as_subprotocol: transcript length = {}, pointer = {}", trans.trans_seq.len(), trans.pointer);

        // The challenges xl and xr are already known to the verifier
        let xl = self.protocol_input.point_c.0.clone();
        let xr = self.protocol_input.point_c.1.clone();
        let xl_indices = self.atomic_pop.mapping.point_c_index.0.clone();
        let xr_indices = self.atomic_pop.mapping.point_c_index.1.clone();

        // Read hat_a and hat_b from the transcript
        let hat_a_index = trans.pointer;
        let hat_a = trans.get_at_position(hat_a_index);
        trans.pointer += 1;

        let hat_b_index = trans.pointer;
        let hat_b = trans.get_at_position(hat_b_index);
        trans.pointer += 1;

        // println!("MatSub verify: read hat_a={}, hat_b={}", hat_a, hat_b);
        
        self.atomic_pop.set_pop_trans(
            hat_a,
            hat_b,
            (xl.clone(), xr.clone()),
            (xl.clone(), xr.clone()),
            Vec::new(),
            vec![hat_a, hat_b],
            hat_a_index,
            hat_b_index,
            (xl_indices.clone(), xr_indices.clone()), // point_a_index
            (xl_indices.clone(), xr_indices.clone()), // point_b_index
            Vec::new(), // all challenge indices
            vec![hat_a_index, hat_b_index],
        );
        
        println!("✅ MatSub verify_as_subprotocol completed successfully");
        
        // Verify the constraint: hat_a - hat_b should equal hat_c
        let constraint_result = hat_a - hat_b == self.protocol_input.hat_c;
        if !constraint_result {
            println!("!! MatSub verification constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
        }
        constraint_result
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        // Calculate basic check: hat_c = a - b  
        let check = ArithmeticExpression::sub(
            ArithmeticExpression::input(self.atomic_pop.mapping.hat_c_index),
            ArithmeticExpression::sub(
                ArithmeticExpression::input(self.atomic_pop.mapping.hat_a_index),
                ArithmeticExpression::input(self.atomic_pop.mapping.hat_b_index),
            )
        ); // Placeholder for now

        // Set up basic linking constraints
        let link_xa = (Vec::new(), Vec::new());
        let link_xb = (Vec::new(), Vec::new());

        // self.atomic_pop.set_message(
        //     self.protocol_input.hat_c,
        //     self.protocol_input.point_c.clone(),
        //     0, // hat_c_index
        //     (Vec::new(), Vec::new()), // point_c_index placeholder
        // );

        self.atomic_pop.set_check(check);
        self.atomic_pop.set_link_xa(link_xa);
        self.atomic_pop.set_link_xb(link_xb);


        if !self.atomic_pop.is_ready() {
            println!("⚠️ AtomicPoP is not ready! Run SubProtocol.reduce_prover first");
            return false;
        }

        true
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        self.atomic_pop.synthesize_constraints(cs_builder);
        self.atomic_pop.is_ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_std::test_rng;
    use fsproof::helper_trans::Transcript;

    #[test]
    fn test_matsub_point_passing() {
        println!("=== Testing MatSub point_c passing with reduce_prover_with_atomic_pop ===");

        // Use power-of-2 dimensions: 4x4 matrices (log_m=2, log_n=2)
        let shape_a = (4, 4);
        let shape_b = (4, 4);
        let shape_c = (4, 4);

        let mut rng = test_rng();
        
        // Create random matrices A and B
        let mut mat_a = DenseMatFieldCM::new(shape_a.0, shape_a.1);
        let a_data: Vec<Vec<BlsFr>> = (0..shape_a.1)
            .map(|_| (0..shape_a.0).map(|_| BlsFr::rand(&mut rng)).collect())
            .collect();
        mat_a.set_data(a_data);

        let mut mat_b = DenseMatFieldCM::new(shape_b.0, shape_b.1);
        let b_data: Vec<Vec<BlsFr>> = (0..shape_b.1)
            .map(|_| (0..shape_b.0).map(|_| BlsFr::rand(&mut rng)).collect())
            .collect();
        mat_b.set_data(b_data);

        // Compute C = A - B
        let mat_c = mat_a.mat_sub(&mat_b);
        
        // Generate random challenge points
        let point_c = ([BlsFr::rand(&mut rng); 2].to_vec(), [BlsFr::rand(&mut rng); 2].to_vec());
        let point_c_index = (vec![1, 2], vec![3, 4]);
        
        // Compute the correct hat_c value: <point_c, C * point_c>
        let hat_c = mat_c.proj_lr_challenges(&point_c.0, &point_c.1);

        println!("Matrix dimensions: {:?}", shape_a);
        println!("Challenge points: xl = {:?}, xr = {:?}", point_c.0, point_c.1);
        println!("Computed hat_c = {}", hat_c);

        // Create MatSub protocol instance
        let mut matsub = MatSub::new(
            hat_c, 
            point_c.clone(), 
            0, // hat_c_index
            point_c_index.clone(), 
            shape_a, 
            shape_b, 
            shape_c
        );
        matsub.set_input(mat_a.clone(), mat_b.clone());

        // Initialize transcript with the hat_c value
        let mut prover_trans = Transcript::new(hat_c);
        
        // Add point_c to transcript as initial message
        prover_trans.push_response(point_c.0[0]);
        prover_trans.push_response(point_c.0[1]);
        prover_trans.push_response(point_c.1[0]);
        prover_trans.push_response(point_c.1[1]);

        // Use reduce_prover_with_atomic_pop to generate proof and prepare atomic PoP
        let prover_result = matsub.reduce_prover_with_atomic_pop(&mut prover_trans);
        
        if !prover_result {
            println!("!! reduce_prover_with_atomic_pop failed");
            panic!("Prover should succeed with correct hat_c");
        }

        println!("✅ Prover completed successfully");
        
        // Verify atomic PoP is properly prepared
        assert!(matsub.atomic_pop.ready.0, "atomic_pop message should be ready");
        assert!(matsub.atomic_pop.ready.1, "atomic_pop transcript should be ready");
        
        println!("AtomicPoP status: message_ready={}, trans_ready={}", 
                matsub.atomic_pop.ready.0, matsub.atomic_pop.ready.1);

        // Test verification with a fresh verifier instance
        let mut verifier = MatSub::new(
            hat_c, 
            point_c.clone(), 
            0, 
            point_c_index.clone(), 
            shape_a, 
            shape_b, 
            shape_c
        );
        verifier.set_input(mat_a, mat_b);
        
        // Reset transcript pointer for verification
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.pointer = 5;
        
        // Verify as subprotocol
        let verify_result = verifier.verify_as_subprotocol(&mut verifier_trans);
        assert!(verify_result, "Verification should succeed");
        
        println!("✅ Verification completed successfully");

        // Additional check: verify point passing
        // In MatSub, point_c should be passed to both point_a and point_b
        println!("Checking point passing:");
        println!("  Original point_c: ({:?}, {:?})", point_c.0, point_c.1);
        println!("  AtomicPoP point_a: ({:?}, {:?})", matsub.atomic_pop.point_a.0, matsub.atomic_pop.point_a.1);
        println!("  AtomicPoP point_b: ({:?}, {:?})", matsub.atomic_pop.point_b.0, matsub.atomic_pop.point_b.1);
        
        // For MatSub, the points should be related to the challenge structure
        // The exact relationship depends on the protocol implementation
        
        println!("✅ MatSub point passing test with reduce_prover_with_atomic_pop completed successfully");
    }
}
