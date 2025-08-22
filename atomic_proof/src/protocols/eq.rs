//! Prove that two matrices a = b
//! This is done by showing that their projections on random vectors are equal.
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
pub struct MatEq<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> MatEq<F> 
{
    pub fn new(
        shape: (usize, usize),
    ) -> Self {
        let protocol_input = AtomicMatProtocolInput {
            op: MatOp::Eq,
            a: DenseMatFieldCM::new(shape.0, shape.1),
            b: DenseMatFieldCM::new(shape.0, shape.1),
            hat_c: F::zero(),
            point_c: (Vec::new(), Vec::new()),
            shape_a: shape,
            shape_b: shape,
            shape_c: shape,
        };

        let mut atomic_pop = AtomicPoP::new();
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

    pub fn set_input(&mut self, mat_a: DenseMatFieldCM<F>, mat_b: DenseMatFieldCM<F>) {
        self.protocol_input.a = mat_a;
        self.protocol_input.b = mat_b;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for MatEq<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }
    
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        // For matrix eqition A + B = C, we use the challenges provided in point_c

        let m = self.protocol_input.shape_a.0; // rows
        let n = self.protocol_input.shape_a.1; // cols
        
        println!("MatEq: m={}, n={}", m, n);
        
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
        }        // Compute projections of matrices A and B onto the given challenges
        
        let hat_a = self.protocol_input.a.proj_lr_challenges(&xl, &xr);
        let hat_b = self.protocol_input.b.proj_lr_challenges(&xl, &xr);
        
        // Push computed values to transcript
        let hat_a_index = trans.pointer;
        trans.push_response(hat_a);
        let hat_b_index = trans.pointer;
        trans.push_response(hat_b);
        
        // The hat_c value should already be set correctly in the constructor.
        // We verify the constraint here.
        let constraint_result = hat_a == hat_b;
        
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

        println!("✅ MatEq reduce_prover completed successfully");
        
        if !constraint_result {
            println!("!! MatEq constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
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

        let hat_b_index = trans.pointer;
        let hat_b = trans.get_at_position(hat_b_index);
        trans.pointer += 1;

        println!("MatEq verify: read hat_a={}, hat_b={}", hat_a, hat_b);
        
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
        
        println!("✅ MatEq verify_as_subprotocol completed successfully");

        // Verify the constraint: hat_a + hat_b should equal hat_c
        let constraint_result = hat_a == hat_b;
        if !constraint_result {
            println!("!! MatEq verification constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
        }
        constraint_result
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        // Calculate basic check: hat_c = a - b  
        let check = ArithmeticExpression::sub(
            ArithmeticExpression::input(self.atomic_pop.mapping.hat_a_index),
            ArithmeticExpression::input(self.atomic_pop.mapping.hat_b_index),
        ); // Placeholder for now

        // Set up basic linking constraints
        let link_xa = (Vec::new(), Vec::new());
        let link_xb = (Vec::new(), Vec::new());

        self.atomic_pop.set_check(check);
        self.atomic_pop.set_link_xa(link_xa);
        self.atomic_pop.set_link_xb(link_xb);


        if !self.atomic_pop.is_ready() {
            println!("⚠️ AtomicPoP is not ready! Run EqProtocol.reduce_prover first");
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
    use crate::atomic_protocol::AtomicMatProtocol;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    // Helper: random matrix (power-of-two dimensions recommended)
    fn rand_mat(rows: usize, cols: usize, rng: &mut impl ark_std::rand::RngCore) -> DenseMatFieldCM<BlsFr> {
        let mut m = DenseMatFieldCM::new(rows, cols);
        // column-major fill
        let mut cols_data = Vec::new();
        for _c in 0..cols { let mut col = Vec::new(); for _r in 0..rows { col.push(BlsFr::rand(rng)); } cols_data.push(col); }
        m.set_data(cols_data); m
    }

    #[test]
    fn test_eqprotocol_basic() {
        let mut rng = test_rng();
        let (rows, cols) = (4, 4); // 2^2 x 2^2
        let a = rand_mat(rows, cols, &mut rng);
        let b = a.clone(); // identical
        let mut proto = MatEq::<BlsFr>::new((rows, cols));
        proto.set_input(a.clone(), b.clone());
        let mut prover_trans = Transcript::new(BlsFr::rand(&mut rng));
        assert!(proto.reduce_prover(&mut prover_trans));

        // verifier
        let mut verifier_trans = prover_trans.clone(); verifier_trans.reset_pointer();
        let mut vproto = MatEq::<BlsFr>::new((rows, cols));
        vproto.set_input(a, b);
        assert!(vproto.verify_as_subprotocol(&mut verifier_trans));
        assert_eq!(prover_trans.pointer, verifier_trans.pointer);
    }

    #[test]
    fn test_eqprotocol_detects_inequality() {
        let mut rng = test_rng();
        let (rows, cols) = (4, 4);
        let a = rand_mat(rows, cols, &mut rng);
        let mut b = a.clone();
        // Flip one entry to ensure inequality
        if !b.data.is_empty() && !b.data[0].is_empty() { b.data[0][0] += BlsFr::from(1u64); }
        let mut proto = MatEq::<BlsFr>::new((rows, cols));
        proto.set_input(a.clone(), b.clone());
        let mut prover_trans = Transcript::new(BlsFr::rand(&mut rng));
        assert!(!proto.reduce_prover(&mut prover_trans)); // should detect mismatch

        // verifier path should also return false
        let mut verifier_trans = prover_trans.clone(); verifier_trans.reset_pointer();
        let mut vproto = MatEq::<BlsFr>::new((rows, cols));
        vproto.set_input(a, b);
        assert!(!vproto.verify_as_subprotocol(&mut verifier_trans));
    }
}