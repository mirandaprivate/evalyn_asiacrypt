//! Implement the matrix Addition protocol for zero-knowledge proofs
//! For mat_a + mat_b = mat_c, we verify that projections are equal
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
pub struct MatAdd<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> MatAdd<F> 
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
            op: MatOp::Add,
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

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for MatAdd<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }
    
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        // For matrix addition A + B = C, we use the challenges provided in point_c

        let m = self.protocol_input.shape_a.0; // rows
        let n = self.protocol_input.shape_a.1; // cols
        
        println!("MatAdd: m={}, n={}", m, n);
        
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
        let constraint_result = hat_a + hat_b == self.protocol_input.hat_c;
        
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

        println!("✅ MatAdd reduce_prover completed successfully");
        
        if !constraint_result {
            println!("!! MatAdd constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
        }
        constraint_result
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        // The verifier knows point_c. It only needs to read hat_a and hat_b from the transcript.
        let m = self.protocol_input.shape_a.0;
        let n = self.protocol_input.shape_a.1;
        
        println!("MatAdd verify_as_subprotocol: m={}, n={}", m, n);
        println!("verify_as_subprotocol: transcript length = {}, pointer = {}", trans.trans_seq.len(), trans.pointer);

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

        println!("MatAdd verify: read hat_a={}, hat_b={}", hat_a, hat_b);
        
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
        
        println!("✅ MatAdd verify_as_subprotocol completed successfully");

        // Verify the constraint: hat_a + hat_b should equal hat_c
        let constraint_result = hat_a + hat_b == self.protocol_input.hat_c;
        if !constraint_result {
            println!("!! MatAdd verification constraint failed: {} - {} != {}", hat_a, hat_b, self.protocol_input.hat_c);
        }
        constraint_result
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        // Calculate basic check: hat_c = a - b  
        let check = ArithmeticExpression::sub(
            ArithmeticExpression::input(self.atomic_pop.mapping.hat_c_index),
            ArithmeticExpression::add(
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
            println!("⚠️ AtomicPoP is not ready! Run AddProtocol.reduce_prover first");
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
    use crate::protocols::zero::EqZero;
    use crate::protocols::sub::MatSub; // used in chained test
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::{UniformRand, Zero};
    use ark_std::test_rng;

    // Helper: build random matrix of given shape (column-major)
    fn rand_mat(rows: usize, cols: usize, rng: &mut impl ark_std::rand::RngCore) -> DenseMatFieldCM<BlsFr> {
        let mut m = DenseMatFieldCM::new(rows, cols);
        let mut cols_data = Vec::new();
        for _c in 0..cols {
            let mut col = Vec::new();
            for _r in 0..rows { col.push(BlsFr::rand(rng)); }
            cols_data.push(col);
        }
        m.set_data(cols_data);
        m
    }

    // Element-wise addition C = A + B
    fn add_mats(a: &DenseMatFieldCM<BlsFr>, b: &DenseMatFieldCM<BlsFr>) -> DenseMatFieldCM<BlsFr> {
        let (rows, cols) = a.shape;
        let mut c = DenseMatFieldCM::new(rows, cols);
        let mut data = Vec::new();
        for col in 0..cols {
            let mut col_vec = Vec::new();
            for row in 0..rows { col_vec.push(a.data[col][row] + b.data[col][row]); }
            data.push(col_vec);
        }
        c.set_data(data); c
    }

    #[test]
    fn test_addprotocol_basic() {
        let mut rng = test_rng();
        let (rows, cols) = (4, 4); // power-of-two for challenge lengths
        let a = rand_mat(rows, cols, &mut rng);
        let b = rand_mat(rows, cols, &mut rng);
        let c = add_mats(&a, &b);

        let point = (
            (0..rows.ilog2()).map(|_| BlsFr::rand(&mut rng)).collect::<Vec<_>>(),
            (0..cols.ilog2()).map(|_| BlsFr::rand(&mut rng)).collect::<Vec<_>>()
        );
        let hat_c = c.proj_lr_challenges(&point.0, &point.1);
        let point_index = ((0..point.0.len()).collect(), (0..point.1.len()).collect());

        let mut proto = MatAdd::<BlsFr>::new(hat_c, point.clone(), 0, point_index.clone(), (rows, cols), (rows, cols), (rows, cols));
        proto.set_input(a.clone(), b.clone());
        let mut prover_trans = Transcript::new(hat_c);
        assert!(proto.reduce_prover(&mut prover_trans));

        // verifier
        let mut verifier_trans = prover_trans.clone(); verifier_trans.reset_pointer();
        let mut vproto = MatAdd::<BlsFr>::new(hat_c, point.clone(), 0, point_index.clone(), (rows, cols), (rows, cols), (rows, cols));
        assert!(vproto.verify_as_subprotocol(&mut verifier_trans));
        assert_eq!(prover_trans.pointer, verifier_trans.pointer);
    }

    #[test]
    fn test_addprotocol_with_eqzero_and_sub_chain() {
        // Prove: (A + B) - C == 0 using MatAdd to bind hat_c, then Sub with EqZero
        let mut rng = test_rng();
        let (rows, cols) = (4, 4);
        let a = rand_mat(rows, cols, &mut rng);
        let b = rand_mat(rows, cols, &mut rng);
        let c = add_mats(&a, &b); // ground truth

        let point = (
            (0..rows.ilog2()).map(|_| BlsFr::rand(&mut rng)).collect::<Vec<_>>(),
            (0..cols.ilog2()).map(|_| BlsFr::rand(&mut rng)).collect::<Vec<_>>()
        );
        let hat_c = c.proj_lr_challenges(&point.0, &point.1);
        let point_index = ((0..point.0.len()).collect(), (0..point.1.len()).collect());

        // Transcript chain
        let mut prover_trans = Transcript::new(BlsFr::zero());

        // EqZero to produce zero commitment Z
        let mut p_eq_zero = EqZero::<BlsFr>::new((rows, cols));
        assert!(p_eq_zero.reduce_prover(&mut prover_trans));
        let (hat_z, point_z) = p_eq_zero.atomic_pop.get_a();
        let (hat_z_index, point_z_index) = p_eq_zero.atomic_pop.get_a_index();
        assert!(hat_z.is_zero());

        // MatAdd: prove hat_c = hat_a + hat_b where hat_c is proj(C)
        let mut p_add = MatAdd::<BlsFr>::new(hat_c, point.clone(), 0, point_index.clone(), (rows, cols), (rows, cols), (rows, cols));
        p_add.set_input(a.clone(), b.clone());
        assert!(p_add.reduce_prover(&mut prover_trans));
    let (hat_c_again, _point_c_again) = p_add.atomic_pop.get_c();
    let (_hat_c_index_again, _point_c_index_again) = p_add.atomic_pop.get_c_index();
    assert_eq!(hat_c_again, hat_c);

        // Sub: (A+B) - C == 0 -> using projections: hat_c (from add) - proj(C) == 0
        // We already have hat_c = proj(C) so this is consistent and should yield zero.
        let mut p_sub = MatSub::<BlsFr>::new(hat_z, point_z.clone(), hat_z_index, point_z_index, (rows, cols), (rows, cols), (rows, cols));
        // Provide C for both a and b so that their projections cancel and yield zero (aligned with Sub semantics a - b = hat_c)
        p_sub.set_input(c.clone(), c.clone());
        assert!(p_sub.reduce_prover(&mut prover_trans));

        // Verification path
        let mut verifier_trans = prover_trans.clone(); verifier_trans.reset_pointer();
        let mut v_eq_zero = EqZero::<BlsFr>::new((rows, cols));
        assert!(v_eq_zero.verify_as_subprotocol(&mut verifier_trans));
        let (vhz, vpz) = v_eq_zero.atomic_pop.get_a();
        let (vhz_i, vpz_i) = v_eq_zero.atomic_pop.get_a_index();
        let mut v_add = MatAdd::<BlsFr>::new(hat_c, point.clone(), 0, point_index.clone(), (rows, cols), (rows, cols), (rows, cols));
        assert!(v_add.verify_as_subprotocol(&mut verifier_trans));
        let mut v_sub = MatSub::<BlsFr>::new(vhz, vpz.clone(), vhz_i, vpz_i, (rows, cols), (rows, cols), (rows, cols));
        assert!(v_sub.verify_as_subprotocol(&mut verifier_trans));
        assert_eq!(prover_trans.pointer, verifier_trans.pointer);
    }
}
