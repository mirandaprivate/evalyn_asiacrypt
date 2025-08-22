//! Implement the Hadamard protocol using LiteBullet
//!

use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use mat::utils::{xi, linear};

use crate::atomic_pop::AtomicPoP;
use crate::atomic_protocol::{AtomicMatProtocol, AtomicMatProtocolInput, MatOp};
use crate::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};

use crate::protocols::litebullet::LiteBullet;


#[derive(Debug, Clone)]
pub struct Hadamard<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
    pub litebullet: LiteBullet<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> Hadamard<F> 
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
            op: MatOp::Hadamard,
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
            hat_c, // use the actual matrix multiplication result
            point_c,
            hat_c_index,
            point_c_index,
        );

        let len = shape_a.0 * shape_a.1;
        let litebullet = LiteBullet::new(hat_c, hat_c_index, len);

        Self {
            protocol_input,
            atomic_pop,
            litebullet,
        }
       
    }

    pub fn set_input(&mut self, mat_a: DenseMatFieldCM<F>, mat_b: DenseMatFieldCM<F>) {

        let xl = self.protocol_input.point_c.0.clone();
        let xr = self.protocol_input.point_c.1.clone();

        // xi_l has length 2^(xl.len()), xi_r has length 2^(xr.len())
        let xi_l = xi::xi_from_challenges::<F>(&xl);
        let xi_r = xi::xi_from_challenges::<F>(&xr);

        let m = xi_l.len(); // This is 2^(xl.len())
        let n = xi_r.len(); // This is 2^(xr.len())


        let vec_a_prime: Vec<F> =
        (0..(m * n)).into_par_iter()
        .map(|idx| {
            let row = idx % m;
            let col = idx / m;
            xi_l[row] * xi_r[col] * F::from(mat_a.data[col][row])
            }
        )
        .collect();

        let vec_b = mat_b.to_vec();

        self.litebullet.set_input(vec_a_prime, vec_b);
    }

}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for Hadamard<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.litebullet.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {

        let log_m = self.protocol_input.point_c.0.len();
        let log_n = self.protocol_input.point_c.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        if (m, n) != self.protocol_input.shape_a || (m, n) != self.protocol_input.shape_b {

            println!("m, n: {:?}", (m, n));
            println!("shape: {:?}", self.protocol_input.shape_a);
            println!("shape: {:?}", self.protocol_input.shape_b);

            panic!("Matrix dimension mismatch");
        }


        let xl = self.protocol_input.point_c.0.clone();
        let xr = self.protocol_input.point_c.1.clone();

        // // swap xl and xr
        let x = [xr.clone(), xl.clone()].concat();
        


        let flag = self.litebullet.reduce_prover(trans);

        let (
            _a_hat, 
            _b_hat,
            challenges,
        ) = (
            self.litebullet.atomic_pop.hat_a,
            self.litebullet.atomic_pop.hat_b,
            self.litebullet.atomic_pop.challenges.clone(),
        );

        let challenges_inv = challenges.iter().map(|c| c.inverse().unwrap()).collect::<Vec<_>>();

        let x_prime =
        linear::vec_element_wise_mul(&x, &challenges_inv);

        let xl_a =
        x_prime[log_n..].to_vec();
        let xr_a =
        x_prime[..log_n].to_vec();

        let mut xl_a_indices = Vec::new();
        for elem in xl_a.clone() {
            xl_a_indices.push(trans.pointer);
            trans.push_response(elem);
        }

        let mut xr_a_indices = Vec::new();
        for elem in xr_a.clone() {
            xr_a_indices.push(trans.pointer);
            trans.push_response(elem);
        }

        let xl_b =
        challenges[log_n..].to_vec();
        let xr_b =
        challenges[..log_n].to_vec();

        let mut xl_b_indices = Vec::new();
        for elem in xl_b.clone() {
            xl_b_indices.push(trans.pointer);
            trans.push_response(elem);
        }

        let mut xr_b_indices = Vec::new();
        for elem in xr_b.clone() {
            xr_b_indices.push(trans.pointer);
            trans.push_response(elem);
        }

        self.atomic_pop.set_pop_trans(
            self.litebullet.atomic_pop.hat_a,
            self.litebullet.atomic_pop.hat_b,
            (xl_a, xr_a),
            (xl_b, xr_b),
            self.litebullet.atomic_pop.challenges.clone(),
            self.litebullet.atomic_pop.responses.clone(),
            self.litebullet.atomic_pop.mapping.hat_a_index,
            self.litebullet.atomic_pop.mapping.hat_b_index,
            (xl_a_indices, xr_a_indices),
            (xl_b_indices, xr_b_indices),
            self.litebullet.atomic_pop.mapping.challenges_index.clone(),
            self.litebullet.atomic_pop.mapping.responses_index.clone(),
        );

        flag

    }

    fn verify_as_subprotocol(
        &mut self,
        trans: &mut Transcript<F>,
    ) -> bool 
    {

        let log_m = self.protocol_input.point_c.0.len();
        let log_n = self.protocol_input.point_c.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let xl = self.protocol_input.point_c.0.clone();
        let xr = self.protocol_input.point_c.1.clone();

        // swap xl and xr
        let x = [xr, xl].concat();

        let len = m * n;

        self.litebullet = LiteBullet::new(
            self.protocol_input.hat_c,
            0,
            len,
        );

        let flag = self.litebullet.verify_as_subprotocol(trans);

        let (
            _a_hat,
            _b_hat,
            challenges,
        ) = (
            self.litebullet.atomic_pop.hat_a,
            self.litebullet.atomic_pop.hat_b,
            self.litebullet.atomic_pop.challenges.clone(),
        );

        // Additional safety check: ensure challenges has the expected length
        let expected_len = log_m + log_n;
        if challenges.len() != expected_len {
            println!("!! Hadamard protocol failed: challenges length mismatch. Expected {}, got {}", expected_len, challenges.len());
            return false;
        }

        let challenges_inv = challenges.iter().map(|c| c.inverse().unwrap()).collect::<Vec<_>>();

        let x_prime =
        linear::vec_element_wise_mul(&x, &challenges_inv);

        // Safety check: ensure x_prime has sufficient length
        if x_prime.len() < log_n {
            println!("!! Hadamard protocol failed: x_prime length {} is less than log_n {}", x_prime.len(), log_n);
            return false;
        }

        let xl_a =
        x_prime[log_n..].to_vec();
        let xr_a =
        x_prime[..log_n].to_vec();

        let mut xl_a_indices = Vec::new();
        for _ in 0..xl_a.len() {
            xl_a_indices.push(trans.pointer);
            trans.pointer += 1;
        }

        let mut xr_a_indices = Vec::new();
        for _ in 0..xr_a.len() {
            xr_a_indices.push(trans.pointer);
            trans.pointer += 1;
        }

        let xl_b =
        challenges[log_n..].to_vec();
        let xr_b =
        challenges[..log_n].to_vec();

        let mut xl_b_indices = Vec::new();
        for _ in 0..xl_b.len() {
            xl_b_indices.push(trans.pointer);
            trans.pointer += 1;
        }

        let mut xr_b_indices = Vec::new();
        for _ in 0..xr_b.len() {
            xr_b_indices.push(trans.pointer);
            trans.pointer += 1;
        }


        self.atomic_pop.set_pop_trans(
            self.litebullet.atomic_pop.hat_a,
            self.litebullet.atomic_pop.hat_b,
            (xl_a, xr_a),
            (xl_b, xr_b),
            self.litebullet.atomic_pop.challenges.clone(),
            self.litebullet.atomic_pop.responses.clone(),
            self.litebullet.atomic_pop.mapping.hat_a_index,
            self.litebullet.atomic_pop.mapping.hat_b_index,
            (xl_a_indices, xr_a_indices),
            (xl_b_indices, xr_b_indices),
            self.litebullet.atomic_pop.mapping.challenges_index.clone(),
            self.litebullet.atomic_pop.mapping.responses_index.clone(),
        );

        flag
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        let log_m = self.protocol_input.point_c.0.len();
        let log_n = self.protocol_input.point_c.1.len();

        // Calculate basic check: hat_c = <la, br>
        let check = ArithmeticExpression::input(0); // Placeholder
        
        // Set up basic linking constraints
        // xl_a \circ challenges_r = xl
        // xr_a \circ challenges_l = xr
        // xl_b = challenges_r
        // xr_b = challenges_l
        // where xl = point_c.0 and xr = point_c.1

        let mut link_xa_l = Vec::new();
        let mut link_xa_r = Vec::new();
        let mut link_xb_l = Vec::new();
        let mut link_xb_r = Vec::new();

        for i in 0..log_m {
            link_xa_l.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::mul(
                        ArithmeticExpression::input(self.atomic_pop.mapping.point_a_index.0[i]),
                        ArithmeticExpression::input(self.atomic_pop.mapping.challenges_index[log_n + i]),
                    ),
                    ArithmeticExpression::input(self.atomic_pop.mapping.point_c_index.0[i]),
                )
            );

            link_xb_l.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::input(self.atomic_pop.mapping.point_b_index.0[i]),
                    ArithmeticExpression::input(self.atomic_pop.mapping.challenges_index[log_n + i]),
                )
            );
        }

        for i in 0..log_n {
            link_xa_r.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::mul(
                        ArithmeticExpression::input(self.atomic_pop.mapping.point_a_index.1[i]),
                        ArithmeticExpression::input(self.atomic_pop.mapping.challenges_index[i]),
                    ),
                    ArithmeticExpression::input(self.atomic_pop.mapping.point_c_index.1[i]),
                )
            );

            link_xb_r.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::input(self.atomic_pop.mapping.point_b_index.1[i]),
                    ArithmeticExpression::input(self.atomic_pop.mapping.challenges_index[i]),
                )
            );
        }

        let link_xa = (link_xa_l, link_xa_r);
        let link_xb = (link_xb_l, link_xb_r);

        self.atomic_pop.set_check(check);
        self.atomic_pop.set_link_xa(link_xa);
        self.atomic_pop.set_link_xb(link_xb);

        let flag = self.litebullet.prepare_atomic_pop();
        if !flag {
            println!("⚠️ LiteBullet AtomicPoP preparation failed! Run LiteBullet.reduce_prover first");
            return false;
        }

        self.atomic_pop.is_ready()
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {

        let flag = self.litebullet.synthesize_atomic_pop_constraints(cs_builder);
        self.atomic_pop.synthesize_constraints(cs_builder);

        flag && self.atomic_pop.is_ready()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::sub::MatSub;
    use crate::protocols::zero::EqZero;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use ark_std::{test_rng, UniformRand};
    use fsproof::helper_trans::Transcript;
    use mat::utils::matdef::DenseMatFieldCM;

    // Helper function to create random matrices for testing
    fn create_random_matrix(m: usize, n: usize) -> DenseMatFieldCM<BlsFr> {
        let mut rng = test_rng();
        let mut mat = DenseMatFieldCM::new(m, n);
        let mut data = vec![vec![BlsFr::zero(); m]; n];
        for i in 0..n {
            for j in 0..m {
                data[i][j] = BlsFr::rand(&mut rng);
            }
        }
        mat.set_data(data);
        mat
    }

    #[test]
    fn test_hadamard_new() {
        println!("=== Testing MatHadamard::new ===");
        let mut rng = test_rng();
        let hat_c = BlsFr::rand(&mut rng);
        let point_c = (vec![BlsFr::rand(&mut rng)], vec![BlsFr::rand(&mut rng)]);
        let hadamard = Hadamard::<BlsFr>::new(hat_c, point_c.clone(), 0, (vec![0], vec![1]), (2, 2), (2, 2), (2, 2));

        assert_eq!(hadamard.protocol_input.op, MatOp::Hadamard);
        assert_eq!(hadamard.protocol_input.shape_a, (2, 2));
        assert_eq!(hadamard.protocol_input.hat_c, hat_c);
        assert_eq!(hadamard.protocol_input.point_c, point_c);
        println!("✅ MatHadamard::new test passed");
    }

    #[test]
    fn test_hadamard_protocol_flow() {
        println!("=== Testing MatHadamard protocol flow ===");
        let mut rng = test_rng();
        let shape = (4, 4);
        let mat_a = create_random_matrix(shape.0, shape.1);
        let mat_b = create_random_matrix(shape.0, shape.1);
        let mat_c = mat_a.hadamard_prod(&mat_b);

        let point_c = (
            (0..(shape.0 as u32).ilog2()).map(|_| BlsFr::rand(&mut rng)).collect(),
            (0..(shape.1 as u32).ilog2()).map(|_| BlsFr::rand(&mut rng)).collect(),
        );
        let hat_c = mat_c.proj_lr_challenges(&point_c.0, &point_c.1);

        let mut hadamard = Hadamard::<BlsFr>::new(hat_c, point_c, 0, (vec![], vec![]), shape, shape, shape);
        hadamard.set_input(mat_a, mat_b);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        let prover_result = hadamard.reduce_prover(&mut prover_trans);
        assert!(prover_result, "Prover should succeed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let verifier_result = hadamard.verify_as_subprotocol(&mut verifier_trans);
        assert!(verifier_result, "Verifier should succeed");

        println!("✅ MatHadamard protocol flow test passed");
    }

    #[test]
    fn test_integrated_hadamard_equation_proof() {
        println!("=== Testing integrated proof: mat_a o mat_b - mat_c = 0 ===");
        let shape = (4, 4);

        // --- Test Data Setup ---
        let mat_a = create_random_matrix(shape.0, shape.1);
        let mat_b = create_random_matrix(shape.0, shape.1);
        let mat_c = mat_a.hadamard_prod(&mat_b);
        let mat_d = mat_a.hadamard_prod(&mat_b); // This is the result of the first operation

        // --- Prover Side ---
        let mut prover_trans = Transcript::new(BlsFr::zero());

        // 1. EqZero for the final zero matrix
        let mut eq_zero_protocol = EqZero::<BlsFr>::new(shape);
        assert!(eq_zero_protocol.reduce_prover(&mut prover_trans));

        let (hat_c_from_zero, point_c_from_zero) = eq_zero_protocol.atomic_pop.get_a();
        let (hat_c_idx_from_zero, point_c_idx_from_zero) = eq_zero_protocol.atomic_pop.get_a_index();

        // 2. MatSub for (mat_a o mat_b) - mat_c
        let mut sub_protocol = MatSub::<BlsFr>::new(
            hat_c_from_zero, point_c_from_zero, hat_c_idx_from_zero, point_c_idx_from_zero,
            shape, shape, shape
        );
        sub_protocol.set_input(mat_d, mat_c.clone());
        assert!(sub_protocol.reduce_prover(&mut prover_trans));

        let (hat_c_from_sub, point_c_from_sub) = sub_protocol.atomic_pop.get_a();
        let (hat_c_idx_from_sub, point_c_idx_from_sub) = sub_protocol.atomic_pop.get_a_index();

        // 3. MatHadamard for mat_a o mat_b
        let mut hadamard_protocol = Hadamard::<BlsFr>::new(
            hat_c_from_sub, point_c_from_sub, hat_c_idx_from_sub, point_c_idx_from_sub,
            shape, shape, shape
        );
        hadamard_protocol.set_input(mat_a.clone(), mat_b.clone());
        assert!(hadamard_protocol.reduce_prover(&mut prover_trans));

        assert_eq!(prover_trans.pointer, prover_trans.trans_seq.len(), "Prover transcript should be fully consumed");

        // --- Verifier Side ---
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();

        // 1. Verify EqZero
        let mut eq_zero_verifier = EqZero::<BlsFr>::new(shape);
        assert!(eq_zero_verifier.verify_as_subprotocol(&mut verifier_trans));
        let (hat_c_v_zero, point_c_v_zero) = eq_zero_verifier.atomic_pop.get_a();
        let (hat_c_idx_v_zero, point_c_idx_v_zero) = eq_zero_verifier.atomic_pop.get_a_index();

        // 2. Verify MatSub
        let mut sub_verifier = MatSub::<BlsFr>::new(
            hat_c_v_zero, point_c_v_zero, hat_c_idx_v_zero, point_c_idx_v_zero,
            shape, shape, shape
        );
        assert!(sub_verifier.verify_as_subprotocol(&mut verifier_trans));
        let (hat_c_v_sub, point_c_v_sub) = sub_verifier.atomic_pop.get_a();
        let (hat_c_idx_v_sub, point_c_idx_v_sub) = sub_verifier.atomic_pop.get_a_index();

        // 3. Verify MatHadamard
        let mut hadamard_verifier = Hadamard::<BlsFr>::new(
            hat_c_v_sub, point_c_v_sub, hat_c_idx_v_sub, point_c_idx_v_sub,
            shape, shape, shape
        );
        assert!(hadamard_verifier.verify_as_subprotocol(&mut verifier_trans));

        assert_eq!(verifier_trans.pointer, verifier_trans.trans_seq.len(), "Verifier transcript should be fully consumed");

        // --- Final Checks ---
        let (c_proj, c_point) = sub_verifier.atomic_pop.get_b();
        let (a_proj, a_point) = hadamard_verifier.atomic_pop.get_a();
        let (b_proj, b_point) = hadamard_verifier.atomic_pop.get_b();

        let c_proj_expected = mat_c.proj_lr_challenges(&c_point.0, &c_point.1);
        let a_proj_expected = mat_a.proj_lr_challenges(&a_point.0, &a_point.1);
        let b_proj_expected = mat_b.proj_lr_challenges(&b_point.0, &b_point.1);

        assert_eq!(c_proj, c_proj_expected, "Projection of mat_c should match expected");
        assert_eq!(a_proj, a_proj_expected, "Projection of mat_a should match expected");
        assert_eq!(b_proj, b_proj_expected, "Projection of mat_b should match expected");
        
        println!("✅ Integrated Hadamard equation proof test passed");
    }
}