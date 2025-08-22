//! Concatenates a series of matrices into a single large vector.
//!
//! Currently, this function assumes all input matrices have the same shape.
//! Support for accommodating matrices with heterogeneous shapes is pending.
//! 
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use mat::utils::{xi, linear};

use crate::atomic_pop::AtomicMultiPoP;
use crate::atomic_protocol::{AtomicMatProtocolMultiInput, MatOp, AtomicMatProtocol};
use crate::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};


#[derive(Debug, Clone)]
pub struct Concat<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolMultiInput<F>,
    pub atomic_pop: AtomicMultiPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> Concat<F> 
{
    pub fn new(
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
        shape: (usize, usize),
        num: usize,
     ) -> Self {
        if !shape.0.is_power_of_two() || !shape.1.is_power_of_two() {
            // Handle non-power-of-two case
            panic!("Input dimensions must be powers of two in Concat");
        }

        let protocol_input = AtomicMatProtocolMultiInput {
            op: MatOp::Concat,
            hat_c: hat_c.clone(),
            point_c: point_c.clone(),
            shape_inputs: shape.clone(),
            num_inputs: num,
            input_mats: Vec::new(),
        };

        let mut atomic_pop = AtomicMultiPoP::new();
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

    pub fn set_input(&mut self, input_mats: Vec<DenseMatFieldCM<F>>) {
        self.protocol_input.input_mats = input_mats;
    }

}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for Concat<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        let input_row_num = self.protocol_input.shape_inputs.0;
        let input_col_num = self.protocol_input.shape_inputs.1;

        let single_len = self.protocol_input.shape_inputs.0 * self.protocol_input.shape_inputs.1;
        let num = self.protocol_input.num_inputs;
        let c_len = (single_len * num).next_power_of_two();


        let log_single_len = single_len.ilog2() as usize;
        let log_n = c_len.ilog2() as usize;
        let log_input_row_num = input_row_num.ilog2() as usize;
        let log_input_col_num = input_col_num.ilog2() as usize;

        if self.protocol_input.point_c.0.len() != log_n || self.protocol_input.point_c.1.len() != 0 {
            panic!("!! Invalid point_c shapes in Concat!");
        }

        let point_c_l = self.protocol_input.point_c.0.clone();
        let point_input_lr = point_c_l[log_n - log_single_len..].to_vec();


        let point_input_r = point_input_lr[..log_input_col_num].to_vec();
        let point_input_l = point_input_lr[log_input_col_num..].to_vec();


        let xxxx = point_c_l[..log_n - log_single_len].to_vec();

        let xi_num = xi::xi_from_challenges(&xxxx);
        let multiplicant_inputs = xi_num[..num].to_vec();

        let mut hat_inputs = Vec::new();

        for i in 0..num {
            let hat_input = self.protocol_input.input_mats[i].proj_lr_challenges(&point_input_l, &point_input_r);
            hat_inputs.push(hat_input);
        }

        let flag = self.protocol_input.hat_c == linear::inner_product(&hat_inputs, &multiplicant_inputs);

        if !flag {
            println!("[concat reduce_prover] hat_c check failed! hat_c: {:?}, expected: {:?}", self.protocol_input.hat_c, linear::inner_product(&hat_inputs, &multiplicant_inputs));
            panic!("!!!!!! Invalid hat_c in Concat!");
        }

        let mut hat_inputs_index = Vec::new();
        for i in 0..num {
            hat_inputs_index.push(trans.pointer);
            trans.push_response(hat_inputs[i]);
        }

        let mut point_input_index_l = Vec::new();
        for i in 0..log_input_row_num {
            point_input_index_l.push(trans.pointer);
            trans.push_response(point_input_l[i]);
        }

        let mut point_input_index_r = Vec::new();
        for i in 0..log_input_col_num {
            point_input_index_r.push(trans.pointer);
            trans.push_response(point_input_r[i]);
        }

        let point_inputs = vec![(point_input_l, point_input_r); num];
        let point_inputs_index = vec![(point_input_index_l, point_input_index_r); num];
        

        self.atomic_pop.set_pop_trans(
            hat_inputs,
            point_inputs,
            Vec::new(),
            Vec::new(),
            hat_inputs_index,
            point_inputs_index,
            Vec::new(),
            Vec::new(),
        );

        flag

    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let input_row_num = self.protocol_input.shape_inputs.0;
        let input_col_num = self.protocol_input.shape_inputs.1;

        let single_len = self.protocol_input.shape_inputs.0 * self.protocol_input.shape_inputs.1;
        let num = self.protocol_input.num_inputs;
        let c_len = (single_len * num).next_power_of_two();


        let log_single_len = single_len.ilog2() as usize;
        let log_n = c_len.ilog2() as usize;
        let log_input_row_num = input_row_num.ilog2() as usize;
        let log_input_col_num = input_col_num.ilog2() as usize;

        if self.protocol_input.point_c.0.len() != log_n || self.protocol_input.point_c.1.len() != 0 {
            panic!("!! Invalid point_c shapes in Concat!");
        }

        let point_c_l = self.protocol_input.point_c.0.clone();
        let point_input_lr = point_c_l[log_n - log_single_len..].to_vec();

        let point_input_r = point_input_lr[..log_input_col_num].to_vec();
        let point_input_l = point_input_lr[log_input_col_num..].to_vec();

        let xxxx = point_c_l[..log_n - log_single_len].to_vec();

        let xi_num = xi::xi_from_challenges(&xxxx);
        let multiplicant_inputs = xi_num[..num].to_vec();

       

        let mut hat_inputs_index = Vec::new();
        let mut hat_inputs = Vec::new();
        for _i in 0..num {
            hat_inputs_index.push(trans.pointer);
            hat_inputs.push(trans.get_at_position(trans.pointer));
            trans.pointer += 1;
        }

        let flag = self.protocol_input.hat_c == linear::inner_product(&hat_inputs, &multiplicant_inputs);

        if !flag {
            println!("[concat verify] hat_c check failed! hat_c: {:?}, expected: {:?}", self.protocol_input.hat_c, linear::inner_product(&hat_inputs, &multiplicant_inputs));
            panic!("!!!!!! Invalid hat_c in Concat!");
        }


        let mut point_input_index_l = Vec::new();
        for _i in 0..log_input_row_num {
            point_input_index_l.push(trans.pointer);
            trans.pointer += 1;
        }

        let mut point_input_index_r = Vec::new();
        for _i in 0..log_input_col_num {
            point_input_index_r.push(trans.pointer);
            trans.pointer += 1;
        }

        let point_inputs = vec![(point_input_l, point_input_r); num];
        let point_inputs_index = vec![(point_input_index_l, point_input_index_r); num];
        

        self.atomic_pop.set_pop_trans(
            hat_inputs,
            point_inputs,
            Vec::new(),
            Vec::new(),
            hat_inputs_index,
            point_inputs_index,
            Vec::new(),
            Vec::new(),
        );

        flag
    }

    fn prepare_atomic_pop(&mut self) -> bool {
        // Only require base transcript data (ready.0). We'll build check & mark ready here.
        if !self.atomic_pop.ready.0 {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop base data not ready before generating constraints (Concat)");
            return false;
        }

        let input_row_num = self.protocol_input.shape_inputs.0;
        let input_col_num = self.protocol_input.shape_inputs.1;

        let single_len = self.protocol_input.shape_inputs.0 * self.protocol_input.shape_inputs.1;
        let num = self.protocol_input.num_inputs;
        let c_len = (single_len * num).next_power_of_two();


        let log_single_len = single_len.ilog2() as usize;
        let log_n = c_len.ilog2() as usize;
        let _log_input_row_num = input_row_num.ilog2() as usize;
        let _log_input_col_num = input_col_num.ilog2() as usize;

        let hat_inputs_index = self.atomic_pop.mapping.hat_inputs_index.clone();
        let point_c_l_index = self.atomic_pop.mapping.point_c_index.0.clone();
        let hat_c_index = self.atomic_pop.mapping.hat_c_index.clone();
        let point_input_index = [
            self.atomic_pop.mapping.point_inputs_index[0].1.as_slice(),
            self.atomic_pop.mapping.point_inputs_index[0].0.as_slice(),
        ]
        .concat();

        // println!("[concat prepare] hat_inputs_index len: {}", hat_inputs_index.len());
        // println!("[concat prepare] point_c_l_index len: {}", point_c_l_index.len());
        // println!("[concat prepare] point_input_index len: {}", point_input_index.len());
        // println!("[concat prepare] log_single_len: {}", log_single_len);


        let hat_input_exprs: Vec<ArithmeticExpression<F>> = hat_inputs_index
            .iter()
            .map(|i| ArithmeticExpression::input(*i))
            .collect();
        let hat_c_expr = ArithmeticExpression::input(hat_c_index);
        let point_input_exprs: Vec<ArithmeticExpression<F>> = point_input_index
            .iter()
            .map(|i| ArithmeticExpression::input(*i))
            .collect();
        let point_c_l_exprs: Vec<ArithmeticExpression<F>> = point_c_l_index
            .iter()
            .map(|i| ArithmeticExpression::<F>::input(*i))
            .collect();
        let xxxx_exprs = point_c_l_exprs[..log_n - log_single_len].to_vec();

        let mut cur_vec_expr = vec![ArithmeticExpression::constant(F::one())];

        // Important: xi_from_challenges applies challenges from last to first.
        // To match that order, iterate xxxx in reverse when expanding the tensor.
        for xx_expr in xxxx_exprs.iter().rev() {
            let vec_l_expr = cur_vec_expr.clone();
            let vec_r_expr: Vec<ArithmeticExpression<F>> = cur_vec_expr
                .iter()
                .map(|e| ArithmeticExpression::mul(e.clone(), xx_expr.clone()))
                .collect();
            cur_vec_expr = vec_l_expr.into_iter().chain(vec_r_expr).collect();
        }


        let multiplicant_exprs = cur_vec_expr[..num].to_vec();

        let mut check = hat_c_expr;

        for i in 0..num {
            check = ArithmeticExpression::sub(
                check,
                ArithmeticExpression::mul(
                    multiplicant_exprs[i].clone(),
                    hat_input_exprs[i].clone(),
                )
            );
        }


        // Set up the main GrandProd atomic_pop's check and links
        self.atomic_pop.set_check(check);

        let mut link_exprs = Vec::new();
        for i in 0..log_single_len {
            let link = ArithmeticExpression::sub(
                point_input_exprs[i].clone(),
                point_c_l_exprs[log_n - log_single_len + i].clone(),
            );
            link_exprs.push(link);
        }

        self.atomic_pop.set_link_inputs(link_exprs);
        
        self.atomic_pop.is_ready()
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in Concat");
            return false;
        }
        self.atomic_pop.synthesize_constraints(cs_builder)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic_protocol::AtomicMatProtocol;
    use crate::protocols::sub::MatSub;
    use crate::protocols::zero::EqZero;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use ark_std::test_rng;
    use mat::utils::matdef::DenseMatFieldCM;

    #[test]
    fn test_concatprotocol_with_eqzero_and_sub() {
        // 1. Setup parameters
        let _rng = test_rng();
        let num_mats = 4;
        let mat_shape = (2, 2);
        let single_len = mat_shape.0 * mat_shape.1;
        let total_len = num_mats * single_len as usize;
        let c_shape = (total_len.next_power_of_two(), 1);

        // 2. Generate matrices a_i and the concatenated vector c
        let mut input_mats = Vec::new();
        let mut c_data_flat = Vec::new();

        for i in 0..num_mats {
            let mut mat = DenseMatFieldCM::new(mat_shape.0, mat_shape.1);
            let mut mat_data = Vec::new();
            for r in 0..mat_shape.0 {
                let mut row = Vec::new();
                for c in 0..mat_shape.1 {
                    let val = BlsFr::from((i * single_len + r * mat_shape.1 + c + 1) as u64);
                    row.push(val);
                }
                mat_data.push(row);
            }
            mat.set_data(mat_data);

            // Flatten and append to c_data_flat
            for r in 0..mat_shape.0 {
                for c in 0..mat_shape.1 {
                    c_data_flat.push(mat.data[r][c]);
                }
            }
            input_mats.push(mat);
        }

        // Pad c_data_flat to the next power of two
        while c_data_flat.len() < c_shape.0 {
            c_data_flat.push(BlsFr::zero());
        }

        // Create matrix c
        let mut mat_c = DenseMatFieldCM::new(c_shape.0, c_shape.1);
        mat_c.set_data(vec![c_data_flat]);

        // 3. Setup the protocol chain: EqZero(Sub(c, Concat(a_i)))
        let mut prover_trans = Transcript::new(BlsFr::zero());
       

        // We want to prove that a matrix is zero.
        // The matrix is the result of `mat_c - Concat(a_i)`.
        // Let's start with EqZero.
        let mut eq_zero_protocol = EqZero::<BlsFr>::new(c_shape);
        // The input to EqZero is the result of the subtraction.
        // We don't have it yet, but we can run the protocol.
        // EqZero will generate challenges for the projection.
        eq_zero_protocol.reduce_prover(&mut prover_trans);

        println!("✅ EqZero reduce_prover completed successfully");

        // The `a` part of EqZero's atomic pop is the projection of the matrix that should be zero.
        // This projection should be zero.
        let (hat_z, point_z) = eq_zero_protocol.atomic_pop.get_a();
        let (hat_z_index, point_z_index) = eq_zero_protocol.atomic_pop.get_a_index();
        assert_eq!(hat_z, BlsFr::zero());

        println!("point_z len: {:?}", point_z.0.len());

        // Now, we use MatSub to state that the projected matrix is the result of a subtraction.
        // hat_z = proj(c, z) - proj(Concat(a_i), z)
        // So, proj(c, z) = proj(Concat(a_i), z)
        let mut sub_protocol =
            MatSub::<BlsFr>::new(hat_z, point_z.clone(), hat_z_index, point_z_index, c_shape, c_shape, c_shape);

        // The inputs to sub_protocol are mat_c and the conceptual Concat(a_i).
        // We need to set the inputs for the sub_protocol to work correctly.
        sub_protocol.set_input(mat_c.clone(), mat_c.clone());
        sub_protocol.reduce_prover(&mut prover_trans);

        // From the sub_protocol, `a` is `c` and `b` is `Concat(a_i)`.
        let (hat_c_from_sub, point_c_from_sub) = sub_protocol.atomic_pop.get_a();
        let (_hat_c_from_sub_index, _point_c_from_sub_index) = sub_protocol.atomic_pop.get_a_index();

        let (hat_concat, point_concat) = sub_protocol.atomic_pop.get_b();
        let (hat_concat_index, point_concat_index) = sub_protocol.atomic_pop.get_b_index();

        // The points must be the same
        assert_eq!(point_z, point_c_from_sub);
        assert_eq!(point_z, point_concat);

        println!("✅ Sub protocol test passed");
        // The projection of c should be consistent.
        let expected_hat_c = mat_c.proj_lr_challenges(&point_z.0, &point_z.1);
        assert_eq!(hat_c_from_sub, expected_hat_c);

        // Now, we use the Concat protocol to handle the projection of the concatenated matrices.
        let mut concat_protocol = Concat::<BlsFr>::new(
            hat_concat,
            point_concat,
            hat_concat_index,
            point_concat_index,
            mat_shape,
            num_mats,
        );
        concat_protocol.set_input(input_mats);
        concat_protocol.reduce_prover(&mut prover_trans);

        // Verification
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();

        let mut v_eq_zero = EqZero::<BlsFr>::new(c_shape);
        assert!(v_eq_zero.verify_as_subprotocol(&mut verifier_trans));
        let (v_hat_z, v_point_z) = v_eq_zero.atomic_pop.get_a();
        let (v_hat_z_index, v_point_z_index) = v_eq_zero.atomic_pop.get_a_index();

        let mut v_sub = MatSub::<BlsFr>::new(
            v_hat_z,
            v_point_z,
            v_hat_z_index,
            v_point_z_index,
            c_shape,
            c_shape,
            c_shape,
        );
        assert!(v_sub.verify_as_subprotocol(&mut verifier_trans));
        let (v_hat_concat, v_point_concat) = v_sub.atomic_pop.get_b();
        let (v_hat_concat_index, v_point_concat_index) = v_sub.atomic_pop.get_b_index();

        let mut v_concat = Concat::<BlsFr>::new(
            v_hat_concat,
            v_point_concat,
            v_hat_concat_index,
            v_point_concat_index,
            mat_shape,
            num_mats,
        );
        assert!(v_concat.verify_as_subprotocol(&mut verifier_trans));

        assert_eq!(prover_trans.pointer, verifier_trans.pointer);
        println!("✅ Concat protocol test with EqZero and MatSub passed!");
    }

    #[test]
    fn test_concatprotocol_constraints() {
        // 1. Setup parameters
        let _rng = test_rng();
        let num_mats = 4;
        let mat_shape = (2, 2);
        let single_len = mat_shape.0 * mat_shape.1;
        let total_len = num_mats * single_len as usize;
        let c_shape = (total_len.next_power_of_two(), 1);

        // 2. Generate col major matrices a_i and the concatenated vector c
        let mut input_mats = Vec::new();
        let mut c_data_flat = Vec::new();

        for i in 0..num_mats {
            let mut mat = DenseMatFieldCM::new(mat_shape.0, mat_shape.1);
            let mut mat_data_cols = Vec::new();
            
            // Create col major matrix data
            for c in 0..mat_shape.1 {
                let mut col = Vec::new();
                for r in 0..mat_shape.0 {
                    let val = BlsFr::from((i * 100 + r * 10 + c + 1) as u64);
                    col.push(val);
                }
                mat_data_cols.push(col);
            }
            mat.set_data(mat_data_cols);

            // Flatten each col major matrix by concatenating columns
            for c in 0..mat_shape.1 {
                for r in 0..mat_shape.0 {
                    c_data_flat.push(mat.data[c][r]);
                }
            }
            input_mats.push(mat);
        }

        while c_data_flat.len() < c_shape.0 {
            c_data_flat.push(BlsFr::zero());
        }

        let mut mat_c = DenseMatFieldCM::new(c_shape.0, c_shape.1);
        mat_c.set_data(vec![c_data_flat]);


        // 3. Prover side: run the protocol chain
        let mut prover_trans = Transcript::new(BlsFr::zero());

        let mut p_eq_zero = EqZero::<BlsFr>::new(c_shape);
        assert!(p_eq_zero.reduce_prover(&mut prover_trans), "Prover EqZero failed");
        let (hat_z, point_z) = p_eq_zero.atomic_pop.get_a();
        let (hat_z_index, point_z_index) = p_eq_zero.atomic_pop.get_a_index();

        let mut p_sub = MatSub::<BlsFr>::new(hat_z, point_z.clone(), hat_z_index, point_z_index, c_shape, c_shape, c_shape);
        p_sub.set_input(mat_c.clone(), mat_c.clone());
        assert!(p_sub.reduce_prover(&mut prover_trans), "Prover MatSub failed");
        let (hat_concat, point_concat) = p_sub.atomic_pop.get_b();
        let (hat_concat_index, point_concat_index) = p_sub.atomic_pop.get_b_index();

        let mut p_concat = Concat::<BlsFr>::new(
            hat_concat,
            point_concat,
            hat_concat_index,
            point_concat_index,
            mat_shape,
            num_mats,
        );
        p_concat.set_input(input_mats.clone());
        assert!(p_concat.reduce_prover(&mut prover_trans), "Prover Concat failed");

        // 4. Verifier side: run the protocol chain
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();

        let mut v_eq_zero = EqZero::<BlsFr>::new(c_shape);
        assert!(v_eq_zero.verify_as_subprotocol(&mut verifier_trans), "Verifier EqZero failed");
        let (v_hat_z, v_point_z) = v_eq_zero.atomic_pop.get_a();
        let (v_hat_z_index, v_point_z_index) = v_eq_zero.atomic_pop.get_a_index();

        let mut v_sub = MatSub::<BlsFr>::new(v_hat_z, v_point_z, v_hat_z_index, v_point_z_index, c_shape, c_shape, c_shape);
        assert!(v_sub.verify_as_subprotocol(&mut verifier_trans), "Verifier MatSub failed");
        let (v_hat_concat, v_point_concat) = v_sub.atomic_pop.get_b();
        let (v_hat_concat_index, v_point_concat_index) = v_sub.atomic_pop.get_b_index();

        let mut v_concat = Concat::<BlsFr>::new(
            v_hat_concat,
            v_point_concat,
            v_hat_concat_index,
            v_point_concat_index,
            mat_shape,
            num_mats,
        );
        assert!(v_concat.verify_as_subprotocol(&mut verifier_trans), "Verifier Concat failed");

        // 5. Prepare Atomic PoPs for constraint generation
        assert!(v_concat.prepare_atomic_pop(), "Concat prepare_atomic_pop failed");
        assert!(v_sub.prepare_atomic_pop(), "MatSub prepare_atomic_pop failed");
        assert!(v_eq_zero.prepare_atomic_pop(), "EqZero prepare_atomic_pop failed");

        // 6. Synthesize and validate constraints
        let mut cs_builder = ConstraintSystemBuilder::new();
        let mut proof_vec = vec![BlsFr::zero()];
        proof_vec.extend(prover_trans.get_fs_proof_vec());
    cs_builder.set_private_inputs(proof_vec);

        assert!(v_concat.synthesize_atomic_pop_constraints(&mut cs_builder), "Concat synthesize failed");
        assert!(v_sub.synthesize_atomic_pop_constraints(&mut cs_builder), "MatSub synthesize failed");
        assert!(v_eq_zero.synthesize_atomic_pop_constraints(&mut cs_builder), "EqZero synthesize failed");

    let validation_result = cs_builder.validate_constraints();
        assert!(validation_result.is_ok(), "Constraints not satisfied: {:?}", validation_result.err());

        println!("✅ Concat constraints test passed!");
    }
}

