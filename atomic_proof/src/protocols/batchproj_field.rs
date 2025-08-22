/// Batch multiple projections into a single large vector projection
/// 
/// 
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use mat::utils::{xi, linear};

use crate::atomic_protocol::AtomicMatProtocol; // removed unused AtomicMatProtocolMultiInput, MatOp
use crate::pop::{
    arithmetic_expression,
    arithmetic_expression::{
        ArithmeticExpression,
        ConstraintSystemBuilder
    }
};

use crate::protocols::litebullet::LiteBullet; // removed unused NSHARD constant

#[derive(Debug, Clone)]
pub struct BatchProjFieldProtocolInput<F: PrimeField> {
    pub hat_inputs: Vec<F>,
    pub point_inputs: Vec<(Vec<F>, Vec<F>)>,
    pub input_mats: Vec<DenseMatFieldCM<F>>,
    pub input_shape: (usize, usize),
    pub num_inputs: usize,
    pub ready: bool,
}

#[derive(Debug, Clone)]
pub struct BatchProjFieldAtomicPoP<F: PrimeField> {
    pub hat_inputs: Vec<F>,
    pub point_inputs: Vec<(Vec<F>, Vec<F>)>,
    pub c_hat: F,
    pub c_point: (Vec<F>, Vec<F>),
    pub c_shape: (usize, usize),
    pub challenge: F,
    pub response: F,
    pub mapping: BatchProjFieldMapping,
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    pub ready: (bool, bool, bool),
}

#[derive(Debug, Clone)]
pub struct BatchProjFieldMapping {
    pub hat_inputs_index: Vec<usize>,
    pub point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub c_hat_index: usize,
    pub c_point_index: (Vec<usize>, Vec<usize>),
    pub challenge_index: usize,
    pub response_index: usize,
}

#[derive(Debug, Clone)]
pub struct BatchProjField<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: BatchProjFieldProtocolInput<F>,
    pub atomic_pop: BatchProjFieldAtomicPoP<F>,
    pub litebullet: LiteBullet<F>,
}

impl<F: PrimeField> BatchProjFieldProtocolInput<F> {
    pub fn new(hat_inputs: Vec<F>, point_inputs: Vec<(Vec<F>, Vec<F>)>) -> Self {
        let num_inputs = hat_inputs.len();
            if num_inputs != point_inputs.len() {
                panic!("BatchProjField: Number of hat_inputs and point_inputs are not equal");
        }

            if num_inputs == 0 {
                panic!("BatchProjField: No inputs provided");
        }

        let log_m = point_inputs[0].0.len();
        let log_n = point_inputs[0].1.len();

        let input_shape = (1 << log_m, 1 << log_n);

        Self {
            hat_inputs: hat_inputs,
            point_inputs: point_inputs,
            input_mats: Vec::new(),
            input_shape: input_shape,
            num_inputs: num_inputs,
            ready: false,
        }
    }

    pub fn set_input(&mut self, input_mats: Vec<DenseMatFieldCM<F>>,
    ) {
            if self.num_inputs != input_mats.len() {
                panic!("BatchProjField: Number of input matrices does not match hat/point count");
        }

        let input_shape = input_mats[0].shape;

        let log_m = self.point_inputs[0].0.len();
        let log_n = self.point_inputs[0].1.len();

            if input_shape.0 != 1 << log_m || input_shape.1 != 1 << log_n {
                panic!("BatchProjField: Input matrix shape does not match point vector length (log_m/log_n)");
        }

        for i in 0..self.num_inputs {
                if input_shape != input_mats[i].shape {
                    panic!("BatchProjField: Found a matrix shape different from the first one");
            }

                if log_m != self.point_inputs[i].0.len() || log_n != self.point_inputs[i].1.len() {
                    panic!("BatchProjField: The i-th point input length is inconsistent with the first one");
            }
        }

        self.input_mats = input_mats;
        self.ready = true;
    }

    pub fn clear(&mut self) {
        self.input_mats = Vec::new();
        self.ready = false;
    }
}

impl<F: PrimeField> BatchProjFieldAtomicPoP<F> {
    pub fn new() -> Self {
        Self {
            hat_inputs: Vec::new(),
            point_inputs: Vec::new(),
            c_hat: F::zero(),
            c_point: (Vec::new(), Vec::new()),
            c_shape: (0, 0),
            challenge: F::zero(),
            response: F::zero(),
            mapping: BatchProjFieldMapping {
                hat_inputs_index: Vec::new(),
                point_inputs_index: Vec::new(),
                c_hat_index: 0,
                c_point_index: (Vec::new(), Vec::new()),
                challenge_index: 0,
                response_index: 0,
            },
            check: ArithmeticExpression::constant(F::zero()),
            link_inputs: Vec::new(),
            ready: (false, false, false),
        }
    }

    pub fn set_message(
        &mut self,
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    ) {
        self.hat_inputs = hat_inputs;
        self.point_inputs = point_inputs;
        self.mapping.hat_inputs_index = hat_inputs_index;
        self.mapping.point_inputs_index = point_inputs_index;
    }

    pub fn set_pop_trans(
        &mut self,
        c_hat: F,
        c_point: (Vec<F>, Vec<F>),
        c_shape: (usize, usize),
        challenge: F,
        response: F,
        c_hat_index: usize,
        c_point_index: (Vec<usize>, Vec<usize>),
        challenge_index: usize,
        response_index: usize,
    ) {
        self.c_hat = c_hat;
        self.c_point = c_point;
        self.c_shape = c_shape;
        self.challenge = challenge;
        self.response = response;
        self.mapping.c_hat_index = c_hat_index;
        self.mapping.c_point_index = c_point_index;
        self.mapping.challenge_index = challenge_index;
        self.mapping.response_index = response_index;
        self.ready.0 = true;
    }

    pub fn set_check(&mut self, check: ArithmeticExpression<F>) {
        self.check = check;
        self.ready.1 = true;
    }

    pub fn set_links(&mut self, link_inputs: Vec<ArithmeticExpression<F>>) {
        self.link_inputs = link_inputs;
        self.ready.2 = true;
    }

    pub fn is_ready(&self) -> bool {
        self.ready.0 && self.ready.1 && self.ready.2
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> BatchProjField<F> {
    pub fn new(
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    ) -> Self {
        let mut atomic_pop = BatchProjFieldAtomicPoP::new();

        atomic_pop.set_message(
            hat_inputs.clone(),
            point_inputs.clone(),
            hat_inputs_index,
            point_inputs_index,
        );


        Self {
            protocol_input: BatchProjFieldProtocolInput::new(hat_inputs, point_inputs),
            atomic_pop: atomic_pop,
            litebullet: LiteBullet::new(F::zero(), 0, 0),
        }
    }

    pub fn set_input(
        &mut self,
        input_mats: Vec<DenseMatFieldCM<F>>,
    ) {
        self.protocol_input.set_input(input_mats);
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for BatchProjField<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.litebullet.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
            // Reduction: combine multiple matrix projections into a single batch commitment
            if !self.protocol_input.ready {
                panic!("⚠️ BatchProjField: Protocol input not set");
            }

       
        let single_len = self.protocol_input.input_shape.0 * self.protocol_input.input_shape.1;
        
        let num_inputs = self.protocol_input.num_inputs;
        let num_input_padded = num_inputs.next_power_of_two();
        let len = num_input_padded * single_len;
        let _log_len = len.ilog2() as usize;
        let log_num = num_input_padded.ilog2() as usize; 

        let mut concat_vec = Vec::new();

        for mat in self.protocol_input.input_mats.iter() {
            concat_vec.extend(mat.to_vec());
        }

        let challenge_index = trans.pointer;
        let challenge = trans.gen_challenge();

        let mut b_vec = Vec::new();
        let mut xxxx_vec = Vec::new();

        let mut batch_hat = F::zero();
        let mut cur_mul = F::one();

        for i in 0..num_inputs {
            batch_hat = batch_hat + self.protocol_input.hat_inputs[i] * cur_mul;

            let point = self.protocol_input.point_inputs[i].clone();
            let xxxx = [point.1.as_slice(), point.0.as_slice()].concat();
            let xi_cur = xi::xi_from_challenges(&xxxx);
            let xi_cur_mul = linear::vec_scalar_mul(&xi_cur, &cur_mul);

            b_vec.extend(xi_cur_mul);
            xxxx_vec.push(xxxx);

            cur_mul = cur_mul * challenge;
        }

        let batch_hat_index = trans.pointer;
        trans.push_response(batch_hat);

  
        if num_inputs < num_input_padded {
            let padding = vec![F::zero(); (num_input_padded - num_inputs) * single_len];
            concat_vec.extend(padding);
            b_vec.extend(vec![F::zero(); (num_input_padded - num_inputs) * single_len]);
        }

        self.litebullet = LiteBullet::new(batch_hat, batch_hat_index, (num_input_padded * single_len) as usize);
        self.litebullet.set_input(concat_vec, b_vec);

        self.litebullet.reduce_prover(trans);

        let (c_hat, c_vec_point) = self.litebullet.atomic_pop.get_a();
        let (c_hat_index, c_vec_point_index) = self.litebullet.atomic_pop.get_a_index();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let single_challenge_point = b_vec_point[log_num..].to_vec();
        let multiplicant_point = b_vec_point[..log_num].to_vec();

        let multiplicants = xi::xi_from_challenges(&multiplicant_point)[..num_inputs].to_vec();
        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point =  self.protocol_input.point_inputs[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_ip = xi::xi_ip_from_challenges(&single_challenge_point, &xxxx);
            b_hat_expected = b_hat_expected + cur_ip * cur_mul * multiplicants[i];
            cur_mul = cur_mul * challenge;
        }

        let flag = b_hat == b_hat_expected;

        assert_eq!(flag, true, "BatchProjField: b_hat 与期望值不一致");

        self.atomic_pop.set_pop_trans(
            c_hat,
            c_vec_point,
            ((num_input_padded * single_len) as usize, 1usize),
            challenge,
            batch_hat,
            c_hat_index,
            c_vec_point_index,
            challenge_index,
            batch_hat_index,
        );

        flag
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let single_len = self.protocol_input.input_shape.0 * self.protocol_input.input_shape.1;
        
        let num_inputs = self.protocol_input.num_inputs;
        let num_input_padded = num_inputs.next_power_of_two();
        let len = num_input_padded * single_len;
        let _log_len = len.ilog2() as usize;
        let log_num = num_input_padded.ilog2() as usize; 

        let challenge_index = trans.pointer;
        let challenge = trans.get_at_position(challenge_index);
        trans.pointer += 1;

        let mut cur_mul = F::one();
        let mut batch_hat_expected = F::zero();
        for i in 0..num_inputs {
            batch_hat_expected = batch_hat_expected + self.protocol_input.hat_inputs[i] * cur_mul;
            cur_mul = cur_mul * challenge;
        }

        let batch_hat_index = trans.pointer;
        let batch_hat = trans.get_at_position(batch_hat_index);
        trans.pointer += 1;

    assert_eq!(batch_hat, batch_hat_expected, "BatchProjField: batch_hat does not match expected value");

        let flag1 = batch_hat == batch_hat_expected;

        self.litebullet = LiteBullet::new(batch_hat, batch_hat_index, num_input_padded * single_len);

        let flag2 = self.litebullet.verify_as_subprotocol(trans);

        let (c_hat, c_vec_point) = self.litebullet.atomic_pop.get_a();
        let (c_hat_index, c_vec_point_index) = self.litebullet.atomic_pop.get_a_index();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let single_challenge_point = b_vec_point[log_num..].to_vec();
        let multiplicant_point = b_vec_point[..log_num].to_vec();

        let multiplicants = xi::xi_from_challenges(&multiplicant_point)[..num_inputs].to_vec();
        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point = self.protocol_input.point_inputs[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_ip = xi::xi_ip_from_challenges(&single_challenge_point, &xxxx);
            b_hat_expected = b_hat_expected + cur_ip * cur_mul * multiplicants[i];
            cur_mul = cur_mul * challenge;
        }

        let flag3 = b_hat == b_hat_expected;

        self.atomic_pop.set_pop_trans(
            c_hat,
            c_vec_point,
            ((num_input_padded * single_len) as usize, 1usize),
            challenge,
            batch_hat,
            c_hat_index,
            c_vec_point_index,
            challenge_index,
            batch_hat_index,
        );

        flag1 && flag2 && flag3
    }

    fn prepare_atomic_pop(&mut self) -> bool {
        if !self.atomic_pop.ready.0 {
            panic!("⚠️ BatchProjField: Proof data not ready, cannot generate atomic PoP constraint");
        }

        let single_len = self.protocol_input.input_shape.0 * self.protocol_input.input_shape.1;
        let num_inputs = self.protocol_input.num_inputs;
        let num_input_padded = num_inputs.next_power_of_two();
        let len = num_input_padded * single_len;
        let _log_len = len.ilog2() as usize;
        let log_num = num_input_padded.ilog2() as usize; 

        let hat_inputs_index = self.atomic_pop.mapping.hat_inputs_index.clone();
        let point_inputs_index = self.atomic_pop.mapping.point_inputs_index.clone();
        let _c_hat_index = self.atomic_pop.mapping.c_hat_index; // unused currently
        let _c_point_index = self.atomic_pop.mapping.c_point_index.clone(); // unused currently
        let challenge_index = self.atomic_pop.mapping.challenge_index;
        let response_index = self.atomic_pop.mapping.response_index;
        let b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let b_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        // 使用公共输入引用（transcript 中对应位置作为公共输入）
        let hat_inputs_exprs = hat_inputs_index.iter()
            .map(|hat| {
                ArithmeticExpression::pub_input(*hat)
            })
            .collect::<Vec<_>>();

        let xxxx_exprs = point_inputs_index.iter()
            .map(|(xl, xr)| {
                let xr_exprs = xr.iter()
                    .map(|x| ArithmeticExpression::pub_input(*x))
                    .collect::<Vec<_>>();
                let xl_exprs = xl.iter()
                    .map(|x| ArithmeticExpression::pub_input(*x))
                    .collect::<Vec<_>>();
                [xr_exprs.as_slice(), xl_exprs.as_slice()].concat()
            })
            .collect::<Vec<_>>();

        let b_hat_expr = ArithmeticExpression::pub_input(b_hat_index);
        let b_point_expr = b_point_index.iter()
            .map(|x| ArithmeticExpression::pub_input(*x))
            .collect::<Vec<_>>();

        let challenge_expr = ArithmeticExpression::pub_input(challenge_index);
        let batch_hat_expr = ArithmeticExpression::pub_input(response_index);


        let mut cur_mul = ArithmeticExpression::constant(F::one());
        let mut check = batch_hat_expr;
        for i in 0..num_inputs {
            check = ArithmeticExpression::sub(
                check,
                ArithmeticExpression::mul(
                    hat_inputs_exprs[i].clone(),
                    cur_mul.clone(),
                ),
            );
            cur_mul = ArithmeticExpression::mul(cur_mul, challenge_expr.clone());
        }

        self.atomic_pop.set_check(check);

        let multiplicant_point_exprs = b_point_expr[..log_num].to_vec();
        // let mut multiplicant_exprs_padded = vec![ArithmeticExpression::constant(F::one())];

        // // Important: xi_from_challenges applies challenges from last to first.
        // // To match that order, iterate xxxx in reverse when expanding the tensor.
        // for xx_expr in multiplicant_point_exprs.iter().rev() {
        //     let vec_l_expr = multiplicant_exprs_padded.clone();
        //     let vec_r_expr: Vec<ArithmeticExpression<F>> = multiplicant_exprs_padded
        //         .iter()
        //         .map(|e| ArithmeticExpression::mul(e.clone(), xx_expr.clone()))
        //         .collect();
        //     multiplicant_exprs_padded = vec_l_expr.into_iter().chain(vec_r_expr).collect();
        // }

        let multiplicant_exprs_padded = arithmetic_expression::xi_from_challenges_exprs(&multiplicant_point_exprs);

        let multiplicants_exprs = multiplicant_exprs_padded[..num_inputs].to_vec();

        let mut cur_mul_expr = ArithmeticExpression::constant(F::one());
        let mut link = b_hat_expr;

        let single_challenge_point_expr = b_point_expr[log_num..].to_vec();

        for i in 0..num_inputs {
            let xxxx = xxxx_exprs[i].clone();

            let cur_ip_expr = arithmetic_expression::xi_ip_from_challenges_exprs(&single_challenge_point_expr, &xxxx);

            link = ArithmeticExpression::sub(
                link,
                ArithmeticExpression::mul(cur_ip_expr, ArithmeticExpression::mul(cur_mul_expr.clone(), multiplicants_exprs[i].clone())),
            );
            cur_mul_expr = ArithmeticExpression::mul(cur_mul_expr, challenge_expr.clone());
        }

        self.atomic_pop.set_links(vec![link]);

        let flag = self.litebullet.prepare_atomic_pop();

        self.atomic_pop.is_ready() && flag
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in BatchProjField");
            return false;
        }
        // 1. Add main check constraint (check = 0)
        cs_builder.add_constraint(self.atomic_pop.check.clone());

        // 2. Add link vector constraints (link_inputs = 0)
        for constraint in &self.atomic_pop.link_inputs {
            cs_builder.add_constraint(constraint.clone());
        }

        self.litebullet.synthesize_atomic_pop_constraints(cs_builder)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic_protocol::AtomicMatProtocol;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use mat::utils::matdef::concat_mat_to_square;

    fn make_mats_and_inputs(num_mats: usize, shape: (usize, usize)) -> (
    Vec<DenseMatFieldCM<BlsFr>>,            // Input matrix set
    Vec<(Vec<BlsFr>, Vec<BlsFr>)>,          // Each matrix's points (l, r)
    Vec<BlsFr>,                             // Each matrix's projection result hat
    ) {
        let (rows, cols) = shape;
        assert!(rows.is_power_of_two() && cols.is_power_of_two());

        let mut mats = Vec::new();
        let mut hats = Vec::new();
        let mut points = Vec::new();

    // Use fixed challenges (log2(rows/cols)=1 for 2x2 case)
        let point_l = vec![BlsFr::from(2u64)];
        let point_r = vec![BlsFr::from(3u64)];

        for i in 0..num_mats {
            let mut mat = DenseMatFieldCM::new(rows, cols);
            let mut data = Vec::new();
            for r in 0..rows {
                let mut row = Vec::new();
                for c in 0..cols {
                    // Simple deterministic value
                    row.push(BlsFr::from((i * rows * cols + r * cols + c + 1) as u64));
                }
                data.push(row);
            }
            mat.set_data(data);

            let hat = mat.proj_lr_challenges(&point_l, &point_r);

            mats.push(mat);
            points.push((point_l.clone(), point_r.clone()));
            hats.push(hat);
        }

    // Use utility function to concatenate into near-square, simulate production path (flatten -> concat -> reshape), to verify total length and shape.
    let big_square = concat_mat_to_square(&mats);
    let total_len = num_mats * rows * cols;
    let log_n_new = ((total_len).ilog2() / 2) as usize;
    let n_new = 1 << log_n_new;
    let m_new = total_len / n_new;
    assert_eq!(big_square.shape, (m_new, n_new), "Composed square matrix shape mismatch");

    (mats, points, hats)
    }

    #[test]
    fn test_batchprojfield_reduce_and_verify_roundtrip() {
    // Basic round: 4 matrices (power of 2)
    let num_mats = 4; // power of 2
        let shape = (2, 2);

        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);
    // 索引：这里使用自然顺序
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![ (vec![0], vec![0]); num_mats];
        let mut protocol = BatchProjField::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProjField reduce_prover failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProjField verify_as_subprotocol failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after verify");
    }

    #[test]
    fn test_batchprojfield_reduce_and_verify_with_padding() {
    // Use non-power-of-2 count to test internal padding logic
    let num_mats = 3; // will be padded to 4 internally
        let shape = (2, 2);

        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![ (vec![0], vec![0]); num_mats];
        let mut protocol = BatchProjField::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProjField reduce_prover (padding) failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProjField verify_as_subprotocol (padding) failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after verify (padding)");
    }

    #[test]
    fn test_batchprojfield_synthesize_constraints() {
        use crate::pop::arithmetic_expression::ConstraintSystemBuilder;
    // 1. Input
    let num_mats = 4; // power of 2 to avoid entering padding branch
        let shape = (2, 2);
        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);

    // 2. Build transcript: record hat/point indices before push
        let mut prover_trans = Transcript::new(BlsFr::zero());
        let mut hat_indices = Vec::new();
        for h in &hats { hat_indices.push(prover_trans.pointer); prover_trans.push_response(*h); }
        let mut point_indices: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for (l, r) in &points {
            let mut l_idx = Vec::new();
            for x in l { l_idx.push(prover_trans.pointer); prover_trans.push_response(*x); }
            let mut r_idx = Vec::new();
            for x in r { r_idx.push(prover_trans.pointer); prover_trans.push_response(*x); }
            point_indices.push((l_idx, r_idx));
        }

    // 3. Prover reduction
        let mut protocol = BatchProjField::<BlsFr>::new(hats.clone(), points.clone(), hat_indices.clone(), point_indices.clone());
        protocol.set_input(mats.clone());

        let mut cs_builder = ConstraintSystemBuilder::new();

        assert!(protocol.reduce_prover_with_constraint_building(&mut prover_trans, &mut cs_builder), "BatchProjField reduce_prover failed");

    // 5. Build private inputs: use the entire transcript sequence as witness (indices fully aligned, no dummy 0 to avoid misalignment)
    let private_inputs = prover_trans.get_trans_seq();

    // Only publicize transcript positions actually referenced by constraints: challenge_index, response_index, hat_b_index, point_b_index (first log_num + remaining single challenge),
    // and hat_inputs_index / point_inputs_index. Other positions are filled with 0 to keep indices unchanged.
    let mapping = &protocol.atomic_pop.mapping; // challenge / response
    let challenge_index = mapping.challenge_index;
    let response_index = mapping.response_index;
    let hat_b_index = protocol.litebullet.atomic_pop.mapping.hat_b_index;
    let point_b_index_tuple = protocol.litebullet.atomic_pop.mapping.point_b_index.clone();

    // Collect all required public indices
    let mut required_pub: Vec<usize> = Vec::new();
    required_pub.push(challenge_index);
    required_pub.push(response_index);
    required_pub.push(hat_b_index);
    required_pub.extend(point_b_index_tuple.0.iter().cloned());
    required_pub.extend(point_b_index_tuple.1.iter().cloned());
    required_pub.extend(hat_indices.iter().cloned());
    for (l_idx, r_idx) in &point_indices { required_pub.extend(l_idx.iter().cloned()); required_pub.extend(r_idx.iter().cloned()); }
    required_pub.sort_unstable();
    required_pub.dedup();

    let max_pub_index = *required_pub.iter().max().unwrap();
    let mut public_inputs = vec![BlsFr::zero(); max_pub_index + 1];
    for &idx in &required_pub { public_inputs[idx] = prover_trans.get_at_position(idx); }

    cs_builder.set_public_inputs(public_inputs);
    cs_builder.set_private_inputs(private_inputs);

    // 6. Validate constraints
    let validation = cs_builder.validate_constraints();
    assert!(validation.is_ok(), "Constraints not satisfied: {:?}", validation.err());
    }

    #[test]
    fn test_batchprojfield_reduce_and_verify_roundtrip_4x4() {
    // Larger shape 4x4 (log dimension=2)
    let num_mats = 4; // power of 2
        let shape = (4, 4);

        let (rows, cols) = shape;
        assert_eq!(rows, 4);
        assert_eq!(cols, 4);

    // Prepare deterministic l/r challenge vectors (length log2(4)=2)
        let point_l = vec![BlsFr::from(2u64), BlsFr::from(5u64)];
        let point_r = vec![BlsFr::from(3u64), BlsFr::from(7u64)];

        let mut mats = Vec::new();
        let mut hats = Vec::new();
        let mut points = Vec::new();

        for i in 0..num_mats {
            let mut mat = DenseMatFieldCM::new(rows, cols);
            let mut data = Vec::new();
            for r in 0..rows {
                let mut row_vec = Vec::new();
                for c in 0..cols {
                    row_vec.push(BlsFr::from((i * rows * cols + r * cols + c + 1) as u64));
                }
                data.push(row_vec);
            }
            mat.set_data(data);
            let hat = mat.proj_lr_challenges(&point_l, &point_r);
            mats.push(mat);
            points.push((point_l.clone(), point_r.clone()));
            hats.push(hat);
        }

    // Indices: natural order; for 4x4, each side has 2 challenge bits, use [0,1]
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![(vec![0,1], vec![0,1]); num_mats];

        let mut protocol = BatchProjField::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProjField 4x4 reduce_prover failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProjField 4x4 verify_as_subprotocol failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after 4x4 verify");
    }
}