//! Batch Multiple projection verification into a large vector projection
//! 
//! Support for matrices with heterogeneous shapes
//! 
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::xi;
use mat::utils::matdef::DenseMatCM;


use crate::atomic_protocol::AtomicMatProtocol; 
use crate::pop::{
    arithmetic_expression,
    arithmetic_expression::{
        ArithmeticExpression,
        ConstraintSystemBuilder
    }
};

use crate::protocols::litebullet::LiteBullet;
use crate::utils::{MatContainerMyInt, PointsContainer, compute_xi_at_position};

use mat::MyInt;

// removed unused rayon parallel import and NSHARD constant

#[derive(Debug, Clone)]
pub struct BatchProjProtocolInput<F: PrimeField> {
    pub mat_container: MatContainerMyInt<F>,
    pub point_container: PointsContainer<F>,
    pub num_inputs: usize,
    pub ready: bool,
}

#[derive(Debug, Clone)]
pub struct BatchProjMapping {
    pub c_hat_index: usize,
    pub c_point_index: (Vec<usize>, Vec<usize>),
    pub final_c_hat_index: usize,
    pub final_c_point_index: (Vec<usize>, Vec<usize>),
    pub challenge_index: usize,
    pub response_index: usize,
}

#[derive(Debug, Clone)]
pub struct BatchProjAtomicPoP<F: PrimeField> {
    pub c_hat: F,
    pub c_point: (Vec<F>, Vec<F>),
    pub c_shape: (usize, usize),
    pub challenge: F,
    pub response: F,
    pub mapping: BatchProjMapping,
    // check
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    pub ready: (bool, bool, bool),
}

impl BatchProjMapping {
    pub fn new() -> Self {
        Self {
            c_hat_index: 0,
            c_point_index: (Vec::new(), Vec::new()),
            final_c_hat_index: 0,
            final_c_point_index: (Vec::new(), Vec::new()),
            challenge_index: 0,
            response_index: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchProj<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: BatchProjProtocolInput<F>,
    pub atomic_pop: BatchProjAtomicPoP<F>,
    pub litebullet: LiteBullet<F>,
}

impl<F: PrimeField> BatchProjProtocolInput<F> {
    pub fn new() -> Self {
        
        Self {
            mat_container: MatContainerMyInt::new(),
            point_container: PointsContainer::new(),
            num_inputs: 0,
            ready: false,
        }
    }

    pub fn set_input_from_matcontainer(&mut self, mat_container: MatContainerMyInt<F>) {
        self.num_inputs = mat_container.sorted_areas.len();
        self.mat_container = mat_container;

        self.ready = true;
    }

    pub fn set_input(&mut self, input_mats: Vec<DenseMatCM<MyInt, F>>,
    ) {
        self.num_inputs = input_mats.len();

        for i in 0..self.num_inputs {
            self.mat_container.push(input_mats[i].clone());
        }

        self.ready = true;
    }

    pub fn push_input(&mut self, mat: DenseMatCM<MyInt, F>) {
        self.num_inputs += 1;
        self.mat_container.push(mat);
        self.ready = true;
    }

    // (消息设置逻辑移动到 BatchProj 实现中)
  

    pub fn clear(&mut self) {
        self.mat_container = MatContainerMyInt::new();
        self.ready = false;
    }


}

impl<F: PrimeField> BatchProjAtomicPoP<F> {
    pub fn new() -> Self {
        Self {
            c_hat: F::zero(),
            c_point: (Vec::new(), Vec::new()),
            c_shape: (0, 0),
            challenge: F::zero(),
            response: F::zero(),
            mapping: BatchProjMapping::new(),
            check: ArithmeticExpression::constant(F::zero()),
            link_inputs: Vec::new(),
            ready: (false, false, false),
        }
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
        final_c_hat_index: usize,
        final_c_point_index: (Vec<usize>, Vec<usize>),
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
        self.mapping.final_c_hat_index = final_c_hat_index;
        self.mapping.final_c_point_index = final_c_point_index;
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


    pub fn get_c(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.c_hat.clone(), (self.c_point.0.clone(), self.c_point.1.clone()))
    }

    pub fn get_c_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.c_hat_index, (self.mapping.c_point_index.0.clone(), self.mapping.c_point_index.1.clone()))
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> BatchProj<F> {
    pub fn new(
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    ) -> Self {
        let mut protocol = Self {
            protocol_input: BatchProjProtocolInput::new(),
            atomic_pop: BatchProjAtomicPoP::new(),
            litebullet: LiteBullet::new(F::zero(), 0, 0),
        };
        protocol.set_message(hat_inputs, point_inputs, hat_inputs_index, point_inputs_index);
        protocol
    }

    pub fn get_pub_input_len(&self) -> usize {
        self.atomic_pop.mapping.final_c_point_index.0.len() + self.atomic_pop.mapping.final_c_point_index.1.len() + 1
    }

    pub fn new_from_point_container(
        point_container: PointsContainer<F>,
    ) -> Self {
        let mut protocol = Self {
            protocol_input: BatchProjProtocolInput::new(),
            atomic_pop: BatchProjAtomicPoP::new(),
            litebullet: LiteBullet::new(F::zero(), 0, 0),
        };
        protocol.protocol_input.num_inputs = point_container.sorted_points.len();
        protocol.protocol_input.point_container = point_container;

        protocol.protocol_input.ready = true;

        protocol
    }

    pub fn set_input(
        &mut self,
        input_mats: Vec<DenseMatCM<MyInt, F>>,
    ) {
        self.protocol_input.set_input(input_mats);
    }

    pub fn set_input_from_matcontainer(
        &mut self,
        mat_container: MatContainerMyInt<F>,
    ) {
        self.protocol_input.set_input_from_matcontainer(mat_container);
    }


    // 将原先放在 ProtocolInput 的 set_message 迁移到此处
    pub fn set_message(
        &mut self,
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    ) {
        self.protocol_input.num_inputs = hat_inputs.len();
        for i in 0..self.protocol_input.num_inputs {
            self.protocol_input.point_container.push(
                hat_inputs[i].clone(),
                point_inputs[i].clone(),
                hat_inputs_index[i],
                point_inputs_index[i].clone(),
            );
        }
        self.protocol_input.ready = true;
    }

    pub fn push_message(
        &mut self,
        hat_input: F,
        point_input: (Vec<F>, Vec<F>),
        hat_input_index: usize,
        point_input_index: (Vec<usize>, Vec<usize>),
    ) {
        self.protocol_input.num_inputs += 1;
        self.protocol_input.point_container.push(hat_input, point_input, hat_input_index, point_input_index);
        self.protocol_input.ready = true;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for BatchProj<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.litebullet.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        // Implement the reduction logic for the prover
        if !self.protocol_input.ready {
            panic!("⚠️  Protocol input not ready for BatchProj reduction");
        }

        let num_inputs = self.protocol_input.num_inputs;

        let challenge_index = trans.pointer;
        let challenge = trans.gen_challenge();

        let hat_batched = self.protocol_input.point_container.hat_batched(challenge);
        let response_index = trans.pointer;
        trans.push_response(hat_batched);

        // Build xi vector and flattened matrices WITHOUT consuming matrix data so that
        // later external checks (tests) can re-flatten and obtain the same padded length.
        // Previously using into_flattened_vec() cleared internal matrix data causing
        // post-protocol flatten length shrink (e.g. 8 vs expected 32). Keeping data here
        // preserves alignment with xi_concat output.
        let b_vec = self.protocol_input.point_container.xi_concat(challenge);
        let concat_myint_vec = self.protocol_input.mat_container.into_flattened_vec();

        let len = b_vec.len();

        assert_eq!(concat_myint_vec.len(), len);

        self.litebullet = LiteBullet::new(hat_batched, response_index, len as usize);
        
        self.litebullet.reduce_prover_with_mixed_input(trans, concat_myint_vec, b_vec);

        let (c_hat, (c_vec_point_l, c_vec_point_r)) = self.litebullet.atomic_pop.get_a();
        let (c_hat_index, c_vec_point_index) = self.litebullet.atomic_pop.get_a_index();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

  
        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point =  self.protocol_input.point_container.sorted_points[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_challenge_point = &b_vec_point[b_vec_point.len() - xxxx.len()..].to_vec();
            let cur_ip = xi::xi_ip_from_challenges(&cur_challenge_point, &xxxx);
            
            let cur_position = self.protocol_input.point_container.sorted_start_position[i];
            let cur_multiplicant = compute_xi_at_position(cur_position, xxxx.len(), &b_vec_point);
            
            b_hat_expected = b_hat_expected + cur_ip * cur_mul * cur_multiplicant;
            
            cur_mul = cur_mul * challenge;
        }

        let flag = b_hat == b_hat_expected;

        // Expose c_hat and its point (c_vec_point) at the end of transcript to be used as public (if desired)
        let final_c_hat_index = trans.pointer;
        let final_c_hat = c_hat;
        trans.push_response(final_c_hat);
        
        let mut final_c_vec_point_index_l = Vec::new();
        for val in &c_vec_point_l { // left coords
            final_c_vec_point_index_l.push(trans.pointer);
            trans.push_response(*val);
        }
        let mut final_c_vec_point_index_r = Vec::new();
        for val in &c_vec_point_r { // right coords
            final_c_vec_point_index_r.push(trans.pointer);
            trans.push_response(*val);
        }

        assert_eq!(flag, true, "BatchProj b_hat mismatch");

        self.atomic_pop.set_pop_trans(
            final_c_hat,
            (c_vec_point_l.clone(), c_vec_point_r.clone()),
            (len, 1usize),
            challenge,
            hat_batched,
            c_hat_index,
            c_vec_point_index,
            final_c_hat_index,
            (final_c_vec_point_index_l.clone(), final_c_vec_point_index_r.clone()),
            challenge_index,
            response_index,
        );

        flag

    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {

        
        let num_inputs = self.protocol_input.num_inputs;
    
        let challenge_index = trans.pointer;
        let challenge = trans.get_at_position(challenge_index);
        trans.pointer += 1;

        let mut cur_mul = F::one();
        let mut batch_hat_expected = F::zero();
        for i in 0..num_inputs {
            let hat_value = self.protocol_input.point_container.sorted_hats[i];
            batch_hat_expected = batch_hat_expected + hat_value * cur_mul;
            cur_mul = cur_mul * challenge;
        }

        let batch_hat_index = trans.pointer;
        let batch_hat = trans.get_at_position(batch_hat_index);
        trans.pointer += 1;

        assert_eq!(batch_hat, batch_hat_expected, "BatchProj batch_hat mismatch");

        let flag1 = batch_hat == batch_hat_expected;

        let len = self.protocol_input.point_container.flatten_len();

        self.litebullet = LiteBullet::new(batch_hat, batch_hat_index, len);

        let flag2 = self.litebullet.verify_as_subprotocol(trans);

        let (c_hat, (c_vec_point_l, c_vec_point_r)) = self.litebullet.atomic_pop.get_a();
        let (c_hat_index, c_vec_point_index) = self.litebullet.atomic_pop.get_a_index();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let len = self.protocol_input.point_container.flatten_len();
        let _log_len = len.ilog2() as usize; // unused (kept for potential future assertions)

        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point =  self.protocol_input.point_container.sorted_points[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_challenge_point = b_vec_point[b_vec_point.len() - xxxx.len()..].to_vec();
            let cur_ip = xi::xi_ip_from_challenges(&cur_challenge_point, &xxxx);
            
            let cur_position = self.protocol_input.point_container.sorted_start_position[i];
            let cur_multiplicant = compute_xi_at_position(cur_position, xxxx.len(), &b_vec_point);
            
            b_hat_expected = b_hat_expected + cur_ip * cur_mul * cur_multiplicant;
            
            cur_mul = cur_mul * challenge;
        }

        let flag3 = b_hat == b_hat_expected;

        let final_c_hat_index = trans.pointer;
        let final_c_hat = trans.get_at_position(final_c_hat_index);
        trans.pointer += 1;

        let flag4 = final_c_hat == c_hat;

        let mut flag5 = true;
        let mut final_c_vec_point_index_l = Vec::new();
        for i in 0..c_vec_point_l.len() { // left coords
            final_c_vec_point_index_l.push(trans.pointer);
            let cur_x = trans.get_at_position(trans.pointer);
            trans.pointer += 1;
            if cur_x != c_vec_point_l[i] { flag5 = false; }
        }
        let mut final_c_vec_point_index_r = Vec::new();
        for i in 0..c_vec_point_r.len() { // right coords
            final_c_vec_point_index_r.push(trans.pointer);
            let cur_x = trans.get_at_position(trans.pointer);
            trans.pointer += 1;
            if cur_x != c_vec_point_r[i] { flag5 = false; }
        }

        self.atomic_pop.set_pop_trans(
            c_hat,
            (c_vec_point_l, c_vec_point_r),
            (len, 1usize),
            challenge,
            batch_hat,
            c_hat_index,
            c_vec_point_index,
            final_c_hat_index,
            (final_c_vec_point_index_l.clone(), final_c_vec_point_index_r.clone()),
            challenge_index,
            batch_hat_index,
        );

        flag1 && flag2 && flag3 && flag4 && flag5
    }

    fn prepare_atomic_pop(&mut self) -> bool {

        if !self.atomic_pop.ready.0 {
            panic!("⚠️  Proof data not ready for BatchProj pop preparation!!");
        }

        let num_inputs = self.protocol_input.num_inputs;
    
        let challenge_index = self.atomic_pop.mapping.challenge_index;

        let _cur_mul = ArithmeticExpression::constant(F::one());
        let _batch_hat_expected = ArithmeticExpression::constant(F::zero());

        let hat_inputs_index = self.protocol_input.point_container.sorted_hats_index.clone();
        let point_inputs_index = self.protocol_input.point_container.sorted_points_index.clone();

        let _challenge_expr = ArithmeticExpression::<F>::input(challenge_index);
        let response_index = self.atomic_pop.mapping.response_index;
        let b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let b_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let hat_inputs_exprs = hat_inputs_index.iter()
            .map(|hat| {
                ArithmeticExpression::input(*hat)
            })
            .collect::<Vec<_>>();

        let point_inputs_exprs = point_inputs_index.iter()
            .map(|(xl, xr)| {
                let xr_exprs = xr.iter()
                    .map(|x| ArithmeticExpression::input(*x))
                    .collect::<Vec<_>>();
                let xl_exprs = xl.iter()
                    .map(|x| ArithmeticExpression::input(*x))
                    .collect::<Vec<_>>();
                (xl_exprs, xr_exprs)
            })
            .collect::<Vec<_>>();

        let b_hat_expr = ArithmeticExpression::input(b_hat_index);
        let b_point_expr = b_point_index.iter()
            .map(|x| ArithmeticExpression::input(*x))
            .collect::<Vec<_>>();

        let challenge_expr = ArithmeticExpression::input(challenge_index);
        let batch_hat_expr = ArithmeticExpression::input(response_index);


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

        
        let _len = self.protocol_input.point_container.flatten_len();
        let _log_num = num_inputs.ilog2() as usize;


        let mut cur_mul = ArithmeticExpression::constant(F::one());
        let mut link1 = b_hat_expr;

        for i in 0..num_inputs {
            
            let cur_point =  point_inputs_exprs[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();

            let cur_challenge_point = b_point_expr[b_point_expr.len() - xxxx.len()..].to_vec();
            let cur_ip = arithmetic_expression::xi_ip_from_challenges_exprs(&cur_challenge_point, &xxxx);
            
            let cur_position = self.protocol_input.point_container.sorted_start_position[i];
            
            let cur_multiplicant = arithmetic_expression::compute_xi_at_position_expr(cur_position, xxxx.len(), &b_point_expr);
            
            link1 = ArithmeticExpression::sub(
                link1,
                ArithmeticExpression::mul(
                    cur_ip,
                    ArithmeticExpression::mul(cur_mul.clone(), cur_multiplicant)
                )
            );

            cur_mul = ArithmeticExpression::mul(cur_mul, challenge_expr.clone());
        }

        let mut links = vec![link1];

        let c_hat_index = self.atomic_pop.mapping.c_hat_index;
        let c_vec_point_index = self.atomic_pop.mapping.c_point_index.clone();
        let final_c_hat_index = self.atomic_pop.mapping.final_c_hat_index;
        let final_c_vec_point_index = self.atomic_pop.mapping.final_c_point_index.clone();

        // 方案2：不直接使用 PubInput 全局索引，统一用 PriInput，公开由全局 link 在 NN 顶层建立
        links.push(
            ArithmeticExpression::sub(
                ArithmeticExpression::pri_input(c_hat_index),
                ArithmeticExpression::pri_input(final_c_hat_index),
            )
        );
        
        for i in 0..c_vec_point_index.0.len() {
            links.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::pri_input(c_vec_point_index.0[i]),
                    ArithmeticExpression::pri_input(final_c_vec_point_index.0[i]),
                )
            );
        }
        for i in 0..c_vec_point_index.1.len() {
            links.push(
                ArithmeticExpression::sub(
                    ArithmeticExpression::pri_input(c_vec_point_index.1[i]),
                    ArithmeticExpression::pri_input(final_c_vec_point_index.1[i]),
                )
            );
        }


        self.atomic_pop.set_links(links);

        let flag = self.litebullet.prepare_atomic_pop();

        self.atomic_pop.is_ready() && flag
        
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in BatchProj (missing set_pop_trans / set_check / set_links)");
            return false;
        }
        
        // 1. Add the main 'check' constraint
        cs_builder.add_constraint(self.atomic_pop.check.clone());

        // 2. Add 'link_inputs' constraints
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
    use mat::utils::matdef::MatOps;

    fn make_mats_and_inputs(num_mats: usize, shape: (usize, usize)) -> (
        Vec<DenseMatCM<MyInt, BlsFr>>,            // input matrices
        Vec<(Vec<BlsFr>, Vec<BlsFr>)>,          // per-matrix points (l, r)
        Vec<BlsFr>,                             // per-matrix projected hat values
    ) {
        let (rows, cols) = shape;
        assert!(rows.is_power_of_two() && cols.is_power_of_two());

        let mut mats = Vec::new();
        let mut hats = Vec::new();
        let mut points = Vec::new();

        // Fixed challenges for simplicity (log2 for rows/cols is 1 when shape=(2,2))
        let point_l = vec![BlsFr::from(2u64)];
        let point_r = vec![BlsFr::from(3u64)];

        for i in 0..num_mats {
            let mut mat = DenseMatCM::<MyInt, BlsFr>::new(rows, cols);
            let mut data = Vec::new();
            for c in 0..cols {
                let mut col = Vec::new();
                for r in 0..rows {
                    // simple deterministic values
                    col.push((i * rows * cols + r * cols + c + 1) as MyInt);
                }
                data.push(col);
            }
            mat.set_data(data);

            let hat = mat.proj_lr(&point_l, &point_r);

            mats.push(mat);
            points.push((point_l.clone(), point_r.clone()));
            hats.push(hat);
        }


    (mats, points, hats)
    }

    #[test]
    fn test_batchproj_reduce_and_verify_roundtrip() {
        // Setup
        let num_mats = 4; // power of two
        let shape = (2, 2);

        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);
    // Indices: here we just use identity ordering
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![ (vec![0], vec![0]); num_mats];
        let mut protocol = BatchProj::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProj reduce_prover failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProj verify_as_subprotocol failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after verify");
    }

    #[test]
    fn test_batchproj_reduce_and_verify_with_padding() {
        // Use a non-power-of-two number of inputs to exercise padding path
        let num_mats = 3; // will be padded to 4 internally
        let shape = (2, 2);

        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![ (vec![0], vec![0]); num_mats];
        let mut protocol = BatchProj::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProj reduce_prover (padding) failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProj verify_as_subprotocol (padding) failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after verify (padding)");
    }

    #[test]
    fn test_batchproj_synthesize_constraints() {
        use crate::pop::arithmetic_expression::ConstraintSystemBuilder;
        // 1. Inputs
        let num_mats = 4; // power of two to avoid padding branch ambiguity
        let shape = (2, 2);
        let (mats, points, hats) = make_mats_and_inputs(num_mats, shape);

        // 2. Build transcript with hat_inputs & point_inputs, recording indices BEFORE each push
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
        let mut protocol = BatchProj::<BlsFr>::new(hats.clone(), points.clone(), hat_indices.clone(), point_indices.clone());
        protocol.set_input(mats.clone());

        let mut cs_builder = ConstraintSystemBuilder::new();

        assert!(protocol.reduce_prover_with_constraint_building(&mut prover_trans, &mut cs_builder), "BatchProj reduce_prover failed");

        // Build private inputs: prepend a dummy zero so that transcript indices align
        let mut private_inputs = vec![BlsFr::zero()];
        private_inputs.extend(prover_trans.get_fs_proof_vec());

        // PUBLIC / PRIVATE 区分:
        // prepare_atomic_pop 中使用 ArithmeticExpression::pub_input(final_c_hat_index / final_c_vec_point_index[i])
        // 这些索引是 transcript 中的原始位置，因此这里构造一个与最大索引同长度的 public 向量，
        // 只在需要公开的位置填入真实值，其余填 0。（这样无需重新映射 expression 中的索引。）
        let mapping = &protocol.atomic_pop.mapping;
        let final_c_hat_index = mapping.final_c_hat_index;
        // tuple of (left,right) indices for point
         let final_c_vec_point_index = mapping.final_c_point_index.clone();

        // 计算需要的 public 向量长度 (使用所有 point indices)
        let mut max_pub_index = final_c_hat_index;
        for &idx in final_c_vec_point_index.0.iter().chain(final_c_vec_point_index.1.iter()) { max_pub_index = max_pub_index.max(idx); }
        let pub_len = max_pub_index + 1; // 索引从 0 开始
        let mut public_inputs = vec![BlsFr::zero(); pub_len];

        // 从 transcript 中取出公开值
        let final_c_hat_val = prover_trans.get_at_position(final_c_hat_index);
        public_inputs[final_c_hat_index] = final_c_hat_val;
        for &idx in final_c_vec_point_index.0.iter().chain(final_c_vec_point_index.1.iter()) {
            public_inputs[idx] = prover_trans.get_at_position(idx);
        }

        // 设置到 builder
        cs_builder.set_public_inputs(public_inputs);
        cs_builder.set_private_inputs(private_inputs);

        // 6. Validate constraints
        let validation = cs_builder.validate_constraints();
        assert!(validation.is_ok(), "Constraints not satisfied: {:?}", validation.err());
    }

    #[test]
    fn test_batchproj_reduce_and_verify_roundtrip_4x4() {
        // Larger matrix shape 4x4 (log dims = 2)
        let num_mats = 4; // power of two
        let shape = (4, 4);

        let (rows, cols) = shape;
        assert_eq!(rows, 4);
        assert_eq!(cols, 4);

        // Prepare deterministic l / r challenge vectors of length log2(4)=2
        let point_l = vec![BlsFr::from(2u64), BlsFr::from(5u64)];
        let point_r = vec![BlsFr::from(3u64), BlsFr::from(7u64)];

        let mut mats = Vec::new();
        let mut hats = Vec::new();
        let mut points = Vec::new();

        for i in 0..num_mats {
            let mut mat = DenseMatCM::<MyInt,BlsFr>::new(rows, cols);
            let mut data = Vec::new();
            for c in 0..cols {
                let mut col_vec = Vec::new();
                for r in 0..rows {
                    col_vec.push((i * rows * cols + r * cols + c + 1) as MyInt);
                }
                data.push(col_vec);
            }
            mat.set_data(data);
            let hat = mat.proj_lr(&point_l, &point_r);
            mats.push(mat);
            points.push((point_l.clone(), point_r.clone()));
            hats.push(hat);
        }

        // Indices: identity; for 4x4 we conceptually have two challenge bits per side so store [0,1]
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let point_indices = vec![(vec![0,1], vec![0,1]); num_mats];

        let mut protocol = BatchProj::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProj 4x4 reduce_prover failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProj 4x4 verify_as_subprotocol failed");

        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after 4x4 verify");
    }

    #[test]
    fn test_matcontainer_bilinear_projection_consistency() {
        // Test that the bilinear projection of flattened matrices equals c_hat
        // This test verifies: <flatten(matrices), c_point> = c_hat
        
        use mat::utils::linear::inner_product;
        use crate::utils::{MatContainerMyInt, PointsContainer};
        
        println!("🧪 Testing MatContainer bilinear projection consistency with different matrix sizes");
        
        // Create matrices of different sizes: 2x2, 4x2, 2x4
        let mut mats = Vec::new();
        let mut hats = Vec::new();
        let mut points = Vec::new();
        let mut areas = Vec::new();
        
    // Matrix 1: 2x2 (area = 4)
        let rows1 = 2; let cols1 = 2;
        let mut mat1 = DenseMatCM::<MyInt,BlsFr>::new(rows1, cols1);
        let data1 = vec![
            vec![1, 3],  // column 0
            vec![2, 4],  // column 1
        ];
        mat1.set_data(data1);
    // Use log2-dimension challenge vectors: log2(2)=1 each side
    let point1_l = vec![BlsFr::from(2u64)];
    let point1_r = vec![BlsFr::from(5u64)];
        let hat1 = mat1.proj_lr(&point1_l, &point1_r);
        
        mats.push(mat1);
        hats.push(hat1);
        points.push((point1_l, point1_r));
        areas.push(4);
        
    // Matrix 2: 4x2 (area = 8)  
        let rows2 = 4; let cols2 = 2;
        let mut mat2 = DenseMatCM::<MyInt,BlsFr>::new(rows2, cols2);
        let data2 = vec![
            vec![5, 9, 13, 17],   // column 0
            vec![6, 10, 14, 18],  // column 1
        ];
        mat2.set_data(data2);
    // log2(4)=2 for rows, log2(2)=1 for cols
    let point2_l = vec![BlsFr::from(11u64), BlsFr::from(13u64)];
    let point2_r = vec![BlsFr::from(23u64)];
        let hat2 = mat2.proj_lr(&point2_l, &point2_r);
        
        mats.push(mat2);
        hats.push(hat2);
        points.push((point2_l, point2_r));
        areas.push(8);
        
    // Matrix 3: 2x4 (area = 8)
        let rows3 = 2; let cols3 = 4;  
        let mut mat3 = DenseMatCM::<MyInt,BlsFr>::new(rows3, cols3);
        let data3 = vec![
            vec![21, 25],  // column 0
            vec![22, 26],  // column 1
            vec![23, 27],  // column 2
            vec![24, 28],  // column 3
        ];
        mat3.set_data(data3);
    // log2(2)=1 for rows, log2(4)=2 for cols
    let point3_l = vec![BlsFr::from(31u64)];
    let point3_r = vec![BlsFr::from(41u64), BlsFr::from(43u64)];
        let hat3 = mat3.proj_lr(&point3_l, &point3_r);
        
        mats.push(mat3);
        hats.push(hat3);
        points.push((point3_l, point3_r));
        areas.push(8);
        
        println!("📋 Created matrices with areas: {:?}", areas);
        
        // Create MatContainer and add matrices
        let mut mat_container = MatContainerMyInt::<BlsFr>::new();
        for mat in &mats {
            mat_container.push(mat.clone());
        }
        
        // Create PointsContainer and add points with proper indices
        let mut point_container = PointsContainer::<BlsFr>::new();
        for i in 0..hats.len() {
            // For this test, we use simple sequential indices
            let hat_index = i;
            let point_index = (
                (0..points[i].0.len()).collect::<Vec<_>>(),  // left indices
                (0..points[i].1.len()).collect::<Vec<_>>(),  // right indices
            );
            
            point_container.push(
                hats[i],
                points[i].clone(),
                hat_index,
                point_index,
            );
        }
        
        println!("🔍 Sorted areas in containers: {:?}", mat_container.sorted_areas);
        println!("🔍 Point container areas: {:?}", point_container.sorted_areas);
        
        // Generate a challenge for batching
        let challenge = BlsFr::from(42u64);
        
        // Get flattened matrices vector
        let flattened_mats = mat_container.into_flattened_vec();
        println!("📏 Flattened matrices length: {}", flattened_mats.len());
        
        // Get xi concatenated vector (challenge evaluation points)
        let xi_vec = point_container.xi_concat(challenge);
        println!("📏 Xi vector length: {}", xi_vec.len());
        
        // Verify lengths match
        assert_eq!(flattened_mats.len(), xi_vec.len(), 
            "Flattened matrices and xi vector must have same length");
        
        // Convert MyInt to field elements for inner product
        let flattened_field: Vec<BlsFr> = flattened_mats.iter()
            .map(|&x| BlsFr::from(x as u64))
            .collect();
        
        // Compute the bilinear projection: <flattened_matrices, xi_vector>
        let computed_projection = inner_product(&flattened_field, &xi_vec);
        
        // Compute expected c_hat using linear combination of individual hats
        let hat_batched = point_container.hat_batched(challenge);
        
        println!("🎯 Computed bilinear projection: {:?}", computed_projection);
        println!("🎯 Expected batched hat (c_hat): {:?}", hat_batched);
        
        // The key assertion: bilinear projection should equal c_hat
        assert_eq!(computed_projection, hat_batched,
            "Bilinear projection <flatten(matrices), xi_vector> should equal c_hat");
        
        println!("✅ MatContainer bilinear projection consistency test passed!");
        println!("✅ Verified: <flatten(matrices), c_point> = c_hat for mixed matrix sizes");
    }

    #[test]
    fn test_batchproj_bilinear_projection_via_protocol() {
        // Use heterogeneous matrix shapes and run full BatchProj reduction, then
        // verify <flatten(mats), xi_concat(challenge)> == batched hat stored in atomic_pop.response.
        use mat::utils::linear::inner_product;
        use ark_bls12_381::Fr as BlsFr;

    // Heterogeneous shapes (all power-of-two dimensions)
    let shapes = vec![(2,2), (4,2), (2,4)];

        let mut mats: Vec<DenseMatCM<MyInt,BlsFr>> = Vec::new();
        let mut hats: Vec<BlsFr> = Vec::new();
        let mut points: Vec<(Vec<BlsFr>, Vec<BlsFr>)> = Vec::new();

        // Deterministic filler counter
        let mut base: i32 = 1;
         for (rows, cols) in &shapes {
            let mut mat = DenseMatCM::<MyInt,BlsFr>::new(*rows, *cols);
            // Column-major data construction
            let mut data: Vec<Vec<MyInt>> = Vec::new();
            for _c in 0..*cols { // each column
                let mut col_vec = Vec::new();
                for _r in 0..*rows {
                    col_vec.push(base as MyInt);
                    base += 1;
                }
                data.push(col_vec);
            }
            mat.set_data(data);

            // Use LOG dimension length vectors (proj_lr expects log2 sized challenge vectors)
            let log_r = (*rows as u64).ilog2() as usize;
            let log_c = (*cols as u64).ilog2() as usize;
            let point_l: Vec<BlsFr> = (0..log_r).map(|i| BlsFr::from((i as u64)+2)).collect();
            let point_r: Vec<BlsFr> = (0..log_c).map(|i| BlsFr::from((i as u64)+11)).collect();
            let hat = mat.proj_lr(&point_l, &point_r);
            mats.push(mat);
            points.push((point_l, point_r));
            hats.push(hat);
        }

        // Indices: sequential hat indices; point indices enumerate coordinate positions per side
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let mut point_indices: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for (l,r) in &points {
            // Indices enumerate challenge positions (log dims)
            point_indices.push(((0..l.len()).collect(), (0..r.len()).collect()));
        }

        let mut protocol = BatchProj::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mats.clone());

        // Run prover reduction
        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "BatchProj reduce_prover failed (heterogeneous)");

        // Reconstruct bilinear projection externally
        let challenge = protocol.atomic_pop.challenge; // batching challenge used
        crate::utils::check_length_consistency(
            &protocol.protocol_input.mat_container,
            &protocol.protocol_input.point_container,
            challenge,
        );
        let flattened_myint = protocol.protocol_input.mat_container.flatten_and_concat();
        let xi_vec = protocol.protocol_input.point_container.xi_concat(challenge);
        let flattened_field: Vec<BlsFr> = flattened_myint.iter().map(|&x| BlsFr::from(x as u64)).collect();
        let bilinear_projection = inner_product(&flattened_field, &xi_vec);

        let batched_hat = protocol.atomic_pop.response; // stored in atomic_pop as 'response'

        println!("🎯 Bilinear projection (external) = {:?}", bilinear_projection);
        println!("🎯 Batched hat from protocol (response) = {:?}", batched_hat);
        println!("ℹ️  c_hat stored in atomic_pop = {:?} (expected to differ: it's LiteBullet a_reduce)", protocol.atomic_pop.c_hat);

        assert_eq!(bilinear_projection, batched_hat, "<flatten(mats), xi_vec> must equal batched hat (protocol response)");

        // Now run verifier to ensure transcript consistency
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "BatchProj verify_as_subprotocol failed (heterogeneous)");
        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch after heterogeneous verify");
    }
}





