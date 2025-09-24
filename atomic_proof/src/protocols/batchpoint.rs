//! Batch Multiple projection verification of the same matrix onto multiple points into a single matrix projection
//! 
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

use crate::protocols::litebullet::LiteBullet;

#[derive(Debug, Clone)]
pub struct BatchPointProtocolInput<F: PrimeField> {
    pub hat_inputs: Vec<F>,
    pub point_inputs: Vec<(Vec<F>, Vec<F>)>,
    pub input_mat: DenseMatFieldCM<F>,
    pub shape: (usize, usize),
    pub num_inputs: usize,
    pub ready: bool,
}

#[derive(Debug, Clone)]
pub struct BatchPointAtomicPoP<F: PrimeField> {
    pub hat_inputs: Vec<F>,
    pub point_inputs: Vec<(Vec<F>, Vec<F>)>,
    pub c_hat: F,
    pub c_point: (Vec<F>, Vec<F>),
    pub challenge: F,
    pub response: F,
    pub mapping: BatchPointMapping,
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    pub ready: (bool, bool, bool),
}

#[derive(Debug, Clone)]
pub struct BatchPointMapping {
    pub hat_inputs_index: Vec<usize>,
    pub point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub c_hat_index: usize,
    pub c_point_index: (Vec<usize>, Vec<usize>),
    pub challenge_index: usize,
    pub response_index: usize,
}


#[derive(Debug, Clone)]
pub struct BatchPoint<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: BatchPointProtocolInput<F>,
    pub atomic_pop: BatchPointAtomicPoP<F>,
    pub litebullet: LiteBullet<F>,
}

impl<F: PrimeField> BatchPointProtocolInput<F> {
    pub fn new(hat_inputs: Vec<F>, point_inputs: Vec<(Vec<F>, Vec<F>)>) -> Self {
        let num_inputs = hat_inputs.len();
        if num_inputs != point_inputs.len() {
            panic!("Inconsistent input num in BatchPoint");
        }

        if num_inputs == 0 {
            panic!("No inputs provided for BatchPoint");
        }

        let log_m = point_inputs[0].0.len();
        let log_n = point_inputs[0].1.len();

        let shape = (1 << log_m, 1 << log_n);

        for i in 0..num_inputs {
        
            if log_m != point_inputs[i].0.len() || log_n != point_inputs[i].1.len() {
                panic!("Inconsistent point input shape in BatchPoint");
            }
        }


        Self {
            hat_inputs: hat_inputs,
            point_inputs: point_inputs,
            input_mat: DenseMatFieldCM::new(0,0),
            shape: shape,
            num_inputs: num_inputs,
            ready: false,
        }
    }

    pub fn set_input(&mut self, input_mat: DenseMatFieldCM<F>,
    ) {

        let shape = input_mat.shape;

        let log_m = self.point_inputs[0].0.len();
        let log_n = self.point_inputs[0].1.len();

        if shape.0 != 1 << log_m || shape.1 != 1 << log_n {
            panic!("Inconsistent input shape in BatchPoint");
        }

        self.input_mat = input_mat;
        self.ready = true;
    }

    pub fn clear(&mut self) {
        self.input_mat.data = Vec::new();
    }
}

impl<F: PrimeField> BatchPointAtomicPoP<F> {
    pub fn new() -> Self {
        Self {
            hat_inputs: Vec::new(),
            point_inputs: Vec::new(),
            c_hat: F::zero(),
            c_point: (Vec::new(), Vec::new()),
            challenge: F::zero(),
            response: F::zero(),
            mapping: BatchPointMapping {
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
        challenge: F,
        response: F,
        c_hat_index: usize,
        c_point_index: (Vec<usize>, Vec<usize>),
        challenge_index: usize,
        response_index: usize,
    ) {
        self.c_hat = c_hat;
        self.c_point = c_point;
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

    pub fn get_c(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.c_hat.clone(), (self.c_point.0.clone(), self.c_point.1.clone()))
    }

    pub fn get_c_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.c_hat_index, (self.mapping.c_point_index.0.clone(), self.mapping.c_point_index.1.clone()))
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> BatchPoint<F> {
    pub fn new(
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    ) -> Self {
        let mut atomic_pop = BatchPointAtomicPoP::new();

        atomic_pop.set_message(
            hat_inputs.clone(),
            point_inputs.clone(),
            hat_inputs_index,
            point_inputs_index,
        );


        Self {
            protocol_input: BatchPointProtocolInput::new(hat_inputs, point_inputs),
            atomic_pop: atomic_pop,
            litebullet: LiteBullet::new(F::zero(), 0, 0),
        }
    }

    pub fn default() -> Self {
        Self::new(
            vec![F::zero()],                // hat_inputs
            vec![(Vec::new(), Vec::new())], // point_inputs
            vec![0],                        // hat_inputs_index placeholder
            vec![(Vec::new(), Vec::new())], // point_inputs_index placeholder
        )
    }

    pub fn set_input(
        &mut self,
        input_mat: DenseMatFieldCM<F>,
    ) {
        self.protocol_input.set_input(input_mat);
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for BatchPoint<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.litebullet.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        // Implement the reduction logic for the prover
        if !self.protocol_input.ready {
            panic!("⚠️  Protocol input not ready for BatchPoint reduction");
        }

        let m = self.protocol_input.shape.0;
        let n = self.protocol_input.shape.1;
        let len = m * n;
        let num_inputs = self.protocol_input.num_inputs;

        let mat_vec = self.protocol_input.input_mat.to_vec();
        self.protocol_input.clear();

        let challenge_index = trans.pointer;
        let challenge = trans.gen_challenge();

        let mut b_vec = vec![F::zero(); len];
    
        let mut batch_hat = F::zero();
        let mut cur_mul = F::one();

        for i in 0..num_inputs {
            batch_hat = batch_hat + self.protocol_input.hat_inputs[i] * cur_mul;

            let point = self.protocol_input.point_inputs[i].clone();
            let xxxx = [point.1.as_slice(), point.0.as_slice()].concat();
            let xi_cur = xi::xi_from_challenges(&xxxx);
            let xi_cur_mul = linear::vec_scalar_mul(&xi_cur, &cur_mul);

            b_vec = linear::vec_addition(&b_vec, &xi_cur_mul);
    
            cur_mul = cur_mul * challenge;
        }

        let batch_hat_index = trans.pointer;
        trans.push_response(batch_hat);

        self.litebullet = LiteBullet::new(batch_hat, batch_hat_index, len);
        self.litebullet.set_input(mat_vec, b_vec);

        self.litebullet.reduce_prover(trans);

        let c_hat = self.litebullet.atomic_pop.hat_a;
        let c_vec_point = self.litebullet.atomic_pop.point_a.0.clone();
        let c_hat_index = self.litebullet.atomic_pop.mapping.hat_a_index;
        let c_vec_point_index = self.litebullet.atomic_pop.mapping.point_a_index.0.clone();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

       
        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point =  self.protocol_input.point_inputs[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_ip = xi::xi_ip_from_challenges(&b_vec_point, &xxxx);
            b_hat_expected = b_hat_expected + cur_ip * cur_mul;
            cur_mul = cur_mul * challenge;
        }

    let flag = b_hat == b_hat_expected;

        let log_n = n.ilog2() as usize;

        let c_point = (c_vec_point[log_n..].to_vec(), c_vec_point[..log_n].to_vec());
        let c_point_index = (c_vec_point_index[log_n..].to_vec(), c_vec_point_index[..log_n].to_vec());

        self.atomic_pop.set_pop_trans(
            c_hat,
            c_point,
            challenge,
            batch_hat,
            c_hat_index,
            c_point_index,
            challenge_index,
            batch_hat_index,
        );

        flag

    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        
        let m = self.protocol_input.shape.0;
        let n = self.protocol_input.shape.1;
        let len = m * n;
        let num_inputs = self.protocol_input.num_inputs;



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

        assert_eq!(batch_hat, batch_hat_expected, "BatchPoint batch_hat mismatch");

        let flag1 = batch_hat == batch_hat_expected;

        self.litebullet = LiteBullet::new(batch_hat, batch_hat_index, len);

        let flag2 = self.litebullet.verify_as_subprotocol(trans);

        let c_hat = self.litebullet.atomic_pop.hat_a;
        let c_vec_point = self.litebullet.atomic_pop.point_a.0.clone();
        let c_hat_index = self.litebullet.atomic_pop.mapping.hat_a_index;
        let c_vec_point_index = self.litebullet.atomic_pop.mapping.point_a_index.0.clone();

        let b_hat = self.litebullet.atomic_pop.hat_b;
        let b_vec_point = self.litebullet.atomic_pop.point_b.0.clone();
        let _b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _b_vec_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let mut cur_mul = F::one();
        let mut b_hat_expected = F::zero();

        for i in 0..num_inputs {
            let cur_point =  self.protocol_input.point_inputs[i].clone();
            let xxxx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_ip = xi::xi_ip_from_challenges(&b_vec_point, &xxxx);
            b_hat_expected = b_hat_expected + cur_ip * cur_mul;
            cur_mul = cur_mul * challenge;
        }


        let flag3 = b_hat == b_hat_expected;

        let log_n = n.ilog2() as usize;

        let c_point = (c_vec_point[log_n..].to_vec(), c_vec_point[..log_n].to_vec());
        let c_point_index = (c_vec_point_index[log_n..].to_vec(), c_vec_point_index[..log_n].to_vec());

        self.atomic_pop.set_pop_trans(
            c_hat,
            c_point,
            challenge,
            batch_hat,
            c_hat_index,
            c_point_index,
            challenge_index,
            batch_hat_index,
        );


        flag1 && flag2 && flag3
    }

    fn prepare_atomic_pop(&mut self) -> bool {
        if !self.atomic_pop.ready.0 {
            panic!("⚠️  Proof data not ready for BatchPoint pop preparation!!");
        }

        let num_inputs = self.protocol_input.num_inputs;
        // // --- Debug: entering prepare_atomic_pop ---
        // println!("[BatchPoint::prepare_atomic_pop] start: ready = {:?}", self.atomic_pop.ready);

        let hat_inputs_index = self.atomic_pop.mapping.hat_inputs_index.clone();
        let point_inputs_index = self.atomic_pop.mapping.point_inputs_index.clone();
        let _c_hat_index = self.atomic_pop.mapping.c_hat_index; // unused currently
        let _c_point_index = self.atomic_pop.mapping.c_point_index.clone(); // unused currently
        let challenge_index = self.atomic_pop.mapping.challenge_index;
        let response_index = self.atomic_pop.mapping.response_index;
        let b_hat_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let b_point_index = self.litebullet.atomic_pop.mapping.point_b_index.0.clone();

        let hat_inputs_exprs = hat_inputs_index.iter()
            .map(|hat| {
                ArithmeticExpression::input(*hat)
            })
            .collect::<Vec<_>>();

        let xxxx_exprs = point_inputs_index.iter()
            .map(|(xl, xr)| {
                let xr_exprs = xr.iter()
                    .map(|x| ArithmeticExpression::input(*x))
                    .collect::<Vec<_>>();
                let xl_exprs = xl.iter()
                    .map(|x| ArithmeticExpression::input(*x))
                    .collect::<Vec<_>>();
                [xr_exprs.as_slice(), xl_exprs.as_slice()].concat()
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
        // println!("[BatchPoint::prepare_atomic_pop] after set_check: ready = {:?}", self.atomic_pop.ready);  

    
        let mut cur_mul_expr = ArithmeticExpression::constant(F::one());
        let mut link = b_hat_expr;


        for i in 0..num_inputs {
            let xxxx = xxxx_exprs[i].clone();

            let cur_ip_expr = arithmetic_expression::xi_ip_from_challenges_exprs(&b_point_expr, &xxxx);

            link = ArithmeticExpression::sub(
                link,
                ArithmeticExpression::mul(cur_ip_expr, cur_mul_expr.clone()),
            );
            cur_mul_expr = ArithmeticExpression::mul(cur_mul_expr, challenge_expr.clone());
        }

        self.atomic_pop.set_links(vec![link]);
        // println!("[BatchPoint::prepare_atomic_pop] after set_links: ready = {:?}", self.atomic_pop.ready);

        let flag = self.litebullet.prepare_atomic_pop();
        // println!("[BatchPoint::prepare_atomic_pop] after litebullet.prepare: litebullet_ready = {}  final_ready = {:?}", flag, self.atomic_pop.ready);

        self.atomic_pop.is_ready() && flag
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        if !self.atomic_pop.is_ready() { return false; }

        // 1. Add the main 'check' constraint
        cs_builder.add_constraint(self.atomic_pop.check.clone());

        // 2. Add 'link_inputs' constraints
        for constraint in &self.atomic_pop.link_inputs {
            cs_builder.add_constraint(constraint.clone());
        }

        // 3. Synthesize nested LiteBullet constraints
        self.litebullet.synthesize_atomic_pop_constraints(cs_builder)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic_protocol::AtomicMatProtocol;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use mat::utils::matdef::DenseMatFieldCM;
    use crate::pop::arithmetic_expression::ConstraintSystemBuilder;

    // Construct a single matrix and compute hat values for multiple (l,r) points
    fn make_single_mat_and_points(shape: (usize, usize), num_points: usize) -> (
        DenseMatFieldCM<BlsFr>,               // single matrix
        Vec<(Vec<BlsFr>, Vec<BlsFr>)>,        // different projection points
        Vec<BlsFr>,                           // hat value for each point
    ) {
        let (rows, cols) = shape;
        assert!(rows.is_power_of_two() && cols.is_power_of_two());

        // Construct matrix (deterministic)
        let mut mat = DenseMatFieldCM::new(rows, cols);
        let mut data = Vec::new();
        for r in 0..rows {
            let mut row_vec = Vec::new();
            for c in 0..cols {
                row_vec.push(BlsFr::from((r * cols + c + 1) as u64));
            }
            data.push(row_vec);
        }
        mat.set_data(data);

        // Log dimensions
        let log_m = rows.ilog2() as usize;
        let log_n = cols.ilog2() as usize;

        // Generate multiple point groups: simple (base + i) approach
        let mut points = Vec::new();
        for i in 0..num_points {
            let mut l_vec = Vec::new();
            let mut r_vec = Vec::new();
            for j in 0..log_m {
                l_vec.push(BlsFr::from((2 + i + j) as u64));
            }
            for j in 0..log_n {
                r_vec.push(BlsFr::from((3 + i + j) as u64));
            }
            points.push((l_vec, r_vec));
        }

        // Calculate hat for each point
        let mut hats = Vec::new();
        for (l, r) in &points {
            let hat = mat.proj_lr_challenges(l, r);
            hats.push(hat);
        }

        (mat, points, hats)
    }

    #[test]
    fn test_batchpoint_single_matrix_multi_points_roundtrip() {
        let shape = (4, 4);
        let num_points = 4; // power of 2, avoid padding
        let (mat, points, hats) = make_single_mat_and_points(shape, num_points);

        // Construct indices (using sequential indices for now; in practice if transcript positions are needed, should push first then record)
        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        // (l_index_vec, r_index_vec) for each point; using placeholder [0,1] / [0,1] here
        let log_m = shape.0.ilog2() as usize;
        let log_n = shape.1.ilog2() as usize;
        let point_indices = vec![( (0..log_m).collect(), (0..log_n).collect() ); num_points];

        let mut protocol = BatchPoint::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mat);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "reduce_prover failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "verify_as_subprotocol failed");
        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch");
    }

    #[test]
    fn test_batchpoint_single_matrix_multi_points_padding() {
        let shape = (4, 4);
        let num_points = 3; // non-power of 2, test padding branch
        let (mat, points, hats) = make_single_mat_and_points(shape, num_points);

        let hat_indices: Vec<usize> = (0..hats.len()).collect();
        let log_m = shape.0.ilog2() as usize;
        let log_n = shape.1.ilog2() as usize;
        let point_indices = vec![( (0..log_m).collect(), (0..log_n).collect() ); num_points];

        let mut protocol = BatchPoint::<BlsFr>::new(hats.clone(), points.clone(), hat_indices, point_indices);
        protocol.set_input(mat);

        let mut prover_trans = Transcript::new(BlsFr::zero());
        assert!(protocol.reduce_prover(&mut prover_trans), "reduce_prover (padding) failed");

        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_protocol = protocol.clone();
        assert!(v_protocol.verify_as_subprotocol(&mut verifier_trans), "verify_as_subprotocol (padding) failed");
        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch (padding)");
    }

    #[test]
    fn test_batchpoint_single_matrix_synthesize_constraints() {
        // Constraint test: need to push hat_inputs and all point elements to transcript first, and record indices
        let shape = (4, 4);
        let num_points = 4;
        let (mat, points, hats) = make_single_mat_and_points(shape, num_points);

        let mut trans = Transcript::new(BlsFr::zero());

        // Record and push hat_inputs
        let mut hat_indices = Vec::new();
        for h in &hats {
            hat_indices.push(trans.pointer);
            trans.push_response(*h);
        }

        // Record and push point (l,r)
        let mut point_indices: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for (l, r) in &points {
            let mut l_idx = Vec::new();
            for x in l {
                l_idx.push(trans.pointer);
                trans.push_response(*x);
            }
            let mut r_idx = Vec::new();
            for x in r {
                r_idx.push(trans.pointer);
                trans.push_response(*x);
            }
            point_indices.push((l_idx, r_idx));
        }

        // Construct protocol
        let mut protocol = BatchPoint::<BlsFr>::new(hats.clone(), points.clone(), hat_indices.clone(), point_indices.clone());
        protocol.set_input(mat);

        // Execute reduction
        assert!(protocol.reduce_prover(&mut trans), "reduce_prover failed in constraint test");

        // Build constraint system
        let mut cs_builder = ConstraintSystemBuilder::new();
        // Prepare PoP (will build expressions based on mapping)
        assert!(protocol.prepare_atomic_pop(), "prepare_atomic_pop failed");

        // Synthesize constraints
        assert!(protocol.synthesize_atomic_pop_constraints(&mut cs_builder), "synthesize_atomic_pop_constraints failed");

        // Prepare input vector (extract all values from transcript)
        let mut proof_vec = vec![BlsFr::zero()];
        proof_vec.extend(trans.get_fs_proof_vec());
    cs_builder.set_private_inputs(proof_vec);

    // Validate
    let validation = cs_builder.validate_constraints();
        assert!(validation.is_ok(), "Constraints not satisfied: {:?}", validation.err());
    }
}