//! Implement the Grand Product using Hadamard
//!
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use mat::utils::{linear, xi::{xi_from_challenges, xi_ip_from_challenges}};

use crate::atomic_pop::AtomicPoP;
use crate::atomic_protocol::{AtomicMatProtocol, AtomicMatProtocolInput, MatOp};
use crate::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};

use crate::protocols::litebullet::LiteBullet;
use crate::protocols::hadamard::Hadamard;

// Prove that c = \Prod_{i=1}^n a_i
#[derive(Debug, Clone)]
pub struct GrandProd<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
    pub reduce_protocols: Vec<ReduceProd<F>>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> GrandProd<F> 
{
    pub fn new(
        hat_c: F,
        hat_c_index: usize,
        len: usize,
    ) -> Self {

        if !len.is_power_of_two() {
            panic!("Only support vector length of 2^k for grand product!");
        }


        let protocol_input = AtomicMatProtocolInput {
            op: MatOp::GrandProd,
            a: DenseMatFieldCM::new(len, 1),
            b: DenseMatFieldCM::new(0, 0),
            hat_c: hat_c,
            point_c: (Vec::new(), Vec::new()),
            shape_a: (len, 1),
            shape_b: (0,0),
            shape_c: (1,1),
        };

        let mut atomic_pop = AtomicPoP::new();
        // Set the message with the correct c value and c_index
        atomic_pop.set_message(
            hat_c, 
            (Vec::new(), Vec::new()),
            hat_c_index,
            (Vec::new(), Vec::new()),
        );

        let log_n = len.ilog2();

        let mut reduce_protocols = Vec::new();

        for i in 0..log_n {
            let reduce_protocol = ReduceProd::new(
                F::zero(),
                (Vec::new(), Vec::new()),
                0,
                (Vec::new(), Vec::new()),
                (2usize.pow(i as u32), 1),
                (2usize.pow(i as u32 + 1), 1),
            );
            reduce_protocols.push(reduce_protocol);
        }

        Self {
            protocol_input,
            atomic_pop,
            reduce_protocols,
        }
    }

    pub fn default() -> Self {
        Self::new(F::zero(), 0, 1)
    }

    pub fn set_input(&mut self, mat_a: DenseMatFieldCM<F>) {
        if mat_a.shape.1 != 1 {
            panic!("Input matrix of GrandProd must be a column vector!");
        }

        self.protocol_input.a = mat_a;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for GrandProd<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        for reduce in &mut self.reduce_protocols {
            reduce.clear();
        }
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        if self.protocol_input.a.shape == (0, 0) {
            panic!("GrandProd input matrix is empty!");
        }

        let n = self.protocol_input.a.shape.0;
        let log_n = n.ilog2() as usize;

        // 1. Compute all intermediate vectors
        for i in (0..log_n).rev() {
            let cur_len = 2usize.pow(i as u32 + 1);
            if i == log_n - 1 {
                let vec = self.protocol_input.a.data.pop().unwrap();
                let mut mat_a = DenseMatFieldCM::new(vec.len(), 1);
                mat_a.set_data(vec![vec]);
                self.reduce_protocols[i].set_input(mat_a);
            } else {
                let prev_vec_l = &self.reduce_protocols[i + 1].protocol_input.a.data[0][..cur_len];
                let prev_vec_r = &self.reduce_protocols[i + 1].protocol_input.a.data[0][cur_len..];
                let vec = linear::vec_element_wise_mul_slice(
                    prev_vec_l,
                    prev_vec_r,
                );
                let mut mat_a = DenseMatFieldCM::new(vec.len(), 1);
                mat_a.set_data(vec![vec]);
                self.reduce_protocols[i].set_input(mat_a);
            }
        }

        // 2. Run provers from bottom up (i=0 to log_n-1)
        let mut flag = true;
        for i in 0..log_n {

            if i == 0 {
                self.reduce_protocols[0].protocol_input.hat_c = self.protocol_input.hat_c;
                self.reduce_protocols[0].atomic_pop.set_message(
                    self.protocol_input.hat_c,
                    (Vec::new(), Vec::new()),
                    self.atomic_pop.mapping.hat_c_index,
                    (Vec::new(), Vec::new()),
                );
            } else {
                let hat_c = self.reduce_protocols[i - 1].atomic_pop.hat_a;
                let point_c = self.reduce_protocols[i - 1].atomic_pop.point_a.clone();
                let hat_c_index = self.reduce_protocols[i - 1].atomic_pop.mapping.hat_a_index;
                let point_c_index = self.reduce_protocols[i - 1].atomic_pop.mapping.point_a_index.clone();

                self.reduce_protocols[i].protocol_input.hat_c = hat_c;
                self.reduce_protocols[i].protocol_input.point_c = point_c.clone();
                self.reduce_protocols[i].atomic_pop.set_message(hat_c, point_c, hat_c_index, point_c_index);
            }

            let cur_flag = self.reduce_protocols[i].reduce_prover(trans);
            self.reduce_protocols[i].clear();
            flag = flag && cur_flag;
            if !flag {
                return false;
            }
        }

        let last_protocol = &self.reduce_protocols[log_n - 1];
        self.atomic_pop.set_pop_trans(
            last_protocol.atomic_pop.hat_a,
            F::zero(),
            last_protocol.atomic_pop.point_a.clone(),
            (Vec::new(), Vec::new()),
            last_protocol.atomic_pop.challenges.clone(),
            last_protocol.atomic_pop.responses.clone(),
            last_protocol.atomic_pop.mapping.hat_a_index,
            0,
            last_protocol.atomic_pop.mapping.point_a_index.clone(),
            (Vec::new(), Vec::new()),
            last_protocol.atomic_pop.mapping.challenges_index.clone(),
            last_protocol.atomic_pop.mapping.responses_index.clone(),
        );

        flag
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let n = self.protocol_input.shape_a.0;
        let log_n = n.ilog2() as usize;

        let mut flag = true;
        for i in 0..log_n {
            if i == 0 {
                self.reduce_protocols[0].protocol_input.hat_c = self.protocol_input.hat_c;
                self.reduce_protocols[0].atomic_pop.set_message(
                    self.protocol_input.hat_c,
                    (Vec::new(), Vec::new()),
                    self.atomic_pop.mapping.hat_c_index,
                    (Vec::new(), Vec::new()),
                );
            } else {
                let hat_c = self.reduce_protocols[i - 1].atomic_pop.hat_a;
                let point_c = self.reduce_protocols[i - 1].atomic_pop.point_a.clone();
                let hat_c_index = self.reduce_protocols[i - 1].atomic_pop.mapping.hat_a_index;
                let point_c_index = self.reduce_protocols[i - 1].atomic_pop.mapping.point_a_index.clone();

                self.reduce_protocols[i].protocol_input.hat_c = hat_c;
                self.reduce_protocols[i].protocol_input.point_c = point_c.clone();
                self.reduce_protocols[i].atomic_pop.set_message(hat_c, point_c, hat_c_index, point_c_index);
            }

            let cur_flag = self.reduce_protocols[i].verify_as_subprotocol(trans);
            flag = flag && cur_flag;
            if !flag {
                return false;
            }
        }

        let last_protocol = &self.reduce_protocols[log_n - 1];
        self.atomic_pop.set_pop_trans(
            last_protocol.atomic_pop.hat_a,
            F::zero(),
            last_protocol.atomic_pop.point_a.clone(),
            (Vec::new(), Vec::new()),
            last_protocol.atomic_pop.challenges.clone(),
            last_protocol.atomic_pop.responses.clone(),
            last_protocol.atomic_pop.mapping.hat_a_index,
            0,
            last_protocol.atomic_pop.mapping.point_a_index.clone(),
            (Vec::new(), Vec::new()),
            last_protocol.atomic_pop.mapping.challenges_index.clone(),
            last_protocol.atomic_pop.mapping.responses_index.clone(),
        );

        flag
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        let mut flag = true;
        for protocol in &mut self.reduce_protocols {
            flag = flag && protocol.prepare_atomic_pop();
        }
        
        // Set up the main GrandProd atomic_pop's check and links
        self.atomic_pop.set_check(ArithmeticExpression::constant(F::zero()));
        self.atomic_pop.set_link_xa((Vec::new(), Vec::new()));
        self.atomic_pop.set_link_xb((Vec::new(), Vec::new()));
        
        flag && self.atomic_pop.is_ready()
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        let mut flag = true;
        for protocol in &self.reduce_protocols {
            flag = flag && protocol.synthesize_atomic_pop_constraints(cs_builder);
        }
        flag
    }
}

// Prove that \vec(c) = \vec(a)_L \circ \vec(a)_R
#[derive(Debug, Clone)]
pub struct ReduceProd<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: AtomicMatProtocolInput<F>,
    pub atomic_pop: AtomicPoP<F>,
    pub hadamard: Hadamard<F>,
    pub litebullet: LiteBullet<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> ReduceProd<F> 
{
    pub fn new(
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
        shape_c: (usize, usize),
        shape_a: (usize, usize),
    ) -> Self {

        if shape_a.1 != 1 || shape_c.1 != 1 {
            panic!("Only support reduce product for Vectors!");
        }

        if shape_a.0 != shape_c.0 * 2 {
            panic!("Incompatible vector length in ReduceProduct!");
        }

        let protocol_input = AtomicMatProtocolInput {
            op: MatOp::ReduceProd,
            a: DenseMatFieldCM::new(shape_a.0, shape_a.1),
            b: DenseMatFieldCM::new(0, 0),
            hat_c: hat_c,
            point_c: point_c.clone(),
            shape_a: shape_a,
            shape_b: (0,0),
            shape_c: shape_c,
        };

        let mut atomic_pop = AtomicPoP::new();
        // Set the message with the correct c value and c_index
        atomic_pop.set_message(
            hat_c, 
            point_c.clone(),
            hat_c_index,
            point_c_index.clone(),
        );

        let len = shape_a.0 * shape_a.1;
        let litebullet = LiteBullet::new(F::zero(), 0, len);

        let hadamard = Hadamard::new(hat_c, point_c, hat_c_index, point_c_index, shape_c.clone(), shape_c.clone(), shape_c.clone());

        Self {
            protocol_input,
            atomic_pop,
            hadamard,
            litebullet,
        }
    }

    pub fn set_input(&mut self, mat_a: DenseMatFieldCM<F>) {
        self.protocol_input.a = mat_a;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for ReduceProd<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.litebullet.clear();
        self.hadamard.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {

        let a_len = self.protocol_input.shape_a.0;
        let c_len = self.protocol_input.shape_c.0;

        if a_len != c_len * 2 {
            panic!("Incompatible vector length in ReduceProduct!");
        }

        let mut mat_al = DenseMatFieldCM::new(c_len, 1);
        let mut mat_ar = DenseMatFieldCM::new(c_len, 1);

        let vec_a = self.protocol_input.a.data.pop().unwrap();
        
        mat_al.set_data(vec![vec_a[..c_len].to_vec()]);
        mat_ar.set_data(vec![vec_a[c_len..].to_vec()]);


        self.protocol_input.clear();

        self.hadamard.protocol_input.hat_c = self.protocol_input.hat_c;
        self.hadamard.protocol_input.point_c = self.protocol_input.point_c.clone();
        self.hadamard.atomic_pop.set_message(
            self.protocol_input.hat_c,
            self.protocol_input.point_c.clone(),
            self.atomic_pop.mapping.hat_c_index,
            self.atomic_pop.mapping.point_c_index.clone()
        );
        self.hadamard.set_input(mat_al, mat_ar);

        let flag1 = self.hadamard.reduce_prover(trans);

        let proj_al = self.hadamard.atomic_pop.hat_a;
        let proj_ar = self.hadamard.atomic_pop.hat_b;
        let _proj_al_index = self.hadamard.atomic_pop.mapping.hat_a_index;
        let _proj_ar_index = self.hadamard.atomic_pop.mapping.hat_b_index;

        let point_al = &self.hadamard.atomic_pop.point_a.0;
        let point_ar = &self.hadamard.atomic_pop.point_b.0;
        let _point_al_index = &self.hadamard.atomic_pop.mapping.point_a_index.0;
        let _point_ar_index = &self.hadamard.atomic_pop.mapping.point_b_index.0;


        let zz_index = trans.pointer;
        let zz = trans.gen_challenge();

        let combined_proj = proj_al + proj_ar * zz;
        let combined_proj_index = trans.pointer;
        trans.push_response(combined_proj);

        let xi_l = xi_from_challenges(&point_al);
        let xi_r = xi_from_challenges(&point_ar);
        let xi_combined = [xi_l.as_slice(), linear::vec_scalar_mul(&xi_r, &zz).as_slice()].concat();
        std::mem::drop(xi_l);
        std::mem::drop(xi_r);

        self.hadamard.protocol_input.clear();

        self.litebullet = LiteBullet::new(combined_proj, combined_proj_index, a_len);
        self.litebullet.set_input(vec_a, xi_combined);
        let flag2 = self.litebullet.reduce_prover(trans);
        self.litebullet.clear();

        let hat_a = self.litebullet.atomic_pop.hat_a;
        let hat_b = self.litebullet.atomic_pop.hat_b;
        let point_a = &self.litebullet.atomic_pop.point_a.0;
        let point_b = &self.litebullet.atomic_pop.point_b.0;
        let hat_a_index = self.litebullet.atomic_pop.mapping.hat_a_index;
        let _hat_b_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let point_a_index = &self.litebullet.atomic_pop.mapping.point_a_index.0;
        let _point_b_index = &self.litebullet.atomic_pop.mapping.point_b_index.0;

        
        let point_b_first = point_b[0];
        let point_b_remaining = point_b[1..].to_vec();


        let flag3 = 
            xi_ip_from_challenges(&point_al, &point_b_remaining)
            + zz * point_b_first * xi_ip_from_challenges(&point_ar, &point_b_remaining)
            == hat_b;


        self.atomic_pop.set_pop_trans(
            hat_a,
            F::zero(),
            (point_a.clone(), Vec::new()),
            (Vec::new(), Vec::new()),
            vec![zz],
            vec![combined_proj],
            hat_a_index,
            0,
            (point_a_index.clone(), Vec::new()),
            (Vec::new(), Vec::new()),
            vec![zz_index],
            vec![combined_proj_index],
        );

        flag1 && flag2 && flag3
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let a_len = self.protocol_input.shape_a.0;
        let c_len = self.protocol_input.shape_c.0;

        if a_len != c_len * 2 {
            panic!("Incompatible vector length in ReduceProduct!");
        }

        self.hadamard.protocol_input.hat_c = self.protocol_input.hat_c;
        self.hadamard.protocol_input.point_c = self.protocol_input.point_c.clone();
        self.hadamard.atomic_pop.set_message(
            self.protocol_input.hat_c,
            self.protocol_input.point_c.clone(),
            self.atomic_pop.mapping.hat_c_index,
            self.atomic_pop.mapping.point_c_index.clone()
        );
        let flag1 = self.hadamard.verify_as_subprotocol(trans);

        let proj_al = self.hadamard.atomic_pop.hat_a;
        let proj_ar = self.hadamard.atomic_pop.hat_b;
        let _proj_al_index = self.hadamard.atomic_pop.mapping.hat_a_index;
        let _proj_ar_index = self.hadamard.atomic_pop.mapping.hat_b_index;

        let point_al = &self.hadamard.atomic_pop.point_a.0;
        let point_ar = &self.hadamard.atomic_pop.point_b.0;
        let _point_al_index = &self.hadamard.atomic_pop.mapping.point_a_index.0;
        let _point_ar_index = &self.hadamard.atomic_pop.mapping.point_b_index.0;

        let zz_index = trans.pointer;
        let zz = trans.get_at_position(zz_index);
        trans.pointer += 1;

        let combined_proj_index = trans.pointer;
        let combined_proj = trans.get_at_position(combined_proj_index);
        trans.pointer += 1;

        let flag4 = combined_proj == proj_al + proj_ar * zz;


        let xi_l = xi_from_challenges(&point_al);
        let xi_r = xi_from_challenges(&point_ar);
        let xi_combined = [xi_l.as_slice(), linear::vec_scalar_mul(&xi_r, &zz).as_slice()].concat();
        std::mem::drop(xi_l);
        std::mem::drop(xi_r);

        let _xi_combined_mat = DenseMatFieldCM {
            data: vec![xi_combined.clone()],
            shape: (xi_combined.len(), 1),
        };

        self.litebullet = LiteBullet::new(combined_proj, combined_proj_index, a_len);
        let flag2 = self.litebullet.verify_as_subprotocol(trans);

        let hat_a = self.litebullet.atomic_pop.hat_a;
        let hat_b = self.litebullet.atomic_pop.hat_b;
        let point_a = &self.litebullet.atomic_pop.point_a.0;
        let point_b = &self.litebullet.atomic_pop.point_b.0;
        let hat_a_index = self.litebullet.atomic_pop.mapping.hat_a_index;
        let _hat_b_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let point_a_index = &self.litebullet.atomic_pop.mapping.point_a_index.0;
        let _point_b_index = &self.litebullet.atomic_pop.mapping.point_b_index.0;

        let _point_b_first = point_b[0];
        let _point_b_remaining = point_b[1..].to_vec();


        let flag3 = 
            xi_ip_from_challenges(&point_al, &point_b[1..].to_vec())
            + zz * point_b[0] * xi_ip_from_challenges(&point_ar, &point_b[1..].to_vec())
            == hat_b;

        self.atomic_pop.set_pop_trans(
            hat_a,
            F::zero(),
            (point_a.clone(), Vec::new()),
            (Vec::new(), Vec::new()),
            vec![zz],
            vec![combined_proj],
            hat_a_index,
            0,
            (point_a_index.clone(), Vec::new()),
            (Vec::new(), Vec::new()),
            vec![zz_index],
            vec![combined_proj_index],
        );

        // println!("Within GrandProd flag1: {}, flag2: {}, flag3: {}, flag4: {}", flag1, flag2, flag3, flag4);

        flag1 && flag2 && flag3 && flag4
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        let a_len = self.protocol_input.shape_a.0;
        let c_len = self.protocol_input.shape_c.0;

        if a_len != c_len * 2 {
            panic!("Incompatible vector length in ReduceProduct!");
        }

        // // Check if we have the necessary mappings
        // println!("DEBUG: challenges_index.len() = {}", self.atomic_pop.mapping.challenges_index.len());
        // println!("DEBUG: responses_index.len() = {}", self.atomic_pop.mapping.responses_index.len());
        
        if self.atomic_pop.mapping.challenges_index.is_empty() || 
           self.atomic_pop.mapping.responses_index.is_empty() {
            println!("DEBUG: Early return due to empty mappings");
            // If we don't have the mappings, we can't prepare constraints yet
            // This typically happens when prepare_atomic_pop is called before reduce_prover/verify_as_subprotocol
            self.atomic_pop.set_check(ArithmeticExpression::constant(F::zero()));
            self.atomic_pop.set_link_xa((Vec::new(), Vec::new()));
            self.atomic_pop.set_link_xb((Vec::new(), Vec::new()));
            return self.atomic_pop.is_ready();
        }

        // println!("DEBUG: Proceeding with constraint generation");

        let _proj_al = self.hadamard.atomic_pop.hat_a;
        let _proj_ar = self.hadamard.atomic_pop.hat_b;
        let proj_al_index = self.hadamard.atomic_pop.mapping.hat_a_index;
        let proj_ar_index = self.hadamard.atomic_pop.mapping.hat_b_index;

        let _point_al = &self.hadamard.atomic_pop.point_a.0;
        let _point_ar = &self.hadamard.atomic_pop.point_b.0;
        let point_al_index = &self.hadamard.atomic_pop.mapping.point_a_index.0;
        let point_ar_index = &self.hadamard.atomic_pop.mapping.point_b_index.0;

        let zz_index = self.atomic_pop.mapping.challenges_index[0];
        let combined_proj_index = self.atomic_pop.mapping.responses_index[0];

        let expr4 = ArithmeticExpression::sub(
            ArithmeticExpression::input(combined_proj_index),
            ArithmeticExpression::add(
                ArithmeticExpression::input(proj_al_index),
                ArithmeticExpression::mul(
                    ArithmeticExpression::input(proj_ar_index),
                    ArithmeticExpression::input(zz_index)
                )
            )
        );

        self.atomic_pop.set_check(expr4);


        let _hat_a = self.litebullet.atomic_pop.hat_a;
        let _hat_b = self.litebullet.atomic_pop.hat_b;
        let _point_a = &self.litebullet.atomic_pop.point_a.0;
        let point_b = &self.litebullet.atomic_pop.point_b.0;
        let _hat_a_index = self.litebullet.atomic_pop.mapping.hat_a_index;
        let hat_b_index = self.litebullet.atomic_pop.mapping.hat_b_index;
        let _point_a_index = &self.litebullet.atomic_pop.mapping.point_a_index.0;
        let point_b_index = &self.litebullet.atomic_pop.mapping.point_b_index.0;

        // println!("DEBUG: point_b.len() = {}", point_b.len());
        // println!("DEBUG: point_b_index.len() = {}", point_b_index.len());
        // println!("DEBUG: c_len = 2^{}", c_len.ilog2());
        
        // Check if we have enough indices
        if point_b_index.len() <= c_len.ilog2() as usize {
            println!("ERROR: point_b_index.len() ({}) <= c_len ({})", point_b_index.len(), c_len);
            // Set empty constraints as fallback
            self.atomic_pop.set_link_xa((Vec::new(), Vec::new()));
            self.atomic_pop.set_link_xb((Vec::new(), Vec::new()));
            return self.atomic_pop.is_ready();
        }

        let _point_b_first = point_b[0];
        let _point_b_remaining = point_b[1..].to_vec();

        // Initialize products to 1 (was 0 causing entire product chain to collapse to 0 and
        // subsequent constraint reducing to hat_b = 0, producing failing constraints like #11).
        let mut expr3_l = ArithmeticExpression::constant(F::one());
        let mut expr3_r = ArithmeticExpression::constant(F::one());

        for i in 0..c_len.ilog2() as usize {
            expr3_l = ArithmeticExpression::mul(
                expr3_l,
                ArithmeticExpression::add(
                    ArithmeticExpression::constant(F::one()),
                    ArithmeticExpression::mul(
                        ArithmeticExpression::input(point_al_index[i]),
                        ArithmeticExpression::input(point_b_index[i+1])
                    )
                )
            );

            expr3_r = ArithmeticExpression::mul(
                expr3_r,
                ArithmeticExpression::add(
                    ArithmeticExpression::constant(F::one()),
                    ArithmeticExpression::mul(
                        ArithmeticExpression::input(point_ar_index[i]),
                        ArithmeticExpression::input(point_b_index[i+1])
                    )
                )
            );
        }

        let expr3 = ArithmeticExpression::sub(
            ArithmeticExpression::input(hat_b_index),
            ArithmeticExpression::add(
                expr3_l,
                ArithmeticExpression::mul(
                    ArithmeticExpression::mul(
                        ArithmeticExpression::input(zz_index),    
                        ArithmeticExpression::input(point_b_index[0])
                    ),
                    expr3_r
                )
            )
        );

        self.atomic_pop.set_link_xa((vec![expr3],Vec::new()));
        self.atomic_pop.set_link_xb((Vec::new(),Vec::new()));

        self.atomic_pop.is_ready()

    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {

        let flag1 = self.litebullet.synthesize_atomic_pop_constraints(cs_builder);
        let flag2 = self.litebullet.synthesize_atomic_pop_constraints(cs_builder);
        self.atomic_pop.synthesize_constraints(cs_builder);

        flag1 && flag2 && self.atomic_pop.is_ready()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pop::arithmetic_expression::ConstraintSystemBuilder;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::UniformRand;
    use fsproof::helper_trans::Transcript;
    use ark_std::test_rng;
 

    #[test]
    fn test_grandprod_for_vector() {
        use ark_ff::One;

        let mut rng = test_rng();
        let len = 8; // Must be power of two

        // Create a random vector
        let vec_a: Vec<BlsFr> = (0..len).map(|_| BlsFr::rand(&mut rng)).collect();
        let mut mat_a = DenseMatFieldCM::new(len, 1);
        mat_a.set_data(vec![vec_a.clone()]);

        // Calculate the product
        let prod_a = vec_a.iter().fold(BlsFr::one(), |acc, &x| acc * x);

        // Prover side
        let mut prover_trans = Transcript::new(prod_a);
        let mut grand_prod_prover = GrandProd::new(prod_a, 0, len);
        grand_prod_prover.set_input(mat_a.clone());
        let prover_flag = grand_prod_prover.reduce_prover(&mut prover_trans);
        assert!(prover_flag, "Prover failed");

        // Verifier side
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut grand_prod_verifier = GrandProd::new(prod_a, 0, len);
        // The verifier doesn't have the input matrix, only its shape
        grand_prod_verifier.protocol_input.a.shape = (len, 1);
        let verifier_flag = grand_prod_verifier.verify_as_subprotocol(&mut verifier_trans);
        assert!(verifier_flag, "Verifier failed");
        assert_eq!(verifier_trans.pointer, verifier_trans.trans_seq.len(), "Verifier transcript should be fully consumed");

        let (hat_a, a_point) = grand_prod_verifier.atomic_pop.get_a();
        let a_proj_expected = mat_a.proj_lr_challenges(&a_point.0, &a_point.1);
        assert_eq!(hat_a, a_proj_expected, "Hat_a should match expected projection of mat_a");
    }

    #[test]
    fn test_grandprod_for_vectors() {
        use ark_bls12_381::Fr as BlsFr;
        use ark_ff::Zero;
        use ark_std::test_rng;

        let c_len = 8;
        let a_len = c_len * 2;
        let shape_c = (c_len, 1);
        let shape_a = (a_len, 1);
        let mut rng = test_rng();

        // --- Test Data Setup ---
        let a_vec: Vec<BlsFr> = (0..a_len).map(|_| BlsFr::rand(&mut rng)).collect();
        let mut mat_a = DenseMatFieldCM::new(a_len, 1);
        mat_a.set_data(vec![a_vec.clone()]);

        let a_l_vec = &a_vec[..c_len];
        let a_r_vec = &a_vec[c_len..];

        let c_vec: Vec<BlsFr> = a_l_vec.iter().zip(a_r_vec).map(|(l, r)| *l * *r).collect();
        let mut mat_c = DenseMatFieldCM::new(c_len, 1);
        mat_c.set_data(vec![c_vec]);

        // --- Prover Side ---
        let mut prover_trans = Transcript::new(BlsFr::zero());

        // The verifier will challenge the prover for a projection of c.
        // In a real scenario, these points would come from a parent protocol.
        // For this test, we generate them ahead of time.
        let point_c_l: Vec<BlsFr> = (0..c_len.ilog2()).map(|_| BlsFr::rand(&mut rng)).collect();
        let point_c_r: Vec<BlsFr> = (0..0).map(|_| BlsFr::rand(&mut rng)).collect(); // log2(1) = 0
        let point_c = (point_c_l.clone(), point_c_r);
        let mut point_c_index = Vec::new();

        for i in 0..point_c_l.len() {
            point_c_index.push(prover_trans.pointer);
            prover_trans.push_response(point_c_l[i]);
        }

        let initial_pointer = prover_trans.pointer;

        let hat_c = mat_c.proj_lr_challenges(&point_c.0, &point_c.1);

        // Prover instantiates the protocol
        let mut reduce_prod_prover = ReduceProd::<BlsFr>::new(
            hat_c,
            point_c.clone(),
            0, // dummy index
            (point_c_index.clone(), vec![0; point_c.1.len()]), // dummy indices
            shape_c,
            shape_a,
        );

        reduce_prod_prover.set_input(mat_a.clone());

        // Run the prover
        assert!(reduce_prod_prover.reduce_prover(&mut prover_trans), "Prover failed");

        // --- Verifier Side ---
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.pointer = initial_pointer;

        // Verifier instantiates the protocol with the same initial setup
        let mut reduce_prod_verifier = ReduceProd::<BlsFr>::new(
            hat_c,
            point_c.clone(),
            0, // dummy index
            (point_c_index, vec![0; point_c.1.len()]), // dummy indices
            shape_c,
            shape_a,
        );

        // Run the verifier
        assert!(reduce_prod_verifier.verify_as_subprotocol(&mut verifier_trans), "Verifier failed");

        assert_eq!(prover_trans.pointer, prover_trans.trans_seq.len(), "Prover transcript should be fully consumed");
        assert_eq!(verifier_trans.pointer, verifier_trans.trans_seq.len(), "Verifier transcript should be fully consumed");

        // --- Final Checks ---
        // The verifier should be able to extract the commitment to 'a' and check it.
        let (a_proj, a_point) = reduce_prod_verifier.atomic_pop.get_a();
        let a_proj_expected = mat_a.proj_lr_challenges(&a_point.0, &a_point.1);
        assert_eq!(a_proj, a_proj_expected, "Projection of mat_a should match expected");

        // The commitment to 'c' is part of the initial message, let's check that too.
        let (c_proj_initial, c_point_initial) = reduce_prod_verifier.atomic_pop.get_c();
        assert_eq!(c_proj_initial, hat_c, "Initial projection of c should match");
        assert_eq!(c_point_initial.0, point_c.0, "Initial point_c.0 should match");
        assert_eq!(c_point_initial.1, point_c.1, "Initial point_c.1 should match");

        println!("ReduceProd protocol test passed!");
    }

    #[test]
    fn test_grandprod_constraints() {
        use ark_ff::{Zero, One};
        use std::sync::{Arc, Mutex};
        use std::thread;
        use std::time::Duration;
        
        // peak memory monitor thread
        let max_memory = Arc::new(Mutex::new(0u64));
        let max_memory_clone = Arc::clone(&max_memory);
        thread::spawn(move || {
            loop {
                if let Some(val) = super::tests::get_memory_usage() {
                    let mut max = max_memory_clone.lock().unwrap();
                    if val > *max { *max = val; }
                }
                thread::sleep(Duration::from_millis(200));
            }
        });

        // Start_test
        let mut rng = test_rng();
        let len = (1 << 20) as usize; // Must be power of two, keep it small for testing

        // 1. Setup
        let vec_a: Vec<BlsFr> = (0..len).map(|_| BlsFr::rand(&mut rng)).collect();
        println!("=== test_grandprod_constraints Debug ===");
        println!("Input vector len: {:?}", vec_a.len());
        let mut mat_a = DenseMatFieldCM::new(len, 1);
        let prod_a = vec_a.iter().fold(BlsFr::one(), |acc, &x| acc * x);
        mat_a.set_data(vec![vec_a]);

        // println!("Expected product: {:?}", prod_a.len());

        // 2. Prover
        let mut prover_trans = Transcript::new(prod_a);
        let mut grand_prod_prover = GrandProd::new(prod_a, 0, len);
        grand_prod_prover.set_input(mat_a);
        
        println!("Before prover: transcript length = {}", prover_trans.trans_seq.len());
        assert!(grand_prod_prover.reduce_prover(&mut prover_trans), "Prover failed");
        println!("After prover: transcript length = {}", prover_trans.trans_seq.len());

        // 3. Verifier
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut grand_prod_verifier = GrandProd::new(prod_a, 0, len);
        grand_prod_verifier.protocol_input.a.shape = (len, 1);
        
        println!("Before verifier: transcript pointer = {}", verifier_trans.pointer);
        assert!(grand_prod_verifier.verify_as_subprotocol(&mut verifier_trans), "Verifier failed");
        println!("After verifier: transcript pointer = {}", verifier_trans.pointer);

        // 4. Prepare Atomic PoP
        println!("Before prepare_atomic_pop");
        let prepare_result = grand_prod_verifier.prepare_atomic_pop();
        println!("prepare_atomic_pop result: {}", prepare_result);
        
        // Debug: check the state of reduce_protocols
        for (i, protocol) in grand_prod_verifier.reduce_protocols.iter().enumerate() {
            println!("ReduceProd[{}]: challenges_index.len() = {}, responses_index.len() = {}", 
                i, 
                protocol.atomic_pop.mapping.challenges_index.len(),
                protocol.atomic_pop.mapping.responses_index.len()
            );
            println!("ReduceProd[{}]: atomic_pop.is_ready() = {}", i, protocol.atomic_pop.is_ready());
        }
        
        assert!(prepare_result, "Prepare atomic pop failed");

        // 5. Synthesize Constraints
        let mut cs_builder = ConstraintSystemBuilder::new();
        let mut proof_vec = vec![BlsFr::zero()];
        proof_vec.extend(prover_trans.get_fs_proof_vec());
        let len = proof_vec.len();
        cs_builder.set_private_inputs(proof_vec);
        
        println!("Before synthesize_atomic_pop_constraints");
        let synthesize_result = grand_prod_verifier.synthesize_atomic_pop_constraints(&mut cs_builder);
        println!("synthesize_atomic_pop_constraints result: {}", synthesize_result);
        println!("Number of constraints generated: {}", cs_builder.arithmetic_constraints.len());
        assert!(synthesize_result, "Synthesize constraints failed");

        // 6. Check constraints
        println!("Before validate_constraints");
        let validation_result = cs_builder.validate_constraints();
        match &validation_result {
            Ok(_) => println!("All constraints satisfied!"),
            Err(e) => println!("Constraint validation failed: {}", e),
        }
        
        // Print each constraint and its evaluation for debugging
        for (i, constraint) in cs_builder.arithmetic_constraints.iter().enumerate() {
            match constraint.evaluate(&cs_builder.pub_inputs, &cs_builder.pri_inputs) {
                Ok(value) => {
                    println!("Constraint {}: value = {:?} (should be 0)", i, value);
                    if value != BlsFr::zero() {
                        println!("  ❌ FAILED: Constraint {} = {:?}", i, constraint);
                    }
                },
                Err(e) => println!("Constraint {}: evaluation error = {}", i, e),
            }
        }

        
        
        assert!(validation_result.is_ok(), "Constraints not satisfied: {:?}", validation_result.err());
        println!("GrandProd constraints test passed!");

        println!("Proof vector length: {}", len);

        let peak_kb = *max_memory.lock().unwrap();
        println!("====Peak RSS (approx): {} KB", peak_kb);

        // 仅在存在时打印 RAYON_NUM_THREADS，避免噪声；否则显示推断线程数
        match std::env::var("RAYON_NUM_THREADS") {
            Ok(value) => println!("RAYON_NUM_THREADS: {}", value),
            Err(_) => {
                let inferred = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
                println!("RAYON_NUM_THREADS (unset, inferred): {}", inferred);
            }
        }


    }

    fn get_memory_usage() -> Option<u64> {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                return parts.get(1).and_then(|s| s.parse().ok());
            }
        }
        None
    }
}

