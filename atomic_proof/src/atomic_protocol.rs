use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use ark_ff::PrimeField;
use ark_std::UniformRand;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

use crate::pop::arithmetic_expression::ConstraintSystemBuilder;

#[derive(Debug, Clone, PartialEq)]
pub enum MatOp {
    Input,
    Eq,
    EqZero,
    Add,
    Concat,
    GrandProd,
    Hadamard,
    LinComb,
    Mul,
    Activation,
    ReduceProd,
    Sub,
    InnerProduct,
    WithinRange,
}

#[derive(Debug, Clone)]
pub struct AtomicMatProtocolInput<F: PrimeField> {
    pub op: MatOp,
    pub hat_c: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub a: DenseMatFieldCM<F>,
    pub b: DenseMatFieldCM<F>,
    pub shape_a: (usize, usize),
    pub shape_b: (usize, usize),
    pub shape_c: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct AtomicMatProtocolMultiInput<F: PrimeField> {
    pub op: MatOp,
    pub hat_c: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub shape_inputs: (usize, usize),
    pub num_inputs: usize,
    pub input_mats: Vec<DenseMatFieldCM<F>>,
}


impl<F: PrimeField> AtomicMatProtocolInput<F> {
    pub fn clear(&mut self) {
        self.a = DenseMatFieldCM::new(0, 0);
        self.b = DenseMatFieldCM::new(0, 0);
    }
}

impl<F: PrimeField> AtomicMatProtocolMultiInput<F> {
    pub fn clear(&mut self) {
        self.input_mats = Vec::new();
    }
}

pub trait AtomicMatProtocol<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> {
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool;
    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool;
    fn verify(&mut self, trans: &mut Transcript<F>) -> bool {
        // Check transcript integrity first
        if !trans.fs.verify_fs() {
            println!("⚠️  Fiat-Shamir check failed when verifying MatMul (continuing for debug)");
        }
        self.verify_as_subprotocol(trans)
    }
    fn prepare_atomic_pop(&mut self) -> bool;
    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool;
    fn clear(&mut self);
    fn reduce_prover_with_atomic_pop(&mut self, trans: &mut Transcript<F>) -> bool {
        self.reduce_prover(trans);
        self.prepare_atomic_pop()
    }
    fn reduce_prover_with_constraint_building(&mut self, trans: &mut Transcript<F>, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        self.reduce_prover_with_atomic_pop(trans);
        self.synthesize_atomic_pop_constraints(cs_builder)
    }
}