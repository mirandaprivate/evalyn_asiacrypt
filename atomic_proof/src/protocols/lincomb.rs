//! Linear combination of a series of matrices.
//!
//! This protocol proves that an output projection `hat_c` equals the linear combination
//! of the projections of several input matrices under the SAME bilinear evaluation point `(xl, xr)`:
//! `hat_c = \sum_i coeff[i] * <A_i, (xl, xr)>`.
//!
//! The matrices all share the same shape (power-of-two dimensions). Unlike `Concat`, no
//! extra selector challenges are needed; the point length is simply `log2(rows)` and
//! `log2(cols)` for left/right respectively.
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;
use mat::utils::linear; // removed unused xi

use crate::atomic_pop::AtomicMultiPoP;
use crate::atomic_protocol::{AtomicMatProtocolMultiInput, MatOp, AtomicMatProtocol};
use crate::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};


#[derive(Debug, Clone)]
pub struct LinComb<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> {
    pub coeff: Vec<F>,              // coefficients for linear combination (public constants)
    pub coeff_index: Vec<usize>,    // (optional) transcript indices for coeff if later exposed
    pub protocol_input: AtomicMatProtocolMultiInput<F>,
    pub atomic_pop: AtomicMultiPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> LinComb<F> {
    pub fn new(
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
        shape: (usize, usize),
        num: usize,
        coeff: Vec<F>,
        coeff_index: Vec<usize>,
     ) -> Self {
        if !shape.0.is_power_of_two() || !shape.1.is_power_of_two() {
            panic!("Input dimensions must be powers of two in LinComb");
        }

        let protocol_input = AtomicMatProtocolMultiInput {
            op: MatOp::LinComb,
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


        Self { coeff, coeff_index, protocol_input, atomic_pop }
    }

    pub fn default() -> Self {
        Self::new(
            F::zero(),
            (Vec::new(), Vec::new()),
            0,
            (Vec::new(), Vec::new()),
            (1, 1),
            0,
            Vec::new(),
            Vec::new(),
        )
    }

    pub fn set_input(&mut self, input_mats: Vec<DenseMatFieldCM<F>>) {
        self.protocol_input.input_mats = input_mats;
    }

    // For linear combinations with constant coefficients
    pub fn prepare_atomic_pop_with_constant_coeff(&mut self) -> bool {
        // Only require base transcript data (ready.0). We'll create check & mark readiness here.
        if !self.atomic_pop.ready.0 {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop base data not ready in LinComb before constraint generation");
            return false;
        }


        // check: hat_c - sum coeff[i] * hat_input[i] == 0
        let mut expr = ArithmeticExpression::input(self.atomic_pop.mapping.hat_c_index);
        for (i, coeff) in self.coeff.iter().enumerate() {
            let term = ArithmeticExpression::mul(
                ArithmeticExpression::constant(*coeff),
                ArithmeticExpression::input(self.atomic_pop.mapping.hat_inputs_index[i]),
            );
            expr = ArithmeticExpression::sub(expr, term);
        }
        self.atomic_pop.set_check(expr);
        // No additional link constraints required (all input points equal point_c and already embedded)
        self.atomic_pop.set_link_inputs(Vec::new());
        self.atomic_pop.is_ready()
    }


}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for LinComb<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        let input_row_num = self.protocol_input.shape_inputs.0;
        let input_col_num = self.protocol_input.shape_inputs.1;
        let num = self.protocol_input.num_inputs;

        let log_input_row_num = input_row_num.ilog2() as usize;
        let log_input_col_num = input_col_num.ilog2() as usize;
        if self.protocol_input.point_c.0.len() != log_input_row_num || self.protocol_input.point_c.1.len() != log_input_col_num {
            panic!("!! Invalid point_c shapes in LinComb: expected ({},{}) got ({},{})",
                log_input_row_num, log_input_col_num,
                self.protocol_input.point_c.0.len(), self.protocol_input.point_c.1.len());
        }

        let point = self.protocol_input.point_c.clone(); // (xl, xr)
        let mut input_hats: Vec<F> = Vec::new();
        let mut input_hats_index: Vec<usize> = Vec::new();

        for i in 0..num {
            if self.protocol_input.input_mats[i].shape != (input_row_num, input_col_num) {
                panic!("Input matrix shape mismatch in LinComb");
            }
            let hat_input = self.protocol_input.input_mats[i].proj_lr_challenges(&point.0, &point.1);
            input_hats.push(hat_input);
            input_hats_index.push(trans.pointer);
            trans.push_response(hat_input);
        }

        let hat_c_expected = linear::inner_product(&input_hats, &self.coeff);
        let flag = hat_c_expected == self.protocol_input.hat_c;
        if !flag {
            println!("[lincomb reduce_prover] hat_c mismatch expected {:?} got {:?}", hat_c_expected, self.protocol_input.hat_c);
        }

        self.atomic_pop.set_pop_trans(
            input_hats,
            vec![self.protocol_input.point_c.clone(); num],
            Vec::new(),
            Vec::new(),
            input_hats_index,
            vec![self.atomic_pop.mapping.point_c_index.clone(); num],
            Vec::new(),
            Vec::new(),
        );

        flag

    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let input_row_num = self.protocol_input.shape_inputs.0;
        let input_col_num = self.protocol_input.shape_inputs.1;
        let num = self.protocol_input.num_inputs;

        let log_input_row_num = input_row_num.ilog2() as usize;
        let log_input_col_num = input_col_num.ilog2() as usize;
        if self.protocol_input.point_c.0.len() != log_input_row_num || self.protocol_input.point_c.1.len() != log_input_col_num {
            panic!("!! Invalid point_c shapes in LinComb (verify)!");
        }

        let mut input_hats_index: Vec<usize> = Vec::new();
        let mut input_hats: Vec<F> = Vec::new();
        for _ in 0..num {
            input_hats_index.push(trans.pointer);
            let hat_i = trans.get_at_position(trans.pointer);
            input_hats.push(hat_i);
            trans.pointer += 1;
        }

        // Read coefficients by their recorded transcript indices (do not consume pointer).
        let mut coeff_read = Vec::with_capacity(self.coeff.len());
        if self.coeff_index.len() != self.coeff.len() {
            println!(
                "[lincomb verify] coeff_index length mismatch: coeff={} index={}",
                self.coeff.len(),
                self.coeff_index.len()
            );
        }
        for &idx in self.coeff_index.iter() {
            coeff_read.push(trans.get_at_position(idx));
        }

        let hat_c_expected = linear::inner_product(&input_hats, &coeff_read);
        let flag = hat_c_expected == self.protocol_input.hat_c;
        if !flag {
            println!("[lincomb verify] hat_c mismatch expected {:?} got {:?}", hat_c_expected, self.protocol_input.hat_c);
        }

    self.atomic_pop.set_pop_trans(
            input_hats,
            vec![self.protocol_input.point_c.clone(); num],
            Vec::new(),
            Vec::new(),
            input_hats_index,
            vec![self.atomic_pop.mapping.point_c_index.clone(); num],
            Vec::new(),
            Vec::new(),
        );

        flag
    }

    fn prepare_atomic_pop(&mut self) -> bool {
        // Only require base transcript data (ready.0). We'll create check & mark readiness here.
        if !self.atomic_pop.ready.0 {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop base data not ready in LinComb before constraint generation");
            return false;
        }

        let mut coeff_expr = Vec::new();
        for idx in self.coeff_index.clone() {
            coeff_expr.push(ArithmeticExpression::input(idx));
        }

        // check: hat_c - sum coeff[i] * hat_input[i] == 0
        let mut expr = ArithmeticExpression::input(self.atomic_pop.mapping.hat_c_index);
        for (i, _) in self.coeff.iter().enumerate() {
            let term = ArithmeticExpression::mul(
                coeff_expr[i].clone(),
                ArithmeticExpression::input(self.atomic_pop.mapping.hat_inputs_index[i]),
            );
            expr = ArithmeticExpression::sub(expr, term);
        }
        self.atomic_pop.set_check(expr);
        // No additional link constraints required (all input points equal point_c and already embedded)
        self.atomic_pop.set_link_inputs(Vec::new());
        self.atomic_pop.is_ready()
    }


    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in LinComb");
            return false;
        }
        self.atomic_pop.synthesize_constraints(cs_builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::{UniformRand, Zero};
   
    // Helper: build random matrix of given shape
    #[allow(dead_code)]
    fn rand_mat(rows: usize, cols: usize, rng: &mut impl ark_std::rand::RngCore) -> DenseMatFieldCM<BlsFr> {
        let mut m = DenseMatFieldCM::new(rows, cols);
        let mut data_cols = Vec::new();
        for _c in 0..cols { // column-major storage as in other tests
            let mut col = Vec::new();
            for _r in 0..rows {
                col.push(BlsFr::rand(rng));
            }
            data_cols.push(col);
        }
        m.set_data(data_cols);
        m
    }

    // Compute C = sum coeff[i] * A_i (element-wise)
    #[allow(dead_code)]
    fn linear_combination(mats: &Vec<DenseMatFieldCM<BlsFr>>, coeff: &Vec<BlsFr>) -> DenseMatFieldCM<BlsFr> {
        let rows = mats[0].shape.0; let cols = mats[0].shape.1;
        let mut c = DenseMatFieldCM::new(rows, cols);
        // initialize zero matrix data (column-major)
        let mut data = vec![vec![BlsFr::zero(); rows]; cols];
        for (mat, cf) in mats.iter().zip(coeff.iter()) {
            for col in 0..cols { for row in 0..rows { data[col][row] += mat.data[col][row] * *cf; }}
        }
        c.set_data(data); c
    }
}