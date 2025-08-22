//! Prove that two matrices b = phi(a)
//! This is done by using a lookup proof and two range proofs
//! 
//! ```text
//! 
//!     alpha <= a <= beta  &&  (alpha, beta, b) lie within the lookup table of phi
//! 
//! ```
//!     
//! showing that their projections on random vectors are equal.
//! 
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;
use mat::utils::matdef::DenseMatFieldCM;

use atomic_proof::atomic_protocol::{AtomicMatProtocol};
use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder; // remove unused ArithmeticExpression

use crate::utils::table_builder::{LOOKUPCONFIG8, LOOKUPCONFIG16};
use crate::utils::matop;

use super::lookup::LookUp; 
use super::range::RangeProof; // correct path
use crate::{MyInt, MyShortInt};
use atomic_proof::protocols::sub::MatSub;
use atomic_proof::protocols::batchpoint::BatchPoint;

#[derive(Debug, Clone)]
pub struct ActivationPoP<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub phi_input_hat: F,
    pub phi_output_hat: F,
    pub lookup_table_hats: Vec<F>,
    pub lookup_target_hats: Vec<F>,
    pub lookup_table_auxiliary_hat: F,
    pub lookup_target_auxiliary_hat: F,
    pub range_table_auxiliary_hat: Vec<F>,
    pub range_target_auxiliary_hat: Vec<F>,

    pub phi_input_point: (Vec<F>, Vec<F>),
    pub phi_output_point: (Vec<F>, Vec<F>),
    pub lookup_table_points: Vec<(Vec<F>, Vec<F>)>,
    pub lookup_target_points: Vec<(Vec<F>, Vec<F>)>,
    pub lookup_table_auxiliary_points: (Vec<F>, Vec<F>),
    pub lookup_target_auxiliary_points: (Vec<F>, Vec<F>),
    pub range_table_auxiliary_points: Vec<(Vec<F>, Vec<F>)>,
    pub range_target_auxiliary_points: Vec<(Vec<F>, Vec<F>)>,

    pub mapping: ActivationPoPMapping,
}
#[derive(Debug, Clone)]
pub struct ActivationPoPMapping {
    pub phi_input_hat_index: usize,
    pub phi_output_hat_index: usize,
    pub lookup_table_hats_index: Vec<usize>,
    pub lookup_target_hats_index: Vec<usize>,
    pub lookup_table_auxiliary_hat_index: usize,
    pub lookup_target_auxiliary_hat_index: usize,
    pub range_table_auxiliary_hat_index: Vec<usize>,
    pub range_target_auxiliary_hat_index: Vec<usize>,

    pub phi_input_point_index: (Vec<usize>, Vec<usize>),
    pub phi_output_point_index: (Vec<usize>, Vec<usize>),
    pub lookup_table_points_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub lookup_target_points_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub lookup_table_auxiliary_points_index: (Vec<usize>, Vec<usize>),
    pub lookup_target_auxiliary_points_index: (Vec<usize>, Vec<usize>),
    pub range_table_auxiliary_points_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub range_target_auxiliary_points_index: Vec<(Vec<usize>, Vec<usize>)>,
}

#[derive(Debug, Clone)]
pub struct Activation<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub k: usize,
    pub atomic_pop: ActivationPoP<F>,
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> ActivationPoP<F> {
    pub fn new() -> Self {
        Self {
            phi_input_hat: F::zero(),
            phi_output_hat: F::zero(),
            lookup_table_hats: Vec::new(),
            lookup_target_hats: Vec::new(),
            lookup_table_auxiliary_hat: F::zero(),
            lookup_target_auxiliary_hat: F::zero(),
            range_table_auxiliary_hat: Vec::new(),
            range_target_auxiliary_hat: Vec::new(),

            phi_input_point: (Vec::new(), Vec::new()),
            phi_output_point: (Vec::new(), Vec::new()),
            lookup_table_points: Vec::new(),
            lookup_target_points: Vec::new(),
            lookup_table_auxiliary_points: (Vec::new(), Vec::new()),
            lookup_target_auxiliary_points: (Vec::new(), Vec::new()),
            range_table_auxiliary_points: Vec::new(),
            range_target_auxiliary_points: Vec::new(),

            mapping: ActivationPoPMapping::new(),
        }
    }
} 

impl ActivationPoPMapping {

    pub fn new() -> Self {
        Self{
            phi_input_hat_index: 0,
            phi_output_hat_index: 0,
            lookup_table_hats_index: Vec::new(),
            lookup_target_hats_index: Vec::new(),
            lookup_table_auxiliary_hat_index: 0,
            lookup_target_auxiliary_hat_index: 0,
            range_table_auxiliary_hat_index: Vec::new(),
            range_target_auxiliary_hat_index: Vec::new(),

            phi_input_point_index: (Vec::new(), Vec::new()),
            phi_output_point_index: (Vec::new(), Vec::new()),
            lookup_table_points_index: Vec::new(),
            lookup_target_points_index: Vec::new(),
            lookup_table_auxiliary_points_index: (Vec::new(), Vec::new()),
            lookup_target_auxiliary_points_index: (Vec::new(), Vec::new()),
            range_table_auxiliary_points_index: Vec::new(),
            range_target_auxiliary_points_index: Vec::new(),
        }
    }
} 




impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> Activation<F> {
    pub fn new() -> Self {
    let k = if std::mem::size_of::<MyShortInt>() == 1 { LOOKUPCONFIG8.2 } else { LOOKUPCONFIG16.2 };
    Self { k, atomic_pop: ActivationPoP::new() }
    }

    pub fn reduce_prover_with_constraint_building(
        &mut self,
        trans: &mut Transcript<F>,
        cs_builder: &mut ConstraintSystemBuilder<F>,
        phi_input_vec: &Vec<MyInt>,          // already flattened input vector
        phi_output_vec: &Vec<MyInt>,         // gamma target vector now passed separately
        lookup_table: &Vec<Vec<MyInt>>,      // still contains (alpha, beta, gamma) table columns
        lookup_target: &Vec<Vec<MyInt>>,     // now only contains (alpha, beta) target columns
        lookup_table_auxiliary: &Vec<MyInt>,
        lookup_target_auxiliary: &Vec<MyInt>,
        range_table_auxiliary: &Vec<Vec<MyInt>>,
        range_target_auxiliary: &Vec<Vec<MyInt>>,
    ) -> bool {
        // Basic dimension checks
        let target_len = lookup_target_auxiliary.len();
        let table_len = lookup_table_auxiliary.len();
        if lookup_table.len() != 3 {
            panic!("lookup_table must contain 3 columns: alpha, beta, gamma");
        }
        if lookup_target.len() != 2 {
            panic!("lookup_target must contain only 2 columns: alpha, beta (gamma separated)");
        }
        if lookup_table[0].len() != table_len
            || lookup_table[1].len() != table_len
            || lookup_table[2].len() != table_len
            || lookup_target[0].len() != target_len
            || lookup_target[1].len() != target_len
            || phi_output_vec.len() != target_len
            || range_target_auxiliary.len() < 2
            || range_target_auxiliary[0].len() != target_len
            || range_target_auxiliary[1].len() != target_len
            || phi_input_vec.len() != target_len {
            panic!("Dimension mismatch in Activation Proof!");
        }

        // Convert input/output/lookup data to field elements
        let input_vec = matop::vec_myint_to_field::<F>(phi_input_vec);
        let gamma_target = matop::vec_myint_to_field::<F>(phi_output_vec);
        let alpha_target = matop::vec_myint_to_field::<F>(&lookup_target[0]);
        let beta_target = matop::vec_myint_to_field::<F>(&lookup_target[1]);
        let alpha_table = matop::vec_myint_to_field::<F>(&lookup_table[0]);
        let beta_table = matop::vec_myint_to_field::<F>(&lookup_table[1]);
        let gamma_table = matop::vec_myint_to_field::<F>(&lookup_table[2]);
        let target_auxiliary = matop::vec_myint_to_field::<F>(lookup_target_auxiliary);
        let table_auxiliary = matop::vec_myint_to_field::<F>(lookup_table_auxiliary);

        // Lookup protocol now assembles target columns (alpha, beta, gamma) using separated gamma vector
        let mut lookup_protocol = LookUp::<F>::new(3, target_len, table_len);
        lookup_protocol.set_input(
            vec![alpha_target.clone(), beta_target.clone(), gamma_target.clone()],
            vec![alpha_table, beta_table, gamma_table],
            target_auxiliary,
            table_auxiliary,
        );
        let flag_lookup = lookup_protocol.reduce_prover_with_constraint_building(trans, cs_builder);


        let alpha_target_hat = lookup_protocol.atomic_pop.target_hats[0];
        let beta_target_hat = lookup_protocol.atomic_pop.target_hats[1];
        let output_hat = lookup_protocol.atomic_pop.target_hats[2];
        let alpha_target_points = lookup_protocol.atomic_pop.target_points[0].clone();
        let beta_target_points = lookup_protocol.atomic_pop.target_points[1].clone();
        let output_points = lookup_protocol.atomic_pop.target_points[2].clone();
        let alpha_target_hat_index = lookup_protocol.atomic_pop.mapping.target_hats_index[0];
        let beta_target_hat_index = lookup_protocol.atomic_pop.mapping.target_hats_index[1];
        let output_hat_index = lookup_protocol.atomic_pop.mapping.target_hats_index[2];
        let alpha_target_points_index = lookup_protocol.atomic_pop.mapping.target_points_index[0].clone();
        let beta_target_points_index = lookup_protocol.atomic_pop.mapping.target_points_index[1].clone();
        let output_points_index = lookup_protocol.atomic_pop.mapping.target_points_index[2].clone();

        self.atomic_pop.lookup_table_hats = lookup_protocol.atomic_pop.table_hats.clone();
        self.atomic_pop.lookup_table_points = lookup_protocol.atomic_pop.table_points.clone();
        self.atomic_pop.mapping.lookup_table_hats_index = lookup_protocol.atomic_pop.mapping.table_hats_index.clone();
        self.atomic_pop.mapping.lookup_table_points_index = lookup_protocol.atomic_pop.mapping.table_points_index.clone();

        self.atomic_pop.lookup_target_auxiliary_hat = lookup_protocol.atomic_pop.auxiliary_target_hat;
        self.atomic_pop.lookup_target_auxiliary_points = lookup_protocol.atomic_pop.auxiliary_target_points.clone();
        self.atomic_pop.mapping.lookup_target_auxiliary_hat_index = lookup_protocol.atomic_pop.mapping.auxiliary_target_hat_index;
        self.atomic_pop.mapping.lookup_target_auxiliary_points_index = lookup_protocol.atomic_pop.mapping.auxiliary_target_points_index.clone();

        self.atomic_pop.lookup_table_auxiliary_hat = lookup_protocol.atomic_pop.auxiliary_table_hat;
        self.atomic_pop.lookup_table_auxiliary_points = lookup_protocol.atomic_pop.auxiliary_table_points.clone();
        self.atomic_pop.mapping.lookup_table_auxiliary_hat_index = lookup_protocol.atomic_pop.mapping.auxiliary_table_hat_index;
        self.atomic_pop.mapping.lookup_table_auxiliary_points_index = lookup_protocol.atomic_pop.mapping.auxiliary_table_points_index.clone();

        self.atomic_pop.phi_output_hat = output_hat;
        self.atomic_pop.phi_output_point = output_points.clone();
        self.atomic_pop.mapping.phi_output_hat_index = output_hat_index;
        self.atomic_pop.mapping.phi_output_point_index = output_points_index.clone();
    
        lookup_protocol.clear();

        // Begin range proof for diff1 = input - alpha
        let diff1: Vec<F> = input_vec.iter().zip(alpha_target.clone()).map(|(i,a)| *i - a).collect();
        let range_target_auxiliary_vec = matop::vec_myint_to_field(&range_target_auxiliary[0]);
        let range_table_auxiliary_vec = matop::vec_myint_to_field(&range_table_auxiliary[0]);

        let mut rangeproof1 = RangeProof::<F>::new(self.k, target_len);
        rangeproof1.set_input(diff1, range_target_auxiliary_vec, range_table_auxiliary_vec);
        let flag_range1 = rangeproof1.reduce_prover_with_constraint_building(trans, cs_builder);

        let diff1_hat = rangeproof1.atomic_pop.target_hat;
        let diff1_points = rangeproof1.atomic_pop.target_point.clone();
        let diff1_hat_index = rangeproof1.atomic_pop.mapping.target_hat_index;
        let diff1_points_index = rangeproof1.atomic_pop.mapping.target_point_index.clone();

        self.atomic_pop.range_target_auxiliary_hat.push(rangeproof1.atomic_pop.auxiliary_target_hat);
        self.atomic_pop.range_target_auxiliary_points.push(rangeproof1.atomic_pop.auxiliary_target_point.clone());
        self.atomic_pop.mapping.range_target_auxiliary_hat_index.push(rangeproof1.atomic_pop.mapping.auxiliary_target_hat_index);
        self.atomic_pop.mapping.range_target_auxiliary_points_index.push(rangeproof1.atomic_pop.mapping.auxiliary_target_point_index.clone());

        self.atomic_pop.range_table_auxiliary_hat.push(rangeproof1.atomic_pop.auxiliary_table_hat);
        self.atomic_pop.range_table_auxiliary_points.push(rangeproof1.atomic_pop.auxiliary_table_point.clone());
        self.atomic_pop.mapping.range_table_auxiliary_hat_index.push(rangeproof1.atomic_pop.mapping.auxiliary_table_hat_index);
        self.atomic_pop.mapping.range_table_auxiliary_points_index.push(rangeproof1.atomic_pop.mapping.auxiliary_table_point_index.clone());

        rangeproof1.clear();

        // Subtraction protocol for diff1
        let mut subtraction_protocol1 = MatSub::<F>::new(diff1_hat, diff1_points.clone(), diff1_hat_index, diff1_points_index.clone(), (target_len, 1), (target_len, 1), (target_len, 1));
        subtraction_protocol1.set_input(
            DenseMatFieldCM::from_data(vec![input_vec.clone()]),
            DenseMatFieldCM::from_data(vec![alpha_target.clone()])
        );
        let flag_sub1 = subtraction_protocol1.reduce_prover_with_constraint_building(trans, cs_builder);

        let (input_hat1, input_point1) = subtraction_protocol1.atomic_pop.get_a();
        let (input_hat1_index, input_point1_index) = subtraction_protocol1.atomic_pop.get_a_index();

        let (alpha_target_hat_prime, alpha_target_point_prime) = subtraction_protocol1.atomic_pop.get_b();
        let (alpha_target_hat_prime_index, alpha_target_point_prime_index) = subtraction_protocol1.atomic_pop.get_b_index();

        subtraction_protocol1.clear();

        // Batch alpha_target projections
        let mut batchpoint_protocol1 = BatchPoint::<F>::new(
            vec![alpha_target_hat, alpha_target_hat_prime],
            vec![alpha_target_points.clone(), alpha_target_point_prime.clone()],
            vec![alpha_target_hat_index, alpha_target_hat_prime_index],
            vec![alpha_target_points_index.clone(), alpha_target_point_prime_index.clone()],
        );

        batchpoint_protocol1.set_input(DenseMatFieldCM::from_data(vec![alpha_target.clone()]));
        let flag_batch1 = batchpoint_protocol1.reduce_prover(trans)
            && batchpoint_protocol1.prepare_atomic_pop()
            && batchpoint_protocol1.synthesize_atomic_pop_constraints(cs_builder);

        let (alpha_target_hat_final, alpha_target_point_final) = batchpoint_protocol1.atomic_pop.get_c();
        let (alpha_target_hat_final_index, alpha_target_point_final_index) = batchpoint_protocol1.atomic_pop.get_c_index();

        batchpoint_protocol1.clear();

        
        // Similar for diff2

        let diff2: Vec<F> = beta_target.iter().zip(input_vec.iter()).map(|(b,i)| *b - *i).collect();
        let range_target_auxiliary_vec = matop::vec_myint_to_field(&range_target_auxiliary[1]);
        let range_table_auxiliary_vec = matop::vec_myint_to_field(&range_table_auxiliary[1]);

        let mut rangeproof2 = RangeProof::<F>::new(self.k, target_len);
        rangeproof2.set_input(diff2, range_target_auxiliary_vec, range_table_auxiliary_vec);
        let flag_range2 = rangeproof2.reduce_prover_with_constraint_building(trans, cs_builder);

        let diff2_hat = rangeproof2.atomic_pop.target_hat;
        let diff2_points = rangeproof2.atomic_pop.target_point.clone();
        let diff2_hat_index = rangeproof2.atomic_pop.mapping.target_hat_index;
        let diff2_points_index = rangeproof2.atomic_pop.mapping.target_point_index.clone();

        self.atomic_pop.range_target_auxiliary_hat.push(rangeproof2.atomic_pop.auxiliary_target_hat);
        self.atomic_pop.range_target_auxiliary_points.push(rangeproof2.atomic_pop.auxiliary_target_point.clone());
        self.atomic_pop.mapping.range_target_auxiliary_hat_index.push(rangeproof2.atomic_pop.mapping.auxiliary_target_hat_index);
        self.atomic_pop.mapping.range_target_auxiliary_points_index.push(rangeproof2.atomic_pop.mapping.auxiliary_target_point_index.clone());

        self.atomic_pop.range_table_auxiliary_hat.push(rangeproof2.atomic_pop.auxiliary_table_hat);
        self.atomic_pop.range_table_auxiliary_points.push(rangeproof2.atomic_pop.auxiliary_table_point.clone());
        self.atomic_pop.mapping.range_table_auxiliary_hat_index.push(rangeproof2.atomic_pop.mapping.auxiliary_table_hat_index);
        self.atomic_pop.mapping.range_table_auxiliary_points_index.push(rangeproof2.atomic_pop.mapping.auxiliary_table_point_index.clone());

        rangeproof2.clear();

        // Subtraction protocol for diff2
        let mut subtraction_protocol2 = MatSub::<F>::new(diff2_hat, diff2_points.clone(), diff2_hat_index, diff2_points_index.clone(), (target_len, 1), (target_len, 1), (target_len, 1));
        subtraction_protocol2.set_input(
            DenseMatFieldCM::from_data(vec![beta_target.clone()]),
            DenseMatFieldCM::from_data(vec![input_vec.clone()])
        );
        let flag_sub2 = subtraction_protocol2.reduce_prover_with_constraint_building(trans, cs_builder);

        let (input_hat2, input_point2) = subtraction_protocol2.atomic_pop.get_b();
        let (input_hat2_index, input_point2_index) = subtraction_protocol2.atomic_pop.get_b_index();

        let (beta_target_hat_prime, beta_target_point_prime) = subtraction_protocol2.atomic_pop.get_a();
        let (beta_target_hat_prime_index, beta_target_point_prime_index) = subtraction_protocol2.atomic_pop.get_a_index();

        subtraction_protocol2.clear();

        // Batch two projections of alpha_target
        let mut batchpoint_protocol2 = BatchPoint::<F>::new(
            vec![beta_target_hat_prime, beta_target_hat],
            vec![beta_target_point_prime.clone(), beta_target_points.clone()],
            vec![beta_target_hat_prime_index, beta_target_hat_index],
            vec![beta_target_point_prime_index.clone(), beta_target_points_index.clone()],
        );

        batchpoint_protocol2.set_input(DenseMatFieldCM::from_data(vec![beta_target.clone()]));
        let flag_batch2 = batchpoint_protocol2.reduce_prover(trans)
            && batchpoint_protocol2.prepare_atomic_pop()
            && batchpoint_protocol2.synthesize_atomic_pop_constraints(cs_builder);

        let (beta_target_hat_final, beta_target_point_final) = batchpoint_protocol2.atomic_pop.get_c();
        let (beta_target_hat_final_index, beta_target_point_final_index) = batchpoint_protocol2.atomic_pop.get_c_index();

        self.atomic_pop.lookup_target_hats.push(alpha_target_hat_final);
        self.atomic_pop.lookup_target_hats.push(beta_target_hat_final);
        self.atomic_pop.lookup_target_points.push(alpha_target_point_final.clone());
        self.atomic_pop.lookup_target_points.push(beta_target_point_final.clone());
        self.atomic_pop.mapping.lookup_target_hats_index.push(alpha_target_hat_final_index);
        self.atomic_pop.mapping.lookup_target_hats_index.push(beta_target_hat_final_index);
        self.atomic_pop.mapping.lookup_target_points_index.push(alpha_target_point_final_index.clone());
        self.atomic_pop.mapping.lookup_target_points_index.push(beta_target_point_final_index.clone());

        batchpoint_protocol2.clear();

        // Batch input_vec projections
        let mut batchpoint_protocol_input = BatchPoint::<F>::new(
            vec![input_hat1, input_hat2],
            vec![input_point1.clone(), input_point2.clone()],
            vec![input_hat1_index, input_hat2_index],
            vec![input_point1_index.clone(), input_point2_index.clone()],
        );

        batchpoint_protocol_input.set_input(DenseMatFieldCM::from_data(vec![input_vec.clone()]));
        let flag_batch_input = batchpoint_protocol_input.reduce_prover(trans)
            && batchpoint_protocol_input.prepare_atomic_pop()
            && batchpoint_protocol_input.synthesize_atomic_pop_constraints(cs_builder);

        let (input_hat_final, input_point_final) = batchpoint_protocol_input.atomic_pop.get_c();
        let (input_hat_final_index, input_point_final_index) = batchpoint_protocol_input.atomic_pop.get_c_index();

        self.atomic_pop.phi_input_hat = input_hat_final;
        self.atomic_pop.phi_input_point = input_point_final.clone();
        self.atomic_pop.mapping.phi_input_hat_index = input_hat_final_index;
        self.atomic_pop.mapping.phi_input_point_index = input_point_final_index.clone();
        batchpoint_protocol_input.clear();

        flag_lookup && flag_range1 && flag_range2 && flag_sub1 && flag_sub2 && flag_batch1 && flag_batch2 && flag_batch_input
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use fsproof::helper_trans::Transcript;
    use crate::utils::{matop, table_builder};
    use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder;
    use ark_ff::Zero;

    // Helper: compute projection of a vector (flattened matrix) against (l,r) tensor point challenges
    fn project_vec<F: PrimeField>(vec: &Vec<F>, point: &(Vec<F>, Vec<F>)) -> F {
        // For a (m x n) flattened row-major matrix turned into 1D vec of length m*n
        let log_m = point.0.len();
        let log_n = point.1.len();
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        assert_eq!(vec.len(), m * n, "Vector length mismatch for projection");

        // Build xi vectors from challenges (same as BatchPoint/LiteBullet convention):
        use mat::utils::xi::xi_from_challenges;
        let xl = xi_from_challenges(&point.0.clone());
        let xr = xi_from_challenges(&point.1.clone());
        assert_eq!(xl.len(), m);
        assert_eq!(xr.len(), n);

        // Compute sum_{i,j} a_{i,j} * xl[i] * xr[j]
        let mut acc = F::zero();
        for i in 0..m { for j in 0..n { acc += vec[i * n + j] * xl[i] * xr[j]; } }
        acc
    }

    #[test]
    fn test_activation_reduce_projections() {
        // Small dimensions power-of-two for simplicity
        let n = 128usize; // rows
        let m = 128usize; // cols
        
        // Build random phi_input matrix and simulate lookup tables via activation table builder
        // We'll reuse table_builder ActivationTable to get (alpha, beta, gamma) logic
        // Generate a random input vector of MyInt within short range
        let bitwidth = std::mem::size_of::<MyShortInt>() * 8;
        let input_mat_myint: Vec<Vec<MyInt>> = {
            let mat_field = matop::gen_rand_matrix::<BlsFr>(n, m, bitwidth);
            mat_field.data.clone()
        };

        // Activation table & lookup targets
        let activation_tb = table_builder::ActivationTable::new();
        let (alpha_table, beta_table, gamma_table) = activation_tb.get_lookup_table();

        // Flatten phi_input
        let flattened_input: Vec<MyInt> = input_mat_myint.iter().flatten().cloned().collect();
        let (alpha_target, beta_target, gamma_target, auxiliary_target, auxiliary_table) = activation_tb.gen_lookup_inputs(&flattened_input);

        // Build lookup vectors
        let lookup_table = vec![alpha_table.clone(), beta_table.clone(), gamma_table.clone()];
        let lookup_target = vec![alpha_target.clone(), beta_target.clone()]; // gamma separated

        // Range auxiliaries
        let k = activation_tb.get_increment_bw();
        let diff1: Vec<MyInt> = flattened_input.iter().zip(alpha_target.iter()).map(|(x,a)| *x - *a).collect();
        let diff2: Vec<MyInt> = beta_target.iter().zip(flattened_input.iter()).map(|(b,x)| *b - *x).collect();
        let (range_target_auxiliary1, range_table_auxiliary1) = table_builder::range_auxiliary_builder(&diff1, k);
        let (range_target_auxiliary2, range_table_auxiliary2) = table_builder::range_auxiliary_builder(&diff2, k);

        // Prepare protocol
        let mut act = Activation::<BlsFr>::new();
        let mut trans = Transcript::new(BlsFr::zero());
        let mut cs_builder = ConstraintSystemBuilder::new();

        let flag = act.reduce_prover_with_constraint_building(
            &mut trans,
            &mut cs_builder,
            &flattened_input,
            &gamma_target, // phi_output_vec
            &lookup_table,
            &lookup_target,
            &auxiliary_table,
            &auxiliary_target,
            &vec![range_table_auxiliary1.clone(), range_table_auxiliary2.clone()],
            &vec![range_target_auxiliary1.clone(), range_target_auxiliary2.clone()],
        );
        assert!(flag, "Activation reduction failed");

        // Recompute projections for phi_input and alpha/beta targets, compare with hats
        // Convert MyInt vectors to field
        let input_vec_f: Vec<BlsFr> = flattened_input.iter().map(|x| BlsFr::from(*x)).collect();
        let alpha_vec_f: Vec<BlsFr> = alpha_target.iter().map(|x| BlsFr::from(*x)).collect();
        let beta_vec_f: Vec<BlsFr> = beta_target.iter().map(|x| BlsFr::from(*x)).collect();
        let gamma_vec_f: Vec<BlsFr> = gamma_target.iter().map(|x| BlsFr::from(*x)).collect();

        // phi_input
        let input_hat_expected = project_vec(&input_vec_f, &act.atomic_pop.phi_input_point);
        assert_eq!(act.atomic_pop.phi_input_hat, input_hat_expected, "phi_input hat mismatch");

        // alpha target
        let alpha_hat_expected = project_vec(&alpha_vec_f, &act.atomic_pop.lookup_target_points[0]);
        assert_eq!(act.atomic_pop.lookup_target_hats[0], alpha_hat_expected, "alpha target hat mismatch");

        // beta target
        let beta_hat_expected = project_vec(&beta_vec_f, &act.atomic_pop.lookup_target_points[1]);
        assert_eq!(act.atomic_pop.lookup_target_hats[1], beta_hat_expected, "beta target hat mismatch");

        // gamma (output)
        let gamma_hat_expected = project_vec(&gamma_vec_f, &act.atomic_pop.phi_output_point);
        assert_eq!(act.atomic_pop.phi_output_hat, gamma_hat_expected, "phi output hat mismatch");
        println!("Transcript Len: {}", trans.trans_seq.len());
        println!("Constraint count: {}", cs_builder.arithmetic_constraints.len());
    }

    #[test]
    fn test_activation_multiple_random_cases() {
        

        let cases = vec![(8usize, 8usize), (8usize, 4usize)]; // (n,m)
        for (n, m) in cases {
            // Generate random MyInt matrix by sampling small integers
            let bitwidth = std::mem::size_of::<MyShortInt>() * 8;
            let rand_mat_field = matop::gen_rand_matrix::<BlsFr>(n, m, bitwidth);
            let phi_input_mat: Vec<Vec<MyInt>> = rand_mat_field.data.clone();

            let activation_tb = table_builder::ActivationTable::new();
            let (alpha_table, beta_table, gamma_table) = activation_tb.get_lookup_table();
            let flattened_input: Vec<MyInt> = phi_input_mat.iter().flatten().cloned().collect();
            let (alpha_target, beta_target, gamma_target, auxiliary_target, auxiliary_table) = activation_tb.gen_lookup_inputs(&flattened_input);

            let lookup_table = vec![alpha_table.clone(), beta_table.clone(), gamma_table.clone()];
            let lookup_target = vec![alpha_target.clone(), beta_target.clone()];
            let k = activation_tb.get_increment_bw();
            let diff1: Vec<MyInt> = flattened_input.iter().zip(alpha_target.iter()).map(|(x,a)| *x - *a).collect();
            let diff2: Vec<MyInt> = beta_target.iter().zip(flattened_input.iter()).map(|(b,x)| *b - *x).collect();
            let (range_target_auxiliary1, range_table_auxiliary1) = table_builder::range_auxiliary_builder(&diff1, k);
            let (range_target_auxiliary2, range_table_auxiliary2) = table_builder::range_auxiliary_builder(&diff2, k);

            let mut act = Activation::<BlsFr>::new();
            let mut trans = Transcript::new(BlsFr::zero());
            let mut cs_builder = ConstraintSystemBuilder::new();
            let flag = act.reduce_prover_with_constraint_building(
                &mut trans,
                &mut cs_builder,
                &flattened_input,
                &gamma_target,
                &lookup_table,
                &lookup_target,
                &auxiliary_table,
                &auxiliary_target,
                &vec![range_table_auxiliary1.clone(), range_table_auxiliary2.clone()],
                &vec![range_target_auxiliary1.clone(), range_target_auxiliary2.clone()],
            );
            assert!(flag, "Activation reduction failed for size ({},{})", n, m);

            let input_vec_f: Vec<BlsFr> = flattened_input.iter().map(|x| BlsFr::from(*x)).collect();
            let alpha_vec_f: Vec<BlsFr> = alpha_target.iter().map(|x| BlsFr::from(*x)).collect();
            let beta_vec_f: Vec<BlsFr> = beta_target.iter().map(|x| BlsFr::from(*x)).collect();
            let gamma_vec_f: Vec<BlsFr> = gamma_target.iter().map(|x| BlsFr::from(*x)).collect();

            let input_hat_expected = project_vec(&input_vec_f, &act.atomic_pop.phi_input_point);
            assert_eq!(act.atomic_pop.phi_input_hat, input_hat_expected, "phi_input hat mismatch size ({},{})", n, m);
            let alpha_hat_expected = project_vec(&alpha_vec_f, &act.atomic_pop.lookup_target_points[0]);
            assert_eq!(act.atomic_pop.lookup_target_hats[0], alpha_hat_expected, "alpha hat mismatch size ({},{})", n, m);
            let beta_hat_expected = project_vec(&beta_vec_f, &act.atomic_pop.lookup_target_points[1]);
            assert_eq!(act.atomic_pop.lookup_target_hats[1], beta_hat_expected, "beta hat mismatch size ({},{})", n, m);
            let gamma_hat_expected = project_vec(&gamma_vec_f, &act.atomic_pop.phi_output_point);
            assert_eq!(act.atomic_pop.phi_output_hat, gamma_hat_expected, "phi_output hat mismatch size ({},{})", n, m);

            println!("[case n={}, m={}] constraint count: {}", n, m, cs_builder.arithmetic_constraints.len());
        }
    }
}