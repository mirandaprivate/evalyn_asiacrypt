/// Proof that each row of a target matrix
/// 
/// ```text
/// 
///     [vec(x)_1, vec(x)_2, ... ]
/// 
/// is looked up from a row in 
/// 
///     [vec(T)_1, vec(T)_2, ...]
/// 
/// 
/// This is proven by using two auxiliary vectors
/// 
///  random challenge z and
/// 
/// 
/// Here the methods for computing the grand products are slightly different
/// Define four grand products:
///
///     Theta1 = ∏_{i=1}^m (z * target-auxiliary_i + 1 + z^2 * x_i1 + z^3 * x_i2 + z^4 * x_i3 )
///     Theta2 = ∏_{j=1}^n (z * table-auxiliary_j + 1 + z^2 * T_j1 + z^3 * T_j2 + z^4 * T_j3)
///     Theta3 = ∏_{i=1}^m (z * ( target-auxiliary_i + 1) + 1 + z^2 * x_i1 + z^3 * x_i2 + z^4 * x_i3 )
///     Theta4 = ∏_{j=1}^n (1 + z^2 * T_j1 + z^3 * T_j2 + z^4 * T_j3)
///
/// For any challenge z, we have:
///     
/// 
///     Theta1 * Theta2 = Theta3 * Theta4
/// ```
/// 
/// 
// use ark_ec::pairing::{Pairing, PairingOutput}; // not needed yet
use ark_ff::{PrimeField, UniformRand};
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

// use ark_poly_commit::smart_pc::SmartPC; // placeholder (unused so far)
// use ark_poly_commit::smart_pc::data_structures::{Trans as PcsTrans, UniversalParams as PcsPP};


use atomic_proof::AtomicMatProtocol;
use atomic_proof::ArithmeticExpression;
use atomic_proof::{
    GrandProd,
    LinComb,
    BatchPoint,
};
use mat::xi;
use mat::utils::matdef::DenseMatFieldCM; // for GrandProd / LinComb inputs

use fsproof::Transcript;
#[allow(unused_imports)]
use rayon::prelude::*; // for parallel lincomb & grandprod

#[derive(Debug, Clone)]
pub struct LookupProtocolInput<F: PrimeField>
{
    pub num_col: usize,
    pub target_len: usize,
    pub table_len: usize,
    pub target: Vec<Vec<F>>,
    pub table: Vec<Vec<F>>,
    pub auxiliary_target: Vec<F>,
    pub auxiliary_table: Vec<F>,
    pub ready: bool,
}

#[derive(Debug, Clone)]
pub struct LookupMapping {
    pub z_index: usize,
    pub theta1_index: usize,
    pub theta2_index: usize,
    pub theta3_index: usize,
    pub theta4_index: usize,
    pub coeffs_index: Vec<usize>,
    pub target_hats_index: Vec<usize>,
    pub target_points_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub table_hats_index: Vec<usize>,
    pub table_points_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub auxiliary_target_hat_index: usize,
    pub auxiliary_target_points_index: (Vec<usize>, Vec<usize>),
    pub auxiliary_table_hat_index: usize,
    pub auxiliary_table_points_index: (Vec<usize>, Vec<usize>),
}

// (Removed placeholder LinearCombination/BatchPoint definitions; using real types from atomic_proof)

#[derive(Debug, Clone)]
pub struct LookupAtomicPoP<F: PrimeField> {
    pub z: F,
    pub theta1: F,
    pub theta2: F,
    pub theta3: F,
    pub theta4: F,
    pub coeffs: Vec<F>,
    pub target_hats: Vec<F>,
    pub target_points: Vec<(Vec<F>, Vec<F>)>,
    pub table_hats: Vec<F>,
    pub table_points: Vec<(Vec<F>, Vec<F>)>,
    pub auxiliary_target_hat: F,
    pub auxiliary_target_points: (Vec<F>, Vec<F>),
    pub auxiliary_table_hat: F,
    pub auxiliary_table_points: (Vec<F>, Vec<F>),
    // The index in the trans_reduce
    pub mapping: LookupMapping,
    // check
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    pub ready: (bool, bool, bool),
}

#[derive(Debug, Clone)]
pub struct LookUp<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub protocol_input: LookupProtocolInput<F>,
    pub atomic_pop: LookupAtomicPoP<F>,
    pub grandprod1: GrandProd<F>,
    pub grandprod2: GrandProd<F>,
    pub grandprod3: GrandProd<F>,
    pub grandprod4: GrandProd<F>,
    pub lincomb1: LinComb<F>,
    pub lincomb2: LinComb<F>,
    pub lincomb3: LinComb<F>,
    pub lincomb4: LinComb<F>,
    pub batchpoint_target: Vec<BatchPoint<F>>,
    pub batchpoint_table: Vec<BatchPoint<F>>,
    pub batchpoint_target_auxiliary: BatchPoint<F>,
    pub batchpoint_table_auxiliary: BatchPoint<F>,
}

impl<F: PrimeField> LookupProtocolInput<F> {
    pub fn new(num_col: usize, target_len: usize, table_len: usize) -> Self {
        Self {
            num_col,
            target_len,
            table_len,
            target: Vec::new(),
            table: Vec::new(),
            auxiliary_target: Vec::new(),
            auxiliary_table: Vec::new(),
            ready: false,
        }
    }

    pub fn set_input(
        &mut self,
        target: Vec<Vec<F>>,
        table: Vec<Vec<F>>,
        auxiliary_target: Vec<F>,
        auxiliary_table: Vec<F>,
    ) {


        if target.len() != self.num_col || table.len() != self.num_col
        || target[0].len() != self.target_len || table[0].len() != self.table_len
        || auxiliary_target.len() != self.target_len || auxiliary_table.len() != self.table_len {
            panic!("Inconsistent input shape in BatchPoint");
        }

        self.target = target;
        self.table = table;
        self.auxiliary_target = auxiliary_target;
        self.auxiliary_table = auxiliary_table;
        self.ready = true;
    }

    pub fn clear(&mut self) {
        self.target = Vec::new();
        self.table = Vec::new();
        self.auxiliary_target = Vec::new();
        self.auxiliary_table = Vec::new();
        self.ready = false;
    }
}

impl<F: PrimeField> LookupAtomicPoP<F> {
    pub fn new() -> Self {
        Self {
            z: F::zero(),
            theta1: F::zero(),
            theta2: F::zero(),
            theta3: F::zero(),
            theta4: F::zero(),
            coeffs: Vec::new(),
            target_hats: Vec::new(),
            target_points: Vec::new(),
            table_hats: Vec::new(),
            table_points: Vec::new(),
            auxiliary_target_hat: F::zero(),
            auxiliary_target_points: (Vec::new(), Vec::new()),
            auxiliary_table_hat: F::zero(),
            auxiliary_table_points: (Vec::new(), Vec::new()),
            mapping: LookupMapping {
                z_index: 0,
                theta1_index: 0,
                theta2_index: 0,
                theta3_index: 0,
                theta4_index: 0,
                coeffs_index: Vec::new(),
                target_hats_index: Vec::new(),
                target_points_index: Vec::new(),
                table_hats_index: Vec::new(),
                table_points_index: Vec::new(),
                auxiliary_target_hat_index: 0,
                auxiliary_target_points_index: (Vec::new(), Vec::new()),
                auxiliary_table_hat_index: 0,
                auxiliary_table_points_index: (Vec::new(), Vec::new()),
            },
            check: ArithmeticExpression::constant(F::zero()),
            link_inputs: Vec::new(),
            ready: (false, false, false),
        }
    }


    pub fn set_pop_trans(
        &mut self,
        z: F,
        theta1: F,
        theta2: F,
        theta3: F,
        theta4: F,
        coeffs: Vec<F>,
        target_hats: Vec<F>,
        target_points: Vec<(Vec<F>, Vec<F>)>,
        table_hats: Vec<F>,
        table_points: Vec<(Vec<F>, Vec<F>)>,
        auxiliary_target_hat: F,
        auxiliary_target_points: (Vec<F>, Vec<F>),
        auxiliary_table_hat: F,
        auxiliary_table_points: (Vec<F>, Vec<F>),
        z_index: usize,
        theta1_index: usize,
        theta2_index: usize,
        theta3_index: usize,
        theta4_index: usize,
        coeffs_index: Vec<usize>,
        target_hats_index: Vec<usize>,
        target_points_index: Vec<(Vec<usize>, Vec<usize>)>,
        table_hats_index: Vec<usize>,
        table_points_index: Vec<(Vec<usize>, Vec<usize>)>,
        auxiliary_target_hat_index: usize,
        auxiliary_target_points_index: (Vec<usize>, Vec<usize>),
        auxiliary_table_hat_index: usize,
        auxiliary_table_points_index: (Vec<usize>, Vec<usize>),
    )
    {
        self.z = z;
        self.theta1 = theta1;
        self.theta2 = theta2;
        self.theta3 = theta3;
        self.theta4 = theta4;
        self.coeffs = coeffs;
        self.target_hats = target_hats;
        self.target_points = target_points;
        self.table_hats = table_hats;
        self.table_points = table_points;
        self.auxiliary_target_hat = auxiliary_target_hat;
        self.auxiliary_target_points = auxiliary_target_points;
        self.auxiliary_table_hat = auxiliary_table_hat;
        self.auxiliary_table_points = auxiliary_table_points;
        // Update mapping indices
        self.mapping.z_index = z_index;
        self.mapping.theta1_index = theta1_index;
        self.mapping.theta2_index = theta2_index;
        self.mapping.theta3_index = theta3_index;
        self.mapping.theta4_index = theta4_index;
        self.mapping.coeffs_index = coeffs_index;
        self.mapping.target_hats_index = target_hats_index;
        self.mapping.target_points_index = target_points_index;
        self.mapping.table_hats_index = table_hats_index;
        self.mapping.table_points_index = table_points_index;
        self.mapping.auxiliary_target_hat_index = auxiliary_target_hat_index;
        self.mapping.auxiliary_target_points_index = auxiliary_target_points_index;
        self.mapping.auxiliary_table_hat_index = auxiliary_table_hat_index;
        self.mapping.auxiliary_table_points_index = auxiliary_table_points_index;

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



impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> LookUp<F> {
    pub fn new(num_col: usize, target_len: usize, table_len: usize) -> Self {
        
        Self {
            protocol_input: LookupProtocolInput::new(num_col, target_len, table_len),
            atomic_pop: LookupAtomicPoP::new(),
            grandprod1: GrandProd::default(),
            grandprod2: GrandProd::default(),
            grandprod3: GrandProd::default(),
            grandprod4: GrandProd::default(),
            lincomb1: LinComb::default(),
            lincomb2: LinComb::default(),
            lincomb3: LinComb::default(),
            lincomb4: LinComb::default(),
            batchpoint_target: Vec::new(),
            batchpoint_table: Vec::new(),
            batchpoint_target_auxiliary: BatchPoint::default(),
            batchpoint_table_auxiliary: BatchPoint::default(),
        }
    }

    pub fn default() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn set_input(
        &mut self,
        target: Vec<Vec<F>>,
        table: Vec<Vec<F>>,
        auxiliary_target: Vec<F>,
        auxiliary_table: Vec<F>,
    ) {
        self.protocol_input.set_input(target, table, auxiliary_target, auxiliary_table);
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for LookUp<F> {
    fn clear(&mut self) {
        self.protocol_input.clear();
        self.grandprod1.clear();
        self.grandprod2.clear();
        self.grandprod3.clear();
        self.grandprod4.clear();
        self.lincomb1.clear();
        self.lincomb2.clear();
        self.lincomb3.clear();
        self.lincomb4.clear();
        for bp in &mut self.batchpoint_target { bp.clear(); }
        for bp in &mut self.batchpoint_table { bp.clear(); }
        self.batchpoint_target_auxiliary.clear();
        self.batchpoint_table_auxiliary.clear();
    }

    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {

        let z_index = trans.pointer; // keep for potential mapping usage
        let z = trans.gen_challenge();

        let z_power_trail: Vec<F> = (0..self.protocol_input.num_col).map(|i| z.pow([ (i as u64) + 2 ])).collect();

        let target_len = self.protocol_input.target_len;
        let table_len = self.protocol_input.table_len;
        let num_col = self.protocol_input.num_col;

        // Here the methods for computing the grand products are slightly different
        // Define four grand products:
        // ```text
        //     Theta1 = ∏_{i=1}^m (z * target-auxiliary_i + 1 + z^2 * x_i1 + z^3 * x_i2 + z^4 * x_i3 )
        //     Theta2 = ∏_{j=1}^n (z * table-auxiliary_j + 1 + z^2 * T_j1 + z^3 * T_j2 + z^4 * T_j3)
        //     Theta3 = ∏_{i=1}^m (z * ( target-auxiliary_i + 1) + 1 + z^2 * x_i1 + z^3 * x_i2 + z^4 * x_i3 )
        //     Theta4 = ∏_{j=1}^n (                                1 + z^2 * T_j1 + z^3 * T_j2 + z^4 * T_j3)
        //
        // For any challenge z, we have:
        //     Theta1 * Theta2 = Theta3 * Theta4
        //```
        // Compute the linear comb vectors for computing Theta1, Theta2, Theta3, Theta4
        let mut target_vec_combined = Vec::new();
        target_vec_combined.push(vec![F::one(); self.protocol_input.target_len]);
        target_vec_combined.push(self.protocol_input.auxiliary_target.clone());
        target_vec_combined.extend(self.protocol_input.target.iter().cloned());

        let mut theta1_coeff = vec![F::one(), z];
        theta1_coeff.extend(z_power_trail.iter().cloned());
        let theta1_vec = lincomb(&target_vec_combined, &theta1_coeff);
        let theta1 = grandprod(&theta1_vec);

        let mut theta3_coeff = vec![z + F::one(), z];
        theta3_coeff.extend(z_power_trail.iter().cloned());
        let theta3_vec = lincomb(&target_vec_combined, &theta3_coeff);
        let theta3 = grandprod(&theta3_vec);


        let mut table_vec_combined = Vec::new();
        table_vec_combined.push(vec![F::one(); self.protocol_input.table_len]);
        table_vec_combined.push(self.protocol_input.auxiliary_table.clone());
        table_vec_combined.extend(self.protocol_input.table.iter().cloned());

        let mut theta2_coeff = vec![F::one(), z];
        theta2_coeff.extend(z_power_trail.iter().cloned());
        let theta2_vec = lincomb(&table_vec_combined, &theta2_coeff);
        let theta2 = grandprod(&theta2_vec);

        let mut theta4_coeff = vec![F::one(), F::zero()];
        theta4_coeff.extend(z_power_trail.iter().cloned());
        let theta4_vec = lincomb(&table_vec_combined, &theta4_coeff);
        let theta4 = grandprod(&theta4_vec);

        // push responses so verifier can read
        let theta1_index = trans.pointer;
        trans.push_response(theta1);
        let theta2_index = trans.pointer;
        trans.push_response(theta2);
        let theta3_index = trans.pointer;
        trans.push_response(theta3);
        let theta4_index = trans.pointer;
        trans.push_response(theta4);
        // simple runtime assertion
        assert_eq!(theta1 * theta2, theta3 * theta4, "Theta relation broken in reduce_prover");

        // =======================Below, we reduce these to verifying projections of target, table, target_auxiliary, table auxiliary=====
        // We are going to fill in the vectors below
        let mut target_hats = Vec::new();
        let mut table_hats = Vec::new();
        let mut target_auxiliary_hat: F = F::zero();
        let mut table_auxiliary_hat: F = F::zero();
        let mut target_points = Vec::new();
        let mut table_points = Vec::new();
        let mut target_auxiliary_points: (Vec::<F>, Vec::<F>) = (Vec::new(), Vec::new());
        let mut table_auxiliary_points: (Vec::<F>, Vec::<F>) = (Vec::new(), Vec::new());
        let mut target_hats_index = Vec::new();
        let mut table_hats_index = Vec::new();
        let mut target_auxiliary_hat_index: usize = 0;
        let mut table_auxiliary_hat_index: usize = 0;
        let mut target_points_index = Vec::new();
        let mut table_points_index = Vec::new();
        let mut target_auxiliary_points_index: (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut table_auxiliary_points_index: (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        // =================================================================================================================
        
        // =================================================================================================================
        // push [0, 1, z, 1+z, z^2, z^3, ...] to the transcript such that it can be read by the verifier
        let mut coeffs_index = Vec::new();
        let mut coeffs = vec![F::zero(), F::one(), z, z + F::one()];

        for i in 0..coeffs.len() {
            coeffs_index.push(trans.pointer);
            trans.push_response(coeffs[i].clone());
        }

        let mut z_power = z * z; // z^2
        for _ in 0..self.protocol_input.num_col { // push z^2 .. z^{num_col+1}
            coeffs_index.push(trans.pointer);
            trans.push_response(z_power);
            coeffs.push(z_power);
            z_power *= z;
        }

        // Build coefficient index vectors (without misusing extend)
        let mut theta1_coeff_index = Vec::new();
        theta1_coeff_index.push(coeffs_index[1]); // 1
        theta1_coeff_index.push(coeffs_index[2]); // z
        theta1_coeff_index.extend(coeffs_index[4..].iter().cloned()); // z^2 ...

        let mut theta2_coeff_index = Vec::new();
        theta2_coeff_index.push(coeffs_index[1]); // 1
        theta2_coeff_index.push(coeffs_index[2]); // z
        theta2_coeff_index.extend(coeffs_index[4..].iter().cloned());

        let mut theta3_coeff_index = Vec::new();
        theta3_coeff_index.push(coeffs_index[3]); // 1+z
        theta3_coeff_index.push(coeffs_index[2]); // z
        theta3_coeff_index.extend(coeffs_index[4..].iter().cloned());

        let mut theta4_coeff_index = Vec::new();
        theta4_coeff_index.push(coeffs_index[1]); // 1
        theta4_coeff_index.push(coeffs_index[0]); // 0
        theta4_coeff_index.extend(coeffs_index[4..].iter().cloned());
        // ===================================================================================================================

        // ===================================================================================================================
        // GrandProd + LinComb chain for Theta1 (store in self.grandprod1 / self.lincomb1; don't clear yet)
        self.grandprod1 = GrandProd::new(theta1, theta1_index, target_len);
        self.grandprod1.set_input(DenseMatFieldCM::from_data(vec![theta1_vec.clone()]));
        let flag11 = self.grandprod1.reduce_prover(trans);
        assert!(flag11, "GrandProd protocol 1 failed");

        let (theta1_vec_hat, theta1_vec_point) = self.grandprod1.atomic_pop.get_a();
        let (theta1_vec_hat_index, theta1_vec_point_index) = self.grandprod1.atomic_pop.get_a_index();

        self.lincomb1 = LinComb::new(
            theta1_vec_hat,
            theta1_vec_point.clone(),
            theta1_vec_hat_index,
            theta1_vec_point_index.clone(),
            (target_len, 1),
            num_col + 2,
            theta1_coeff.clone(),
            theta1_coeff_index.clone(),
        );
        let mut lincomb_input1: Vec<DenseMatFieldCM<F>> = Vec::new();
        for v in target_vec_combined.iter() { lincomb_input1.push(DenseMatFieldCM::from_data(vec![v.clone()])); }
        self.lincomb1.set_input(lincomb_input1);
        let flag12 = self.lincomb1.reduce_prover(trans);
        assert!(flag12, "LinComb protocol 1 failed");

        // Get the projections of the input matrices of Theta1
        let (target_combined_hats1, target_combined_points1) = self.lincomb1.atomic_pop.get_inputs();
        let (target_combined_hats_index1, target_combined_points_index1) = self.lincomb1.atomic_pop.get_inputs_index();

        let flag1 = flag11 && flag12;

        // -------------------------------------------------------------------------------------------------------------

        // GrandProd + LinComb chain for Theta3 (store in self.grandprod3 / self.lincomb3)
        self.grandprod3 = GrandProd::new(theta3, theta3_index, target_len);
        self.grandprod3.set_input(DenseMatFieldCM::from_data(vec![theta3_vec.clone()]));
        let flag31 = self.grandprod3.reduce_prover(trans);
        assert!(flag31, "GrandProd protocol 3 failed");

        let (theta3_vec_hat, theta3_vec_point) = self.grandprod3.atomic_pop.get_a();
        let (theta3_vec_hat_index, theta3_vec_point_index) = self.grandprod3.atomic_pop.get_a_index();

        self.lincomb3 = LinComb::new(
            theta3_vec_hat,
            theta3_vec_point.clone(),
            theta3_vec_hat_index,
            theta3_vec_point_index.clone(),
            (target_len, 1),
            num_col + 2,
            theta3_coeff.clone(),
            theta3_coeff_index.clone(),
        );
        let mut lincomb_input3: Vec<DenseMatFieldCM<F>> = Vec::new();
        for v in target_vec_combined.iter() { lincomb_input3.push(DenseMatFieldCM::from_data(vec![v.clone()])); }
        self.lincomb3.set_input(lincomb_input3);
        let flag32 = self.lincomb3.reduce_prover(trans);
        assert!(flag32, "LinComb protocol 3 failed");

        // Get the projections of the input matrices of Theta3
        let (target_combined_hats3, target_combined_points3) = self.lincomb3.atomic_pop.get_inputs();
        let (target_combined_hats_index3, target_combined_points_index3) = self.lincomb3.atomic_pop.get_inputs_index();

        let flag3 = flag31 && flag32;

        // -------------------------------------------------------------------------------------------------------------

        // Batch the projections of target_combined_hat1 and target_combined_hats3

        let mut flag_target = flag1 && flag3;

        for i in 0..(num_col + 2) {
            let mut bp = BatchPoint::<F>::new(
                vec![target_combined_hats1[i].clone(), target_combined_hats3[i]],
                vec![target_combined_points1[i].clone(), target_combined_points3[i].clone()],
                vec![target_combined_hats_index1[i].clone(), target_combined_hats_index3[i].clone()],
                vec![target_combined_points_index1[i].clone(), target_combined_points_index3[i].clone()],
            );

            bp.set_input(DenseMatFieldCM::from_data(vec![target_vec_combined[i].clone()]));
            flag_target = bp.reduce_prover(trans) && flag_target;

            let (hat, point) = bp.atomic_pop.get_c();
            let (hat_index, point_index) = bp.atomic_pop.get_c_index();

            if i == 1 {
                target_auxiliary_hat = hat;
                target_auxiliary_points = point;
                target_auxiliary_hat_index = hat_index;
                target_auxiliary_points_index = point_index;
                self.batchpoint_target_auxiliary = bp;
            } else if i == 0 {
                assert_eq!(
                    hat,
                    xi::xi_ip_from_challenges(&point.0, &vec![F::one(); point.0.len()]),
                    "!! Vector 1 projection check failed in LookUp"
                );
            } else {
                target_hats.push(hat);
                target_points.push(point);
                target_hats_index.push(hat_index);
                target_points_index.push(point_index);
                self.batchpoint_target.push(bp);
            }
        } 
        
        // ===================================================================================================================
        // === We are going to prove Theta2 and Theta4 similarly
        // -------------------------------------------------------------------------------------------------------------------
        // GrandProd + LinComb chain for Theta2 (store in self.grandprod2 / self.lincomb2)
        self.grandprod2 = GrandProd::new(theta2, theta2_index, table_len);
        self.grandprod2.set_input(DenseMatFieldCM::from_data(vec![theta2_vec.clone()]));
        let flag21 = self.grandprod2.reduce_prover(trans);
        assert!(flag21, "GrandProd protocol 2 failed");

        let (theta2_vec_hat, theta2_vec_point) = self.grandprod2.atomic_pop.get_a();
        let (theta2_vec_hat_index, theta2_vec_point_index) = self.grandprod2.atomic_pop.get_a_index();

        self.lincomb2 = LinComb::new(
            theta2_vec_hat,
            theta2_vec_point.clone(),
            theta2_vec_hat_index,
            theta2_vec_point_index.clone(),
            (table_len, 1), // shape matches table vectors
            num_col + 2,
            theta2_coeff.clone(),
            theta2_coeff_index.clone(),
        );
        let mut lincomb_input2: Vec<DenseMatFieldCM<F>> = Vec::new();
        for v in table_vec_combined.iter() { lincomb_input2.push(DenseMatFieldCM::from_data(vec![v.clone()])); }
        self.lincomb2.set_input(lincomb_input2);
        let flag22 = self.lincomb2.reduce_prover(trans);
        assert!(flag22, "LinComb protocol 2 failed");

        // Get the projections of the input matrices of Theta2
        let (table_combined_hats2, table_combined_points2) = self.lincomb2.atomic_pop.get_inputs();
        let (table_combined_hats_index2, table_combined_points_index2) = self.lincomb2.atomic_pop.get_inputs_index();

        let flag2 = flag21 && flag22;

        // -------------------------------------------------------------------------------------------------------------

        // GrandProd + LinComb chain for Theta4 (store in self.grandprod4 / self.lincomb4)
        self.grandprod4 = GrandProd::new(theta4, theta4_index, table_len);
        self.grandprod4.set_input(DenseMatFieldCM::from_data(vec![theta4_vec.clone()]));
        let flag41 = self.grandprod4.reduce_prover(trans);
        assert!(flag41, "GrandProd protocol 4 failed");

        let (theta4_vec_hat, theta4_vec_point) = self.grandprod4.atomic_pop.get_a();
        let (theta4_vec_hat_index, theta4_vec_point_index) = self.grandprod4.atomic_pop.get_a_index();

        self.lincomb4 = LinComb::new(
            theta4_vec_hat,
            theta4_vec_point.clone(),
            theta4_vec_hat_index,
            theta4_vec_point_index.clone(),
            (table_len, 1),
            num_col + 2,
            theta4_coeff.clone(),
            theta4_coeff_index.clone(),
        );
        let mut lincomb_input4: Vec<DenseMatFieldCM<F>> = Vec::new();
        for v in table_vec_combined.iter() { lincomb_input4.push(DenseMatFieldCM::from_data(vec![v.clone()])); }
        self.lincomb4.set_input(lincomb_input4);
        let flag42 = self.lincomb4.reduce_prover(trans);
        assert!(flag42, "LinComb protocol 4 failed");

        // Get the projections of the input matrices of Theta4
        let (table_combined_hats4, table_combined_points4) = self.lincomb4.atomic_pop.get_inputs();
        let (table_combined_hats_index4, table_combined_points_index4) = self.lincomb4.atomic_pop.get_inputs_index();

        let flag4 = flag41 && flag42;

        // -------------------------------------------------------------------------------------------------------------

        // Batch the projections of target_combined_hat2 and target_combined_hats4

        let mut flag_table = flag2 && flag4;

        for i in 0..(num_col + 2) {
            let mut bp = BatchPoint::<F>::new(
                vec![table_combined_hats2[i].clone(), table_combined_hats4[i]],
                vec![table_combined_points2[i].clone(), table_combined_points4[i].clone()],
                vec![table_combined_hats_index2[i].clone(), table_combined_hats_index4[i].clone()],
                vec![table_combined_points_index2[i].clone(), table_combined_points_index4[i].clone()],
            );

            bp.set_input(DenseMatFieldCM::from_data(vec![table_vec_combined[i].clone()]));
            flag_table = bp.reduce_prover(trans) && flag_table;

            let (hat, point) = bp.atomic_pop.get_c();
            let (hat_index, point_index) = bp.atomic_pop.get_c_index();

            if i == 1 {
                table_auxiliary_hat = hat;
                table_auxiliary_points = point;
                table_auxiliary_hat_index = hat_index;
                table_auxiliary_points_index = point_index;
                self.batchpoint_table_auxiliary = bp;
            } else if i == 0 {
                assert_eq!(
                    hat,
                    xi::xi_ip_from_challenges(&point.0, &vec![F::one(); point.0.len()]),
                    "!! Vector 1 projection check failed in LookUp"
                );
            } else {
                table_hats.push(hat);
                table_points.push(point);
                table_hats_index.push(hat_index);
                table_points_index.push(point_index);
                self.batchpoint_table.push(bp);
            }
        }

        // ===================================================================================================================



        // Record transcript data into atomic_pop
        self.atomic_pop.set_pop_trans(
            z, theta1, theta2, theta3, theta4,
            coeffs.clone(),
            target_hats, target_points, table_hats, table_points,
            target_auxiliary_hat, target_auxiliary_points,
            table_auxiliary_hat, table_auxiliary_points,
            z_index, theta1_index, theta2_index, theta3_index, theta4_index,
            coeffs_index.clone(),
            target_hats_index, target_points_index, table_hats_index, table_points_index,
            target_auxiliary_hat_index, target_auxiliary_points_index,
            table_auxiliary_hat_index, table_auxiliary_points_index,
        );

        println!("✅ LookUp reduce_prover completed successfully");

        true

    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        if !self.atomic_pop.ready.0 {
            panic!("AtomicPoP of Lookup is not ready! Run reduce_prover first");
        }

        let target_len = self.protocol_input.target_len;
        let table_len = self.protocol_input.table_len;
        let num_col = self.protocol_input.num_col;
        
        // read z challenge
        let z_index = trans.pointer;
        let z = trans.get_at_position(z_index);
        trans.pointer += 1; // challenge

        let theta1_index = trans.pointer;
        let theta1 = trans.get_at_position(theta1_index);
        trans.pointer += 1;

        let theta2_index = trans.pointer;
        let theta2 = trans.get_at_position(theta2_index);
        trans.pointer += 1;

        let theta3_index = trans.pointer;
        let theta3 = trans.get_at_position(theta3_index);
        trans.pointer += 1;

        let theta4_index = trans.pointer;
        let theta4 = trans.get_at_position(theta4_index);
        trans.pointer += 1;

        // Theta identity check (boolean): theta1 * theta2 == theta3 * theta4
        let flag_theta = theta1 * theta2 == theta3 * theta4;

        // =======================Below, we reduce these to verifying projections of target, table, target_auxiliary, table auxiliary=====
        // We are going to fill in the vectors below
        let mut target_hats = Vec::new();
        let mut table_hats = Vec::new();
        let mut target_auxiliary_hat: F = F::zero();
        let mut table_auxiliary_hat: F = F::zero();
        let mut target_points = Vec::new();
        let mut table_points = Vec::new();
        let mut target_auxiliary_points: (Vec::<F>, Vec::<F>) = (Vec::new(), Vec::new());
        let mut table_auxiliary_points: (Vec::<F>, Vec::<F>) = (Vec::new(), Vec::new());
        let mut target_hats_index = Vec::new();
        let mut table_hats_index = Vec::new();
        let mut target_auxiliary_hat_index: usize = 0;
        let mut table_auxiliary_hat_index: usize = 0;
        let mut target_points_index = Vec::new();
        let mut table_points_index = Vec::new();
        let mut target_auxiliary_points_index: (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut table_auxiliary_points_index: (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        // =================================================================================================================
        
        // =================================================================================================================
        // push [0, 1, z, 1+z, z^2, z^3, ...] to the transcript such that it can be read by the verifier
        let mut flag_coeffs = true;

        let mut coeffs_index = Vec::new();
        let mut coeffs = vec![F::zero(), F::one(), z, z + F::one()];

        for i in 0..coeffs.len() {
            coeffs_index.push(trans.pointer);
            let coeff_in_trans = trans.get_at_position(trans.pointer);
            trans.pointer += 1;
            
            flag_coeffs &= coeff_in_trans == coeffs[i];
        }

        let mut z_power = z * z; // z^2
        for _ in 0..self.protocol_input.num_col { // push z^2 .. z^{num_col+1}
            coeffs_index.push(trans.pointer);
            let coeff_in_trans = trans.get_at_position(trans.pointer);
            trans.pointer += 1;
            
            flag_coeffs &= coeff_in_trans == z_power;

            coeffs.push(coeff_in_trans);

            z_power *= z;
        }

        // Build coefficient index vectors (without misusing extend)
        let mut theta1_coeff_index = Vec::new();
        theta1_coeff_index.push(coeffs_index[1]); // 1
        theta1_coeff_index.push(coeffs_index[2]); // z
        theta1_coeff_index.extend(coeffs_index[4..].iter().cloned()); // z^2 ...

        let mut theta2_coeff_index = Vec::new();
        theta2_coeff_index.push(coeffs_index[1]); // 1
        theta2_coeff_index.push(coeffs_index[2]); // z
        theta2_coeff_index.extend(coeffs_index[4..].iter().cloned());

        let mut theta3_coeff_index = Vec::new();
        theta3_coeff_index.push(coeffs_index[3]); // 1+z
        theta3_coeff_index.push(coeffs_index[2]); // z
        theta3_coeff_index.extend(coeffs_index[4..].iter().cloned());

        let mut theta4_coeff_index = Vec::new();
        theta4_coeff_index.push(coeffs_index[1]); // 1
        theta4_coeff_index.push(coeffs_index[0]); // 0
        theta4_coeff_index.extend(coeffs_index[4..].iter().cloned());

        // Reconstruct coefficient value vectors from transcript values (matching reduce_prover)
        let mut theta1_coeff: Vec<F> = Vec::with_capacity(num_col + 2);
        theta1_coeff.push(coeffs[1]); // 1
        theta1_coeff.push(coeffs[2]); // z
        theta1_coeff.extend(coeffs[4..].iter().cloned()); // z^2..

        let mut theta2_coeff: Vec<F> = Vec::with_capacity(num_col + 2);
        theta2_coeff.push(coeffs[1]); // 1
        theta2_coeff.push(coeffs[2]); // z
        theta2_coeff.extend(coeffs[4..].iter().cloned());

        let mut theta3_coeff: Vec<F> = Vec::with_capacity(num_col + 2);
        theta3_coeff.push(coeffs[3]); // 1+z
        theta3_coeff.push(coeffs[2]); // z
        theta3_coeff.extend(coeffs[4..].iter().cloned());

        let mut theta4_coeff: Vec<F> = Vec::with_capacity(num_col + 2);
        theta4_coeff.push(coeffs[1]); // 1
        theta4_coeff.push(coeffs[0]); // 0
        theta4_coeff.extend(coeffs[4..].iter().cloned());
        // ===================================================================================================================

        // ===================================================================================================================
        // GrandProd + LinComb chain for Theta1 (fix accessor usage & input construction)
        let mut grandprod_protocol1 = GrandProd::new(theta1, theta1_index, target_len);
       
        let flag11 = grandprod_protocol1.verify_as_subprotocol(trans);
        assert!(flag11, "GrandProd protocol 1 failed");

        let (theta1_vec_hat, theta1_vec_point) = grandprod_protocol1.atomic_pop.get_a();
        let (theta1_vec_hat_index, theta1_vec_point_index) = grandprod_protocol1.atomic_pop.get_a_index();

       
        let mut lincomb_protocol1 = LinComb::new(
                theta1_vec_hat,
                theta1_vec_point.clone(),
                theta1_vec_hat_index,
                theta1_vec_point_index.clone(),
                (target_len, 1),
                num_col + 2,
                theta1_coeff.clone(),
                theta1_coeff_index.clone(),
            );
        let flag12 = lincomb_protocol1.verify_as_subprotocol(trans);
        assert!(flag12, "LinComb protocol 1 failed");

        // Get the projections of the input matrices of Theta1
        let (target_combined_hats1, target_combined_points1) = lincomb_protocol1.atomic_pop.get_inputs();
        let (target_combined_hats_index1, target_combined_points_index1) = lincomb_protocol1.atomic_pop.get_inputs_index();


        let flag1 = flag11 && flag12;

        // -------------------------------------------------------------------------------------------------------------

        // GrandProd + LinComb chain for Theta3 (fix accessor usage & input construction)
        let mut grandprod_protocol3 = GrandProd::new(theta3, theta3_index, target_len);
        let flag31 = grandprod_protocol3.verify_as_subprotocol(trans);
        assert!(flag31, "GrandProd protocol 3 failed");

        let (theta3_vec_hat, theta3_vec_point) = grandprod_protocol3.atomic_pop.get_a();
        let (theta3_vec_hat_index, theta3_vec_point_index) = grandprod_protocol3.atomic_pop.get_a_index();

       
        let mut lincomb_protocol3 = LinComb::new(
                theta3_vec_hat,
                theta3_vec_point.clone(),
                theta3_vec_hat_index,
                theta3_vec_point_index.clone(),
                (target_len, 1),
                num_col + 2,
                theta3_coeff.clone(),
                theta3_coeff_index.clone(),
            );
        let flag32 = lincomb_protocol3.verify_as_subprotocol(trans);
        assert!(flag32, "LinComb protocol 3 failed");

        // Get the projections of the input matrices of Theta3
        let (target_combined_hats3, target_combined_points3) = lincomb_protocol3.atomic_pop.get_inputs();
        let (target_combined_hats_index3, target_combined_points_index3) = lincomb_protocol3.atomic_pop.get_inputs_index();


        let flag3 = flag31 && flag32;

        // -------------------------------------------------------------------------------------------------------------

        // Batch the projections of target_combined_hat1 and target_combined_hats3

        let mut flag_target = flag1 && flag3;

        for i in 0..(num_col + 2) {
            let mut batchpoint_protocol = BatchPoint::<F>::new(
                vec![target_combined_hats1[i].clone(), target_combined_hats3[i]],
                vec![target_combined_points1[i].clone(), target_combined_points3[i].clone()],
                vec![target_combined_hats_index1[i].clone(), target_combined_hats_index3[i].clone()],
                vec![target_combined_points_index1[i].clone(), target_combined_points_index3[i].clone()],
            );

            flag_target = batchpoint_protocol.verify_as_subprotocol(trans) && flag_target;

            let (hat, point) = batchpoint_protocol.atomic_pop.get_c();
            let (hat_index, point_index) = batchpoint_protocol.atomic_pop.get_c_index();
           

            if i == 1 {
                target_auxiliary_hat = hat;
                target_auxiliary_points = point;
                target_auxiliary_hat_index = hat_index;
                target_auxiliary_points_index = point_index;
            } else if i == 0 {
                flag_target &= xi::xi_ip_from_challenges(&point.0, &vec![F::one(); point.0.len()]) == hat;
            } else {
                target_hats.push(hat);
                target_points.push(point);
                target_hats_index.push(hat_index);
                target_points_index.push(point_index);
            }
        } 
        
        // ===================================================================================================================
        // === We are going to verify Theta2 and Theta4 similarly
        // -------------------------------------------------------------------------------------------------------------------
        // GrandProd + LinComb chain for Theta2
        // theta2 vector length == table_len (table rows), must pass table_len to GrandProd
        let mut grandprod_protocol2 = GrandProd::new(theta2, theta2_index, table_len);
        let flag21 = grandprod_protocol2.verify_as_subprotocol(trans);
        assert!(flag21, "GrandProd protocol 2 failed");

        let (theta2_vec_hat, theta2_vec_point) = grandprod_protocol2.atomic_pop.get_a();
        let (theta2_vec_hat_index, theta2_vec_point_index) = grandprod_protocol2.atomic_pop.get_a_index();


        let mut lincomb_protocol2 = LinComb::new(
                theta2_vec_hat,
                theta2_vec_point.clone(),
                theta2_vec_hat_index,
                theta2_vec_point_index.clone(),
                (table_len, 1), // shape matches table vectors
                num_col + 2,
                theta2_coeff.clone(),
                theta2_coeff_index.clone(),
            );
        let flag22 = lincomb_protocol2.verify_as_subprotocol(trans);
        assert!(flag22, "LinComb protocol 2 failed");


        // Get the projections of the input matrices of Theta2
        let (table_combined_hats2, table_combined_points2) = lincomb_protocol2.atomic_pop.get_inputs();
        let (table_combined_hats_index2, table_combined_points_index2) = lincomb_protocol2.atomic_pop.get_inputs_index();


        let flag2 = flag21 && flag22;

        // -------------------------------------------------------------------------------------------------------------

        // GrandProd + LinComb chain for Theta4 (fix accessor usage & input construction)
        // theta4 vector length == table_len as well
        let mut grandprod_protocol4 = GrandProd::new(theta4, theta4_index, table_len);
        let flag41 = grandprod_protocol4.verify_as_subprotocol(trans);
        assert!(flag41, "GrandProd protocol 4 failed");

        let (theta4_vec_hat, theta4_vec_point) = grandprod_protocol4.atomic_pop.get_a();
        let (theta4_vec_hat_index, theta4_vec_point_index) = grandprod_protocol4.atomic_pop.get_a_index();

        let mut lincomb_protocol4 = LinComb::new(
            theta4_vec_hat,
            theta4_vec_point.clone(),
            theta4_vec_hat_index,
            theta4_vec_point_index.clone(),
            (table_len, 1),
            num_col + 2,
            theta4_coeff.clone(),
            theta4_coeff_index.clone(),
        );
        let flag42 = lincomb_protocol4.verify_as_subprotocol(trans);
        assert!(flag42, "LinComb protocol 4 failed");

        // Get the projections of the input matrices of Theta4
        let (table_combined_hats4, table_combined_points4) = lincomb_protocol4.atomic_pop.get_inputs();
        let (table_combined_hats_index4, table_combined_points_index4) = lincomb_protocol4.atomic_pop.get_inputs_index();


        let flag4 = flag41 && flag42;

        // -------------------------------------------------------------------------------------------------------------

        // Batch the projections of target_combined_hat2 and target_combined_hats4

        let mut flag_table = flag2 && flag4;

        for i in 0..(num_col + 2) {
            let mut batchpoint_protocol = BatchPoint::<F>::new(
                vec![table_combined_hats2[i].clone(), table_combined_hats4[i]],
                vec![table_combined_points2[i].clone(), table_combined_points4[i].clone()],
                vec![table_combined_hats_index2[i].clone(), table_combined_hats_index4[i].clone()],
                vec![table_combined_points_index2[i].clone(), table_combined_points_index4[i].clone()],
            );

            // Verification does not reconstruct original table column input matrix; skip set_input
            flag_table = batchpoint_protocol.verify_as_subprotocol(trans) && flag_table;

            let (hat, point) = batchpoint_protocol.atomic_pop.get_c();
            let (hat_index, point_index) = batchpoint_protocol.atomic_pop.get_c_index();



            if i == 1 {
                table_auxiliary_hat = hat;
                table_auxiliary_points = point;
                table_auxiliary_hat_index = hat_index;
                table_auxiliary_points_index = point_index;
                self.batchpoint_table_auxiliary = batchpoint_protocol;
            } else if i == 0 {
                flag_table &= xi::xi_ip_from_challenges(&point.0, &vec![F::one(); point.0.len()]) == hat;
            } else {
                table_hats.push(hat);
                table_points.push(point);
                table_hats_index.push(hat_index);
                table_points_index.push(point_index);
                self.batchpoint_table.push(batchpoint_protocol);
            }
        }

        // ===================================================================================================================



        // Record transcript data into atomic_pop
        self.atomic_pop.set_pop_trans(
            z, theta1, theta2, theta3, theta4,
            coeffs,
            target_hats, target_points, table_hats, table_points,
            target_auxiliary_hat, target_auxiliary_points,
            table_auxiliary_hat, table_auxiliary_points,
            z_index, theta1_index, theta2_index, theta3_index, theta4_index,
            coeffs_index,
            target_hats_index, target_points_index, table_hats_index, table_points_index,
            target_auxiliary_hat_index, target_auxiliary_points_index,
            table_auxiliary_hat_index, table_auxiliary_points_index,
        );

        println!("✅ LookupProof reduce_prover completed successfully");


        flag_theta && flag_coeffs && flag_target && flag_table
    }

    fn prepare_atomic_pop(&mut self) -> bool {

        if !self.atomic_pop.ready.0 {
            panic!("⚠️  Proof data not ready for Lookup pop preparation!! Run reduce_prover first..");
        }

        let z_index = self.atomic_pop.mapping.z_index;
        let theta1_index = self.atomic_pop.mapping.theta1_index;
        let theta2_index = self.atomic_pop.mapping.theta2_index;
        let theta3_index = self.atomic_pop.mapping.theta3_index;
        let theta4_index = self.atomic_pop.mapping.theta4_index;
        let coeffs_index = self.atomic_pop.mapping.coeffs_index.clone();

        let z_expr = ArithmeticExpression::input(z_index);
        let theta1_expr = ArithmeticExpression::input(theta1_index);
        let theta2_expr = ArithmeticExpression::input(theta2_index);
        let theta3_expr = ArithmeticExpression::input(theta3_index);
        let theta4_expr = ArithmeticExpression::input(theta4_index);
        let coeffs_expr: Vec<ArithmeticExpression<F>> = coeffs_index.iter().map(|&i| ArithmeticExpression::input(i)).collect();

        // Check that Theta1 * Theta2 = Theta3 * Theta4
        let check = ArithmeticExpression::sub(
            ArithmeticExpression::mul(theta1_expr, theta2_expr),
            ArithmeticExpression::mul(theta3_expr, theta4_expr),
        );
        self.atomic_pop.set_check(check);

        // Check the coeffs_expr is consistant with z_expr
        // [0, 1, z, 1+z, z^2, z^3, ...] to the transcript such that it can be read by the verifier
        let mut links = Vec::new();
        links.push(coeffs_expr[0].clone()); // the first coefficient is 0
        links.push(
            ArithmeticExpression::sub(
                coeffs_expr[1].clone(), // 1
                ArithmeticExpression::constant(F::one())
            )
        ); // the second coefficient is 1
        links.push(
            ArithmeticExpression::sub(
                coeffs_expr[2].clone(), // 1
                z_expr.clone()
            )
        ); // the third coefficient is z
        links.push(
            ArithmeticExpression::sub(
                coeffs_expr[3].clone(), // 1
                ArithmeticExpression::add(z_expr.clone(), ArithmeticExpression::constant(F::one()))
            )
        ); // the fourth coefficient is 1+z
        let mut z_power_expr = ArithmeticExpression::mul(z_expr.clone(), z_expr.clone()); // z^2
        while links.len() < coeffs_expr.len() {
            links.push(
                ArithmeticExpression::sub(
                    coeffs_expr[links.len()].clone(),
                    z_power_expr.clone()
                )
            );
            z_power_expr = ArithmeticExpression::mul(z_power_expr, z_expr.clone());
        }

        self.atomic_pop.set_links(links);

        let flag1 = self.grandprod1.prepare_atomic_pop() && self.lincomb1.prepare_atomic_pop();
        let flag2 = self.grandprod2.prepare_atomic_pop() && self.lincomb2.prepare_atomic_pop();
        let flag3 = self.grandprod3.prepare_atomic_pop() && self.lincomb3.prepare_atomic_pop();
        let flag4 = self.grandprod4.prepare_atomic_pop() && self.lincomb4.prepare_atomic_pop();

        let flag_batch_auxiliary = self.batchpoint_target_auxiliary.prepare_atomic_pop() && self.batchpoint_table_auxiliary.prepare_atomic_pop();
        let flag_batch_target = (0..self.batchpoint_target.len())
            .map(|i| self.batchpoint_target[i].prepare_atomic_pop())
            .fold(true, |a, b| a && b);
        let flag_batch_table = (0..self.batchpoint_table.len())
            .map(|i| self.batchpoint_table[i].prepare_atomic_pop())
            .fold(true, |a, b| a && b);

        self.atomic_pop.is_ready() && flag1 && flag2 && flag3 && flag4 && flag_batch_auxiliary && flag_batch_target && flag_batch_table
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder<F>) -> bool {

        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in Lookup when synthesizing constraints");
            return false;
        }
        
        // 1. Add the main 'check' constraint
        cs_builder.add_constraint(self.atomic_pop.check.clone());

        // 2. Add 'link_inputs' constraints
        for constraint in &self.atomic_pop.link_inputs {
            cs_builder.add_constraint(constraint.clone());
        }

        let flag1 = self.grandprod1.synthesize_atomic_pop_constraints(cs_builder)
            && self.lincomb1.synthesize_atomic_pop_constraints(cs_builder);
        let flag2 = self.grandprod2.synthesize_atomic_pop_constraints(cs_builder)
            && self.lincomb2.synthesize_atomic_pop_constraints(cs_builder);
        let flag3 = self.grandprod3.synthesize_atomic_pop_constraints(cs_builder)
            && self.lincomb3.synthesize_atomic_pop_constraints(cs_builder);
        let flag4 = self.grandprod4.synthesize_atomic_pop_constraints(cs_builder)
            && self.lincomb4.synthesize_atomic_pop_constraints(cs_builder);

        let flag_batch_auxiliary = self.batchpoint_target_auxiliary.synthesize_atomic_pop_constraints(cs_builder)
            && self.batchpoint_table_auxiliary.synthesize_atomic_pop_constraints(cs_builder);
        let flag_batch_target = (0..self.batchpoint_target.len())
            .map(|i| self.batchpoint_target[i].synthesize_atomic_pop_constraints(cs_builder))
            .fold(true, |a, b| a && b);
        let flag_batch_table = (0..self.batchpoint_table.len())
            .map(|i| self.batchpoint_table[i].synthesize_atomic_pop_constraints(cs_builder))
            .fold(true, |a, b| a && b);

        self.atomic_pop.is_ready() && flag1 && flag2 && flag3 && flag4 && flag_batch_auxiliary && flag_batch_target && flag_batch_table

     }

}



/// Utility functions for lincomb of vectors 
fn lincomb<F: PrimeField + Send + Sync>(vecs: &Vec<Vec<F>>, coeffs: &Vec<F>) -> Vec<F> {
    assert!(!vecs.is_empty(), "lincomb: empty vecs");
    assert_eq!(vecs.len(), coeffs.len(), "lincomb: length mismatch");
    let row_len = vecs[0].len();
    // Parallel over row positions
    (0..row_len).into_par_iter().map(|i| {
        let mut acc = F::zero();
        for (col_idx, col_vec) in vecs.iter().enumerate() { acc += coeffs[col_idx] * col_vec[i]; }
        acc
    }).collect()
}

fn grandprod<F: PrimeField + Send + Sync>(vec: &Vec<F>) -> F {
    if vec.is_empty() { return F::one(); }
    vec.par_iter().cloned().reduce(|| F::one(), |a,b| a * b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{Zero, One};
    use crate::utils::table_builder::ActivationTable;
    use fsproof::Transcript;
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;
    use std::time::Instant;

    use mat::MyInt;


    fn to_field_vec(src: &Vec<MyInt>) -> Vec<Fr> {
        src.iter().map(|&x| if x>=0 { Fr::from(x as u64) } else { -Fr::from((-x) as u64) }).collect()
    }

    #[test]
    fn test_lookupprotocol() {
        // Build table and random sample of inputs
        let table = ActivationTable::new();
        let (min_x, max_x) = table.get_input_range();
        let mut rng = StdRng::seed_from_u64(2025);
        let m = (1 << 20) as usize; // small size for quick unit test
        let mut x_vec = Vec::with_capacity(m);
        for _ in 0..m { x_vec.push(rng.random_range(min_x..=max_x)); }

        // Generate lookup inputs (alpha,beta,gamma,target_aux,table_aux)
        let (t_alpha, t_beta, t_gamma, t_aux, tbl_aux) = table.gen_lookup_inputs(&x_vec);
        let (tbl_alpha, tbl_beta, tbl_gamma) = table.get_lookup_table();

        // Convert to field elements for protocol
        let target_alpha_f = to_field_vec(&t_alpha);
        let target_beta_f  = to_field_vec(&t_beta);
        let target_gamma_f = to_field_vec(&t_gamma);
        let target_aux_f   = to_field_vec(&t_aux); // counts are non-negative
        let table_alpha_f  = to_field_vec(&tbl_alpha);
        let table_beta_f   = to_field_vec(&tbl_beta);
        let table_gamma_f  = to_field_vec(&tbl_gamma);
        let table_aux_f    = to_field_vec(&tbl_aux);

        // Assemble protocol input: num_col=3 (alpha,beta,gamma)
        let mut proto = LookUp::<Fr>::new(3, target_alpha_f.len(), table_alpha_f.len());
        proto.set_input(
            vec![target_alpha_f, target_beta_f, target_gamma_f],
            vec![table_alpha_f,  table_beta_f,  table_gamma_f ],
            target_aux_f,
            table_aux_f,
        );

        let mut trans = Transcript::new(Fr::zero());
        let t_start = Instant::now();
         assert!(proto.reduce_prover(&mut trans), "reduce_prover returned false");
        let elapsed = t_start.elapsed();
        println!("lookup reduce_prover time: {:?}", elapsed);
            
        assert!(trans.pointer >= 5, "Transcript pointer unexpected (got {})", trans.pointer);

        // reduce_prover 之后仅应保证基本 transcript 数据 ready. check / links 在 prepare_atomic_pop 设置
        assert!(proto.atomic_pop.ready.0, "basic transcript data not marked ready after reduce_prover");
        assert!(!proto.atomic_pop.ready.1 && !proto.atomic_pop.ready.2, "check/links should not be set before prepare_atomic_pop");

        // New: validate Theta multiplicative relation using stored values
        assert_eq!(proto.atomic_pop.theta1 * proto.atomic_pop.theta2, proto.atomic_pop.theta3 * proto.atomic_pop.theta4, "Theta multiplicative relation broken (post reduce)");

        // New: validate coefficient sequence correctness and length
        let z_val = proto.atomic_pop.z;
        let coeffs = &proto.atomic_pop.coeffs;
        assert_eq!(coeffs.len(), 4 + proto.protocol_input.num_col, "coeffs length mismatch: expected {} got {}", 4 + proto.protocol_input.num_col, coeffs.len());
        assert_eq!(coeffs[0], Fr::zero(), "c0 should be 0");
        assert_eq!(coeffs[1], Fr::one(), "c1 should be 1");
        assert_eq!(coeffs[2], z_val, "c2 should be z");
        assert_eq!(coeffs[3], z_val + Fr::one(), "c3 should be 1+z");
        let mut z_pow = z_val * z_val; // z^2
        for (i, c) in coeffs[4..].iter().enumerate() { // expect z^{2+i}
            assert_eq!(*c, z_pow, "coeff power mismatch at index {} expected z^{}", 4 + i, 2 + i);
            z_pow *= z_val;
        }
        let coeffs_index = &proto.atomic_pop.mapping.coeffs_index;
        assert_eq!(coeffs_index.len(), coeffs.len(), "coeffs_index length mismatch");
        // record snapshot to ensure prepare_atomic_pop does not mutate
        let coeffs_snapshot = coeffs.clone();
        let coeffs_index_snapshot = coeffs_index.clone();

        // === Added checks: verify each stored hat equals projection of corresponding input matrix at its point ===
        use mat::utils::matdef::DenseMatFieldCM;
        // For BatchProj combo test we need proj_lr on DenseMatCM, bring MatOps trait into scope
        // use mat::utils::matdef::MatOps;

        // Target columns (skip ones & auxiliary which are not stored in target_hats)
        for (k, hat) in proto.atomic_pop.target_hats.iter().enumerate() {
            let col_vec = &proto.protocol_input.target[k]; // k maps to target column k
            let mat = DenseMatFieldCM::from_data(vec![col_vec.clone()]);
            let point = &proto.atomic_pop.target_points[k]; // (xl,xr)
            let expected = mat.proj_lr_challenges(&point.0, &point.1);
            assert_eq!(*hat, expected, "target column {} hat mismatch", k);
        }

        // Table columns
        for (k, hat) in proto.atomic_pop.table_hats.iter().enumerate() {
            let col_vec = &proto.protocol_input.table[k];
            let mat = DenseMatFieldCM::from_data(vec![col_vec.clone()]);
            let point = &proto.atomic_pop.table_points[k];
            let expected = mat.proj_lr_challenges(&point.0, &point.1);
            assert_eq!(*hat, expected, "table column {} hat mismatch", k);
        }

        // === New: test prepare_atomic_pop ===
        assert!(proto.prepare_atomic_pop(), "prepare_atomic_pop failed");
        assert_eq!(proto.atomic_pop.coeffs, coeffs_snapshot, "coeffs mutated during prepare_atomic_pop");
        assert_eq!(proto.atomic_pop.mapping.coeffs_index, coeffs_index_snapshot, "coeffs_index mutated during prepare_atomic_pop");
        assert!(proto.atomic_pop.ready.1 && proto.atomic_pop.ready.2, "check/links not set by prepare_atomic_pop");
        assert!(proto.atomic_pop.is_ready(), "atomic_pop should report ready after prepare_atomic_pop");

        // === New: test synthesize_atomic_pop_constraints ===
        use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder;
        let mut cs_builder = ConstraintSystemBuilder::<Fr>::new();
        // 将 transcript 序列值作为私有输入（ArithmeticExpression::input 对应 transcript 顺序索引）。
        let pri_inputs_lookup = trans.get_trans_seq();
        cs_builder.set_public_inputs(Vec::new());
        cs_builder.set_private_inputs(pri_inputs_lookup.clone());
        let ok = proto.synthesize_atomic_pop_constraints(&mut cs_builder);
        assert!(ok, "synthesize_atomic_pop_constraints returned false");
        assert!(cs_builder.arithmetic_constraints.len() >= 1 + proto.atomic_pop.coeffs.len(),
            "unexpected number of constraints: {} < {}", cs_builder.arithmetic_constraints.len(), 1 + proto.atomic_pop.coeffs.len());

        // === New: use Groth16 to prove & verify the synthesized constraints ===
        {
            use atomic_proof::pop::groth16::curves::bls12_381::{ProvingKey, VerifyingKey, Proof};
            use atomic_proof::pop::groth16::Groth16Prover;
            use ark_std::rand::rngs::StdRng;
            use ark_std::rand::SeedableRng;

            let mut builder_all_pub = cs_builder.clone(); // 已含私有输入
            builder_all_pub.set_public_inputs(Vec::new());

            // 约束自检
            if let Err(e) = builder_all_pub.validate_constraints() {
                // Debug: dump first 20 constraints and the problematic index mentioned in error string
                println!("DEBUG Lookup constraint dump (first 20): pub_len={} pri_len={} total_constraints={}",
                    builder_all_pub.num_pub_inputs, builder_all_pub.num_pri_inputs, builder_all_pub.arithmetic_constraints.len());
                for (ci, cexpr) in builder_all_pub.arithmetic_constraints.iter().enumerate().take(20) {
                    println!("Constraint[{}] = {:?}", ci, cexpr);
                }
                panic!("Lookup constraints validation failed before Groth16: {}", e);
            }

            let mut g_rng = StdRng::seed_from_u64(987654321u64);
            let (pk, vk): (ProvingKey, VerifyingKey) = Groth16Prover::setup_bls12_381(&builder_all_pub, &mut g_rng)
                .expect("Groth16 setup should succeed for lookup");

            let pub_inputs_vec = builder_all_pub.pub_inputs.clone();
            let pri_inputs_vec = builder_all_pub.pri_inputs.clone(); // empty

            let proof: Proof = Groth16Prover::prove_with_pub_pri_bls12_381(
                &pk,
                builder_all_pub,
                pub_inputs_vec.clone(),
                pri_inputs_vec.clone(),
                &mut g_rng,
            ).expect("Groth16 proof should succeed for lookup");

            let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
            let valid = Groth16Prover::verify_bls12_381(&prepared_vk, &pub_inputs_vec, &proof)
                .expect("verification should not error");
            assert!(valid, "Groth16 proof for lookup constraints should be valid");
        }

        // Auxiliary vectors (if points stored non-empty)
        if !proto.atomic_pop.auxiliary_target_points.0.is_empty() {
            let aux_vec = &proto.protocol_input.auxiliary_target;
            let mat = DenseMatFieldCM::from_data(vec![aux_vec.clone()]);
            let p = &proto.atomic_pop.auxiliary_target_points;
            let expected = mat.proj_lr_challenges(&p.0, &p.1);
            assert_eq!(proto.atomic_pop.auxiliary_target_hat, expected, "auxiliary target hat mismatch");
        }
        if !proto.atomic_pop.auxiliary_table_points.0.is_empty() {
            let aux_vec = &proto.protocol_input.auxiliary_table;
            let mat = DenseMatFieldCM::from_data(vec![aux_vec.clone()]);
            let p = &proto.atomic_pop.auxiliary_table_points;
            let expected = mat.proj_lr_challenges(&p.0, &p.1);
            assert_eq!(proto.atomic_pop.auxiliary_table_hat, expected, "auxiliary table hat mismatch");
        }



        use atomic_proof::protocols::batchproj::BatchProj;
        use mat::utils::matdef::DenseMatCM; 
        use mat::MyInt; 
        use atomic_proof::atomic_protocol::AtomicMatProtocol;


        let hat_indices_bp = vec![
            proto.atomic_pop.mapping.auxiliary_target_hat_index,
            proto.atomic_pop.mapping.auxiliary_table_hat_index,
        ];
        let point_indices_bp = vec![
            proto.atomic_pop.mapping.auxiliary_target_points_index.clone(),
            proto.atomic_pop.mapping.auxiliary_table_points_index.clone(),
        ];
        let hats_bp = vec![
            proto.atomic_pop.auxiliary_target_hat,
            proto.atomic_pop.auxiliary_table_hat,
        ];
        let points_bp = vec![
            proto.atomic_pop.auxiliary_target_points.clone(),
            proto.atomic_pop.auxiliary_table_points.clone(),
        ];

        
        let mut mats_bp: Vec<DenseMatCM<MyInt, Fr>> = Vec::new();
        {
           
            let rows_t = proto.protocol_input.target_len;
            let mut mat_t = DenseMatCM::<MyInt, Fr>::new(rows_t, 1);
            let mut col_t: Vec<MyInt> = Vec::with_capacity(rows_t);
            for v in &t_aux { col_t.push(*v as MyInt); }
            mat_t.set_data(vec![col_t]);
            mats_bp.push(mat_t);

            let rows_tab = proto.protocol_input.table_len;
            let mut mat_tab = DenseMatCM::<MyInt, Fr>::new(rows_tab, 1);
            let mut col_tab: Vec<MyInt> = Vec::with_capacity(rows_tab);
            for v in &tbl_aux { col_tab.push(*v as MyInt); }
            mat_tab.set_data(vec![col_tab]);
            mats_bp.push(mat_tab);
        }

        let mut batchproj = BatchProj::new(hats_bp.clone(), points_bp.clone(), hat_indices_bp.clone(), point_indices_bp.clone());
        batchproj.set_input(mats_bp);

        assert!(batchproj.reduce_prover(&mut trans), "BatchProj reduce_prover failed (combo)");
        assert!(batchproj.atomic_pop.ready.0, "BatchProj basic transcript data not ready");

        assert!(batchproj.prepare_atomic_pop(), "BatchProj prepare_atomic_pop failed (combo)");
        assert!(batchproj.atomic_pop.is_ready(), "BatchProj should be ready after prepare");

        use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder as CSB2;
        let mut cs_builder_combo = CSB2::<Fr>::new();
        let all_pri_inputs_combo = trans.get_trans_seq();
        let mapping_bp = &batchproj.atomic_pop.mapping;
        let mut max_pub_idx = mapping_bp.final_c_hat_index;
        for &idx in mapping_bp.final_c_point_index.0.iter() { if idx > max_pub_idx { max_pub_idx = idx; } }
        for &idx in mapping_bp.final_c_point_index.1.iter() { if idx > max_pub_idx { max_pub_idx = idx; } }
        let mut pub_inputs_combo = vec![Fr::zero(); max_pub_idx + 1];
        pub_inputs_combo[mapping_bp.final_c_hat_index] = trans.get_at_position(mapping_bp.final_c_hat_index);
        for &idx in mapping_bp.final_c_point_index.0.iter() { pub_inputs_combo[idx] = trans.get_at_position(idx); }
        for &idx in mapping_bp.final_c_point_index.1.iter() { pub_inputs_combo[idx] = trans.get_at_position(idx); }
        cs_builder_combo.set_public_inputs(pub_inputs_combo);
        cs_builder_combo.set_private_inputs(all_pri_inputs_combo);

        assert!(proto.synthesize_atomic_pop_constraints(&mut cs_builder_combo), "Lookup synthesize failed in combo builder");
        let lookup_constraint_count = cs_builder_combo.arithmetic_constraints.len();
        assert!(batchproj.synthesize_atomic_pop_constraints(&mut cs_builder_combo), "BatchProj synthesize failed in combo builder");
        let combo_constraint_count = cs_builder_combo.arithmetic_constraints.len();
        assert!(combo_constraint_count > lookup_constraint_count, "Combo constraints count not increased after adding BatchProj");

        if let Err(e) = cs_builder_combo.validate_constraints() {
            panic!("Combined (Lookup + BatchProj) constraints validation failed: {}", e);
        }


        
        // =========================================================Test Verify=================
        let mut trans_verifier = trans.clone();
        trans_verifier.reset_pointer();

        assert!(proto.verify_as_subprotocol(&mut trans_verifier), "Protocol verification failed");
        assert!(batchproj.verify_as_subprotocol(&mut trans_verifier), "BatchProj verification failed");

        assert_eq!(trans_verifier.pointer, trans_verifier.trans_seq.len(), "Transcript pointer mismatch");

    }
}