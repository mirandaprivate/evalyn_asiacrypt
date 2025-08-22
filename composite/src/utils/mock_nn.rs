//! ```text
//! Generate a mock NN with the layers as follows:
//!     
//!      X_{i+1} = phi( W_i * X_i + 127 * B_i )
//!     
//! where phi is the quantized function for sigmoid
//! phi is defined in table_builder.rs
//! 
//! Let C_i = W_i * X_i + 127 * B_i
//! 
//! The input to Evalyn to prove this mock NN is 
//!     
//!     [W_1, ..., W_D]
//!     [B_1, ..., B_D]
//!     [X_1, ..., X_D]
//!     [C_1, ..., C_D]
//! 
//! The vec 
//!     
//!     (alpha, beta, gamma) and auxiliary s 
//!   
//! for phi such that:  
//!     alpha <= X < beta
//! 
//! The key-value table of phi
//!     
//!     (k, v) and auxiliary t
//! 
//! The lookup auxiliary for the inequalities 
//!     
//!     s_alpha,  t_alpha,      for     X - alpha + 1 ∈ [1, ..., 2^k]
//!     s_beta,   t_beta,        for       beta - X  ∈ [1, ..., 2^k]
//! ```
//! 
use ark_ff::{PrimeField, Zero};
use ark_ec::pairing::{Pairing, PairingOutput};

use ark_poly_commit::smart_pc::SmartPC;
use ark_poly_commit::smart_pc::data_structures::{
    Trans as PcsTrans,
    UniversalParams as PcsPP,
};

use ark_serialize::{CanonicalSerialize,Compress};

// Removed unused imports (RefCell, Rc, rayon prelude, RNG traits)

use atomic_proof::{MatContainerMyInt, PointsContainer, PointInfo};
use mat::DenseMatCM;

use super::{matop,table_builder};
use crate::{MyInt, MyShortInt};


#[derive(Debug, Clone)]
pub struct MockNNRawInput
{
    pub depth: usize,
    pub weight_shape: (usize, usize),
    pub weights: Vec<Vec<Vec<MyShortInt>>>,
    pub biases: Vec<Vec<MyShortInt>>,
    pub x: Vec<MyShortInt>,
}

#[derive(Debug, Clone)]
pub struct MockNN<F>
where
    F: PrimeField + From<MyInt> + Send + Sync,
{
    pub depth: usize,
    pub weight_shape: (usize, usize),
    pub weights: Vec<DenseMatCM<MyInt, F>>,
    pub biases: Vec<DenseMatCM<MyInt, F>>,
    pub nn_input: DenseMatCM<MyInt, F>,
    pub nn_output: DenseMatCM<MyInt, F>,
    pub layer_inputs: Vec<DenseMatCM<MyInt, F>>,
    pub layer_outputs: Vec<DenseMatCM<MyInt, F>>,
    pub mul_outputs: Vec<DenseMatCM<MyInt, F>>,
    pub phi_inputs: Vec<DenseMatCM<MyInt, F>>, // per-layer phi inputs
    pub lookup_table: Vec<Vec<MyInt>>,
    pub lookup_target: Vec<Vec<MyInt>>,
    pub lookup_table_auxiliary: Vec<MyInt>,
    pub lookup_target_auxiliary: Vec<MyInt>,
    pub range_table_auxiliary: Vec<Vec<MyInt>>,
    pub range_target_auxiliary: Vec<Vec<MyInt>>,
}


/// The message from the veifier's view
#[derive(Debug, Clone)]
pub struct MockNNProj<F>
where
    F: PrimeField + From<MyInt> + Send + Sync,
{
    pub weights_proj: Vec<PointInfo<F>>,
    pub biases_proj: Vec<PointInfo<F>>,
    pub nn_input_proj: PointInfo<F>,
    pub nn_output_proj: PointInfo<F>,
    pub layer_inputs_proj: Vec<PointInfo<F>>,
    pub layer_outputs_proj: Vec<PointInfo<F>>,
    pub mul_outputs_proj: Vec<PointInfo<F>>,
    pub phi_inputs_proj: Vec<PointInfo<F>>,
    pub lookup_table_proj: Vec<PointInfo<F>>,
    pub lookup_target_proj: Vec<PointInfo<F>>,
    pub lookup_table_auxiliary_proj: PointInfo<F>,
    pub lookup_target_auxiliary_proj: PointInfo<F>,
    pub range_table_auxiliary_proj: Vec<PointInfo<F>>,
    pub range_target_auxiliary_proj: Vec<PointInfo<F>>,
}

impl<F> MockNNProj<F>
where
    F: PrimeField + From<MyInt> + Send + Sync,
{
    pub fn new() -> Self {
        MockNNProj {
            weights_proj: Vec::new(),
            biases_proj: Vec::new(),
            nn_input_proj: PointInfo::default(),
            nn_output_proj: PointInfo::default(),
            layer_inputs_proj: Vec::new(),
            layer_outputs_proj: Vec::new(),
            mul_outputs_proj: Vec::new(),
            phi_inputs_proj: Vec::new(),
            lookup_table_proj: Vec::new(),
            lookup_target_proj: Vec::new(),
            lookup_table_auxiliary_proj: PointInfo::default(),
            lookup_target_auxiliary_proj: PointInfo::default(),
            range_table_auxiliary_proj: Vec::new(),
            range_target_auxiliary_proj: Vec::new(),
        }
    }

        // Push leaf node Projections of the NN to the point container for batch proj
        pub fn push_leaves_to_point_container(&self, point_container: &mut PointsContainer<F>) {

        for i in 0..self.weights_proj.len() {
            point_container.push_point(&self.weights_proj[i]);
        }

        for i in 0..self.biases_proj.len() {
            point_container.push_point(&self.biases_proj[i]);
        }

        point_container.push_point(&self.nn_input_proj);
        point_container.push_point(&self.nn_output_proj);

        for i in 0..self.lookup_table_proj.len() {
            point_container.push_point(&self.lookup_table_proj[i]);
        }

        for i in 0..self.layer_outputs_proj.len() {
            point_container.push_point(&self.layer_outputs_proj[i]);
        }

        for i in 0..self.lookup_target_proj.len() {
            point_container.push_point(&self.lookup_target_proj[i]);
        }

        point_container.push_point(&self.lookup_table_auxiliary_proj);
        point_container.push_point(&self.lookup_target_auxiliary_proj);

        for i in 0..self.range_table_auxiliary_proj.len() {
            point_container.push_point(&self.range_table_auxiliary_proj[i]);
        }

        for i in 0..self.range_target_auxiliary_proj.len() {
            point_container.push_point(&self.range_target_auxiliary_proj[i]);
        }
    }
}


impl MockNNRawInput {
    pub fn gen_random(
        depth: usize,
        weight_shape: (usize, usize),
    ) -> Self {
        let m = weight_shape.0;
        let n = weight_shape.1;

        let weights = (0..depth).into_iter().map(|_| matop::gen_rand_shortint_mat(m, n)).collect();
        let biases = (0..depth).into_iter().map(|_| matop::gen_rand_shortint_vec(m)).collect();
        let x = matop::gen_rand_shortint_vec(m);

        MockNNRawInput {
            depth,
            weight_shape,
            weights,
            biases,
            x,
        }
    }
}

impl<F> MockNN<F>
where
    F: PrimeField + From<MyInt> + Send + Sync,
{
    // Generate random NN parameters, NN input and internal values are zero at this stage
    pub fn new() -> Self {
        MockNN {
            depth: 0,
            weight_shape: (0, 0),
            weights: Vec::new(),
            biases: Vec::new(),
            nn_input: DenseMatCM::<MyInt, F>::default(),
            nn_output: DenseMatCM::<MyInt, F>::default(),
            layer_inputs: Vec::new(),
            layer_outputs: Vec::new(),
            mul_outputs: Vec::new(),
            phi_inputs: Vec::new(),
            lookup_table: Vec::new(),
            lookup_target: Vec::new(),
            lookup_table_auxiliary: Vec::new(),
            lookup_target_auxiliary: Vec::new(),
            range_table_auxiliary: Vec::new(),
            range_target_auxiliary: Vec::new(),
        }
    }

    pub fn load_weights_from_shortint(&mut self, shortint_weights:&Vec<Vec<Vec<MyShortInt>>>) {
        self.weights = (0..self.depth).map(|i| matop::shortint_to_myint_mat(&shortint_weights[i])).collect();
    }

    pub fn drop_weights(&mut self) {
        self.weights = Vec::new();
    }

    pub fn load_raw_input(&mut self, raw_input: &MockNNRawInput) {
        self.depth = raw_input.depth;
        self.weight_shape = raw_input.weight_shape;

        self.weights = (0..self.depth).map(|i| matop::shortint_to_myint_mat(&raw_input.weights[i])).collect();
        self.biases = (0..self.depth).map(|i| matop::shortint_to_myint_mat(&vec![raw_input.biases[i].clone()])).collect();

        self.nn_input = matop::shortint_to_myint_mat(&vec![raw_input.x.clone()]);

        let activation = table_builder::ActivationTable::new();
        let (alpha_table, beta_table, gamma_table) = activation.get_lookup_table();

        let lookup_table = vec![alpha_table.clone(), beta_table.clone(), gamma_table.clone()];
        
        // Other input dependent result are set to zero at this stage

        let n = self.weight_shape.0;

        // Proper zero column matrix with actual data so flatten produces zeros (not empty)
        let zero_mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; n]]);
        let layer_inputs = vec![zero_mat.clone(); self.depth];
        let layer_outputs = vec![zero_mat.clone(); self.depth];
        let mul_outputs = vec![zero_mat.clone(); self.depth];
        let phi_inputs = vec![zero_mat.clone(); self.depth];

        self.nn_output = zero_mat.clone();

        let alpha_target = vec![0 as MyInt; n * self.depth];
        let beta_target = vec![0 as MyInt; n * self.depth];
        let lookup_target = vec![alpha_target, beta_target]; // gamma separated / reconstructed during fill

        let lookup_target_auxiliary = vec![0 as MyInt; n * self.depth];
        let lookup_table_auxiliary = vec![0 as MyInt; alpha_table.len()];

        let activation = table_builder::ActivationTable::new();
        let k = activation.get_increment_bw();

        let range_target_auxiliary_vec = vec![0 as MyInt; n * self.depth];
        let range_table_auxiliary_vec = vec![0 as MyInt; (1 << k) as usize];

        self.lookup_table = lookup_table;

        self.layer_inputs = layer_inputs;
        self.layer_outputs = layer_outputs;
        self.mul_outputs = mul_outputs;
        // store per-layer phi inputs matrix form
        self.phi_inputs = phi_inputs;
        self.lookup_target = lookup_target;
        self.lookup_table_auxiliary = lookup_table_auxiliary;
        self.lookup_target_auxiliary = lookup_target_auxiliary;
        self.range_table_auxiliary = vec![range_table_auxiliary_vec; 2];
        self.range_target_auxiliary = vec![range_target_auxiliary_vec; 2];
    }
    

    pub fn clear(&mut self) {
        self.weights = Vec::new();
        self.biases = Vec::new();
        self.nn_input = DenseMatCM::<MyInt, F>::default();
        self.nn_output = DenseMatCM::<MyInt, F>::default();
        self.layer_inputs = Vec::new();
        self.layer_outputs = Vec::new();
        self.mul_outputs = Vec::new();
        self.phi_inputs = Vec::new();
        self.lookup_table = Vec::new();
        self.lookup_target = Vec::new();
        self.lookup_table_auxiliary = Vec::new();
        self.lookup_target_auxiliary = Vec::new();
    }

    pub fn get_scale(&self) -> MyInt {
        let bitwidth = std::mem::size_of::<MyShortInt>() * 8;
        (1 << (bitwidth - 1)) as MyInt - 1
    }

    pub fn fill_internal(&mut self) {
        let depth = self.depth;
        let n = self.weight_shape.0;
        let _bitwidth = std::mem::size_of::<MyShortInt>() * 8;

        // Use the provided nn_input as the starting vector to keep mul_outputs
        // consistent with the MatMul inputs used later in the protocol.
        // This avoids a mismatch where mul_outputs are derived from a different
        // random input than the one MatMul actually uses (self.nn_input).
        let mut cur_input = self.nn_input.clone();

        let mut layer_input = Vec::new();
        let mut phi_input = Vec::new();
        let mut layer_outputs = Vec::new();
        let mut mul_outputs = Vec::new();

        println!("======================Unverified NN Feed Forward ============");

        let timer = std::time::Instant::now();

        for i in 0..depth {
            layer_input.push(cur_input.clone());

            let mul_output_data = matop::mat_mul_myint(
                &self.weights[i].data,
                &cur_input.data
            );
            let mut cur_mul_output = DenseMatCM::new(n, 1);
            cur_mul_output.data = mul_output_data;
            mul_outputs.push(cur_mul_output.clone());

            let phi_input_data = matop::mat_add_myint(
                &cur_mul_output.data,
                &matop::mat_scalar_mul_myint(&self.biases[i].data, self.get_scale())
            );
            let mut cur_phi_input = DenseMatCM::new(n, 1);
            cur_phi_input.data = phi_input_data;
            phi_input.push(cur_phi_input.clone());

            let cur_output = matop::element_wise_phi(&cur_phi_input);
            layer_outputs.push(cur_output.clone());

            cur_input = cur_output;
        }

        let duration = timer.elapsed().as_secs_f64();
        println!("Feed forward completed in {:.6} seconds", duration);
        println!("=================================================");

        let flattened_phi_input = matop::flatten_and_concat(&phi_input).data.pop().unwrap();

        let activation = table_builder::ActivationTable::new();
        let (alpha_target, beta_target, gamma_target, auxiliary_target, auxiliary_table) = 
            activation.gen_lookup_inputs(&flattened_phi_input);  

        let flattened_phi_output = matop::flatten_and_concat(&layer_outputs).data.pop().unwrap();
        assert_eq!(flattened_phi_output[..10].to_vec(), gamma_target[..10].to_vec());


        let flattened_phi_input = matop::flatten_and_concat(&phi_input).data.pop().unwrap();
        let diff1 = matop::vec_sub_myint(&flattened_phi_input, &alpha_target);
        let diff2 = matop::vec_sub_myint(&&beta_target, &flattened_phi_input);

        let lookup_target = vec![alpha_target, beta_target];
        let lookup_target_auxiliary = auxiliary_target;
        let lookup_table_auxiliary = auxiliary_table;


        let k = activation.get_increment_bw();
        let (range_target_auxiliary1, range_table_auxiliary1) = table_builder::range_auxiliary_builder(&diff1, k);
        let (range_target_auxiliary2, range_table_auxiliary2) = table_builder::range_auxiliary_builder(&diff2, k);

        self.layer_inputs = layer_input;
        self.phi_inputs = phi_input;
        self.layer_outputs = layer_outputs.clone();
        self.mul_outputs = mul_outputs;
        self.lookup_target = lookup_target;
        self.lookup_table_auxiliary = lookup_table_auxiliary;
        self.lookup_target_auxiliary = lookup_target_auxiliary;
        self.range_target_auxiliary = vec![range_target_auxiliary1, range_target_auxiliary2];
        self.range_table_auxiliary = vec![range_table_auxiliary1, range_table_auxiliary2];
        self.nn_output = layer_outputs.last().unwrap().clone();
        println!("NN Feed Forward completed");
        // flattened_phi_input and flattened_phi_output can be reconstructed on demand via helper methods

    }

    // Helper: get flattened phi input vector (concatenate per-layer phi_inputs)
    pub fn flattened_phi_input(&self) -> Vec<MyInt> {
        matop::flatten_and_concat(&self.phi_inputs).data[0].clone()
    }

    // Helper: get flattened phi output vector (concatenate per-layer layer_outputs)
    pub fn flattened_phi_output(&self) -> Vec<MyInt> {
        matop::flatten_and_concat(&self.layer_outputs).data[0].clone()
    }

    // Push leaf nodes of the NN to the matrix container for batch commitment
    pub fn push_to_mat_container(&self, mat_container: &mut MatContainerMyInt<F>) {

        for i in 0..self.depth {
            mat_container.push(self.weights[i].clone());
        }

        for i in 0..self.depth {
            mat_container.push(self.biases[i].clone());
        }

        mat_container.push(self.nn_input.clone());
        mat_container.push(self.nn_output.clone());

        for i in 0..self.lookup_table.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_table[i].clone()]);
            mat_container.push(mat);
        }

        for i in 0..self.depth {
            mat_container.push(self.layer_outputs[i].clone());
        }

        for i in 0..self.lookup_target.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_target[i].clone()]);
            mat_container.push(mat);
        }

        let lookup_table_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_table_auxiliary.clone()]);
        mat_container.push(lookup_table_auxiliary_mat);
        let lookup_target_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_target_auxiliary.clone()]);
        mat_container.push(lookup_target_auxiliary_mat);

    for i in 0..self.range_table_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.range_table_auxiliary[i].clone()]);
            mat_container.push(mat);
        }

    for i in 0..self.range_target_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.range_target_auxiliary[i].clone()]);
            mat_container.push(mat);
        }
    }

    pub fn push_pars_to_mat_container(&self, mat_container: &mut MatContainerMyInt<F>) {

        for i in 0..self.depth {
            mat_container.push(self.weights[i].clone());
        }

        for i in 0..self.depth {
            mat_container.push(self.biases[i].clone());
        }

        mat_container.push(DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.nn_input.data[0].len()]]));
        mat_container.push(DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.nn_output.data[0].len()]]));

        for i in 0..self.lookup_table.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_table[i].clone()]);
            mat_container.push(mat);
        }

        for i in 0..self.depth {
            // Zero matrix matching layer_output shape (rows x 1)
            let zero_col = vec![0 as MyInt; self.layer_outputs[i].shape.0];
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![zero_col]);
            mat_container.push(mat);
        }

        for i in 0..self.lookup_target.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.lookup_target[i].len()]]);
            mat_container.push(mat);
        }

        let lookup_table_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.lookup_table_auxiliary.len()]]);
        mat_container.push(lookup_table_auxiliary_mat);
        let lookup_target_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.lookup_target_auxiliary.len()]]);
        mat_container.push(lookup_target_auxiliary_mat);

        for i in 0..self.range_table_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.range_table_auxiliary[i].len()]]);
            mat_container.push(mat);
        }

        for i in 0..self.range_target_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.range_target_auxiliary[i].len()]]);
            mat_container.push(mat);
        }
    }

    // Push leaf nodes of the NN to the matrix container for batch commitment
    pub fn push_to_mat_container_without_pars(&self, mat_container: &mut MatContainerMyInt<F>) {

        let zero_mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.weight_shape.0]; self.weight_shape.1]);
        let zero_vec = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.weight_shape.0]]);

        for _ in 0..self.depth {
            mat_container.push(zero_mat.clone());
        }

        for _ in 0..self.depth {
            mat_container.push(zero_vec.clone());
        }

        mat_container.push(self.nn_input.clone());
        mat_container.push(self.nn_output.clone());

        for _ in 0..self.lookup_table.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![vec![0 as MyInt; self.lookup_table[0].len()]]);
            mat_container.push(mat);
        }

        for i in 0..self.depth {
            mat_container.push(self.layer_outputs[i].clone());
        }

        for i in 0..self.lookup_target.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_target[i].clone()]);
            mat_container.push(mat);
        }

        let lookup_table_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_table_auxiliary.clone()]);
        mat_container.push(lookup_table_auxiliary_mat);
        let lookup_target_auxiliary_mat = DenseMatCM::<MyInt, F>::from_data(vec![self.lookup_target_auxiliary.clone()]);
        mat_container.push(lookup_target_auxiliary_mat);

        for i in 0..self.range_table_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.range_table_auxiliary[i].clone()]);
            mat_container.push(mat);
        }

        for i in 0..self.range_target_auxiliary.len() {
            let mat = DenseMatCM::<MyInt, F>::from_data(vec![self.range_target_auxiliary[i].clone()]);
            mat_container.push(mat);
        }
    }

    pub fn to_flattened_leaves(&self) -> Vec<MyInt> {
        let mut mat_container = MatContainerMyInt::new();
        self.push_to_mat_container(&mut mat_container);
        mat_container.into_flattened_vec()
    }

    // Debug helper: flattened leaves with zeroed parameters
    pub fn to_flattened_leaves_without_pars(&self) -> Vec<MyInt> {
        let mut mat_container = MatContainerMyInt::new();
        self.push_to_mat_container_without_pars(&mut mat_container);
        mat_container.into_flattened_vec()
    }

    // Collect the sorted shapes (after area-based insertion) as a signature
    pub fn shape_signature(&self) -> Vec<(usize, usize)> {
        let mut mc = MatContainerMyInt::new();
        self.push_to_mat_container(&mut mc);
        mc.sorted_shapes.clone()
    }

    pub fn shape_signature_without_pars(&self) -> Vec<(usize, usize)> {
        let mut mc = MatContainerMyInt::new();
        self.push_to_mat_container_without_pars(&mut mc);
        mc.sorted_shapes.clone()
    }

    
    pub fn into_flattened_leaves(&mut self) -> Vec<MyInt> {
        let vec = self.to_flattened_leaves();
        self.clear();
        vec
    }

    pub fn commit_to_leaves<E: Pairing>(&self, pcsrs: &PcsPP<E>) -> (PairingOutput<E>, Vec<E::G1>) {
        let vec = self.to_flattened_leaves();
        
        let (leaves_com, leaves_com_cache) = SmartPC::<E>::commit_square_myint(
            pcsrs, &vec![vec], E::ScalarField::zero()
        ).expect("commit_full failed for witness");

        (leaves_com, leaves_com_cache)
    }

    pub fn commit_to_leaves_without_pars<E: Pairing>(&self, pcsrs: &PcsPP<E>) -> (PairingOutput<E>, Vec<E::G1>) {
        let mut mat_container = MatContainerMyInt::new();
        self.push_to_mat_container_without_pars(&mut mat_container);
        let vec = mat_container.into_flattened_vec();

        let (leaves_com, leaves_com_cache) = SmartPC::<E>::commit_square_myint(
            pcsrs, &vec![vec], E::ScalarField::zero()
        ).expect("commit_full failed for witness");

        (leaves_com, leaves_com_cache)
    }

    pub fn commit_to_pars<E: Pairing>(&self, pcsrs: &PcsPP<E>) -> (PairingOutput<E>, Vec<E::G1>) {
        let mut mat_container = MatContainerMyInt::new();
        self.push_pars_to_mat_container(&mut mat_container);
        let vec = mat_container.into_flattened_vec();

        let (par_com, par_com_cache) = SmartPC::<E>::commit_square_myint(
            pcsrs, &vec![vec], E::ScalarField::zero()
        ).expect("commit_full failed for witness");

        (par_com, par_com_cache)
    }

    pub fn open_leaf_commitment<E: Pairing>(
        &self,
        pcsrs: &PcsPP<E>,
        leaf_com: &PairingOutput<E>,
        hat: &E::ScalarField,
        point: &Vec<E::ScalarField>,
        cache: &mut Vec<E::G1>,
    ) -> PcsTrans<E> {
        // Reconstruct the same flattened matrix used in commitment (single row)
        let mut mat_container = MatContainerMyInt::new();
        self.push_to_mat_container(&mut mat_container);
        let vec = mat_container.into_flattened_vec(); // Vec<MyInt>
    

        let hat_com = pcsrs.u * hat;

        let trans = SmartPC::<E>::open_square_myint(
            pcsrs,
            &vec![vec],
            point,
            &Vec::new(),
            hat_com,
            leaf_com.clone(),
            cache,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        ).expect("open_square failed in open_leaf_commitment");

        println!("Leaf commitment openning proof size: {:?} B", trans.serialized_size(Compress::Yes));
        trans
    }

    pub fn verify_leaf_commitment<E:Pairing>(
        &self,
        pcsrs: &PcsPP<E>,
        trans: &PcsTrans<E>,
        leaf_com: PairingOutput<E>,
        hat: E::ScalarField,
        point: &Vec<E::ScalarField>,
    ) -> bool {
        let hat_com = pcsrs.u * hat;
        let result = SmartPC::<E>::verify_square(
            pcsrs,
            leaf_com,
            hat_com,
            point,
            &Vec::new(),
            trans,
        ).unwrap_or(false);

        println!("Leaf commitment verification result: {}", result);
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;

    #[test]
    fn test_mock_nn() {
        // Test parameters
        let depth = 64;
        let weight_shape = (64, 64); // 4x4 weight matrices

        let raw_nn = MockNNRawInput::gen_random(depth, weight_shape);

        // Create MockNN instance
        let mut mock_nn = MockNN::<BlsFr>::new();
        mock_nn.load_raw_input(&raw_nn);
        mock_nn.fill_internal();

        // Test basic structure
        assert_eq!(mock_nn.depth, depth, "Depth should match input");
        assert_eq!(mock_nn.weight_shape, weight_shape, "Weight shape should match input");

        // Test that all vectors have correct length
        assert_eq!(mock_nn.weights.len(), depth, "Should have {} weight matrices", depth);
        assert_eq!(mock_nn.biases.len(), depth, "Should have {} bias vectors", depth);
        assert_eq!(mock_nn.layer_inputs.len(), depth, "Should have {} layer inputs", depth);
        assert_eq!(mock_nn.layer_outputs.len(), depth, "Should have {} layer outputs", depth);
        assert_eq!(mock_nn.mul_outputs.len(), depth, "Should have {} multiplication outputs", depth);
        assert_eq!(mock_nn.phi_inputs.len(), depth, "Should have {} phi inputs", depth);

        // Test matrix dimensions
        for i in 0..depth {
            assert_eq!(mock_nn.weights[i].shape.0, weight_shape.0, 
                "Weight matrix {} should have {} rows", i, weight_shape.0);
            assert_eq!(mock_nn.weights[i].shape.1, weight_shape.1, 
                "Weight matrix {} should have {} columns", i, weight_shape.1);
            
            assert_eq!(mock_nn.biases[i].shape.0, weight_shape.0, 
                "Bias vector {} should have {} rows", i, weight_shape.0);
            assert_eq!(mock_nn.biases[i].shape.1, 1, 
                "Bias vector {} should have 1 column", i);

            assert_eq!(mock_nn.layer_inputs[i].shape.0, weight_shape.0, 
                "Layer input {} should have {} rows", i, weight_shape.0);
            assert_eq!(mock_nn.layer_inputs[i].shape.1, 1, 
                "Layer input {} should have 1 column", i);
        }

        // Test lookup table structure
        assert_eq!(mock_nn.lookup_table.len(), 3, "Should have 3 lookup table components (alpha, beta, gamma)");
        assert_eq!(mock_nn.lookup_target.len(), 2, "Should have 2 lookup target components (alpha, beta)");

        // Test auxiliary data is not empty
        assert!(!mock_nn.lookup_table_auxiliary.is_empty(), "Lookup table auxiliary should not be empty");
        assert!(!mock_nn.lookup_target_auxiliary.is_empty(), "Lookup target auxiliary should not be empty");

        println!("✅ MockNN new function test passed!");
        println!("   - Depth: {}", mock_nn.depth);
        println!("   - Weight shape: {:?}", mock_nn.weight_shape);
        println!("   - Lookup table size: {}", mock_nn.lookup_table[0].len());
        println!("   - Auxiliary table size: {}", mock_nn.lookup_table_auxiliary.len());
    }


}

