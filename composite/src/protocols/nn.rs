//! Verify that the committed Transcript is correctly produced from Fiat-Shamir
// 
use ark_ff::{PrimeField, UniformRand, Zero, One};
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::rand::thread_rng;

use ark_poly_commit::smart_pc::SmartPC;
use ark_poly_commit::smart_pc::data_structures::{
    Trans as PcsTrans,
    UniversalParams as PcsPP,
};
use ark_poly_commit::smart_pc::utils::add_vec_g1;



use ark_groth16::{
    Proof as GrothProof,
    ProvingKey as GrothPK,
    PreparedVerifyingKey as CircCommit,
};

use atomic_proof::{
    Groth16Prover,
    MLPCSCommitment as PCCommit,
};
use atomic_proof::{PointsContainer, PointInfo};
// debug utils removed
use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder; // import builder
use atomic_proof::MatContainerMyInt;
use atomic_proof::protocols::{
    BatchProj,
    BatchPoint,
    Concat,
    LinComb,
    MatMul,
    MatEq,
};
use atomic_proof::AtomicMatProtocol; // bring reduce_prover trait method into scope
use mat::{DenseMatCM, DenseMatFieldCM};

// use atomic_proof::protocols::{BatchPoint, BatchProj, Hadamard, MatMul, MatSub, EqZero};
// use atomic_proof::{AtomicMatProtocol, MatContainer, PointsContainer};

use fsproof::Transcript; // BatchConstraints unused currently
 
use crate::utils::mock_nn::MockNN;
use crate::utils::mock_nn::MockNNProj;
use crate::protocols::activation::Activation;
use crate::protocols::fsbatch_groth::FSBatchGroth;
use crate::utils::matop;
use crate::utils::mock_nn::MockNNRawInput;



use mat::MyInt;


#[derive(Clone)]
pub struct ProtocolNN<E: Pairing>
where
    E: Pairing,
    E::ScalarField: Absorb + UniformRand + PrimeField + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
{
    pub depth: usize,
    pub shape: (usize, usize),
    pub par_commit: PairingOutput<E>,
    pub pop_circ_commit: CircCommit<E>,
    pub witness_commit: PairingOutput<E>,
    pub trans_reduce_commit: PCCommit<E>,
    // Private to provers
    pub par_commit_cache: Vec<E::G1>,
    pub witness_commit_cache: Vec<E::G1>,
    pub private_trans_reduce: Transcript<E::ScalarField>, // transcript unknown to verifiers
    pub pop_pk: Option<GrothPK<E>>,
    // Protocol inputs
    pub pcsrs: PcsPP<E>,
    // pub raw_data: MockNNRawInput,
    pub mock_nn: MockNN<E::ScalarField>,
    pub mock_nn_proj: MockNNProj<E::ScalarField>,
    pub cs_builder: ConstraintSystemBuilder<E::ScalarField>,
    // Transcript containing the following
    pub leaf_hat: E::ScalarField, // proj value of the leaf from BatchProj
    pub leaf_point: Vec<E::ScalarField>, // point value (left challenges) of the leaf
    pub trans_pcs: PcsTrans<E>,
    pub trans_pop: GrothProof<E>,
    pub fsbatch: FSBatchGroth<E>,

    // Indices (in Transcript) for the final public exposure produced by BatchProj
    // Only these positions are treated as public inputs; everything else stays private.
    pub leaf_hat_index: usize,
    pub leaf_point_index: (Vec<usize>, Vec<usize>),
}


impl<E> ProtocolNN<E>
where
    E: Pairing,
    E::ScalarField: Absorb + UniformRand + PrimeField,
{
    
    pub fn new(depth: usize, shape: (usize, usize)) -> Self {

        let raw_data = MockNNRawInput::gen_random(depth, shape);
        println!("************************************************************************");
        println!("========Preparing NN parameters=========================================");
        let mut mock_nn = MockNN::<E::ScalarField>::new();
        mock_nn.load_raw_input(&raw_data);

        let size = depth * shape.0 * shape.1 * 2;
        let mut q_log: usize = (size.ilog2()/2 + 1) as usize;
        if q_log < 15 { q_log = 15;}
        println!("========End Preparing NN parameters===================================== ");
        println!("************************************************************************");

        println!("************************************************************************");
        println!("========Preparing SRS parameters=========================================");
        let _timer = std::time::Instant::now();
        let mut rng = thread_rng();
        let pcsrs = SmartPC::<E>::setup(q_log, &mut rng).expect("pcs setup failed");
        let srs_size = pcsrs.serialized_size(Compress::Yes);

        println!("⬜ \x1b[1m SRS size: {:?} B  \x1b[0m", srs_size);
        println!("🕒 \x1b[1m Setup time: {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("========End Setup=========================================");
        println!("************************************************************************");

        // Populate internal (non-parameter) leaves
        mock_nn.fill_internal();

        Self {
            depth,
            shape,
            par_commit: PairingOutput::default(),
            pop_circ_commit: CircCommit::default(),
            witness_commit: PairingOutput::default(),
            trans_reduce_commit: PCCommit::default(),
            par_commit_cache: Vec::new(),
            witness_commit_cache: Vec::new(),
            private_trans_reduce: Transcript::default(),
            pop_pk: None,
            trans_pcs: PcsTrans::new(),
            trans_pop: GrothProof::default(),
            fsbatch: FSBatchGroth::default(),
            pcsrs,
            // raw_data,
            mock_nn,
            mock_nn_proj: MockNNProj::new(),
            cs_builder: ConstraintSystemBuilder::new(),
            leaf_hat: E::ScalarField::zero(),
            leaf_point: Vec::new(),
            leaf_hat_index: 0,
            leaf_point_index: (Vec::new(), Vec::new()),
        }
    }

    pub fn commit_to_pars(&mut self){
        // Populate internal (non-parameter) leaves

        println!("************************************************************************");
        println!("========Committing to Params=========================================");
        let _timer = std::time::Instant::now();
        let (par_commit, par_commit_cache) = self.mock_nn.commit_to_pars(&self.pcsrs);


        println!("\n🕒 \x1b[1m Committing to params took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("⬜ \x1b[1m NN Parameters Commitment size: {} B \x1b[0m", par_commit.serialized_size(Compress::Yes));
        println!("========End Committing to Params=========================================");
        println!("************************************************************************");

        self.par_commit = par_commit.clone();
        self.par_commit_cache = par_commit_cache.clone();
    }

    pub fn reset_pointer(&mut self) {
        self.private_trans_reduce.reset_pointer();
        self.fsbatch.reset_pointer();
    }

    pub fn commit_to_witness(&mut self) {


        println!("========Committing to Auxiliaries=========================================");
        let _timer = std::time::Instant::now();
        // Commit only the witness (non-parameter) part: parameters zeroed out
        let (witness_commit, witness_commit_cache) = self.mock_nn.commit_to_leaves_without_pars::<E>(&self.pcsrs);
        println!("🕒 \x1b[1m Committing to auxiliaries took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("========End Committing to Auxiliaries=========================================");

        self.witness_commit = witness_commit.clone();
        self.witness_commit_cache = witness_commit_cache.clone();
    }

    /// Batch leave projections onto points in a point container to the projection on a large vector
    pub fn batch_leaves_proj(
        &mut self,
        trans: &mut Transcript<E::ScalarField>,
        cs_builder: &mut ConstraintSystemBuilder<E::ScalarField>
    ) -> (E::ScalarField, Vec<E::ScalarField>) {


        let mut point_container = PointsContainer::new();
        self.mock_nn_proj.push_leaves_to_point_container(&mut point_container);
        let mut batchproj = BatchProj::<E::ScalarField>::new_from_point_container(point_container);

        println!("========Batch Leaf Projection=========================================");
        let _timer = std::time::Instant::now();
        let mut mat_container = MatContainerMyInt::new();
        self.mock_nn.push_to_mat_container(&mut mat_container);

    // debug checks removed
        
        
        batchproj.set_input_from_matcontainer(mat_container);

        let flag = batchproj.reduce_prover_with_constraint_building(trans, cs_builder);
        
        println!("🕒 \x1b[1m Batch Leaf projection took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("========End Batch Leaf projection======================================");

        assert_eq!(flag, true, "Batch projection in NN failed");

        let proof_size = trans.get_proof_size_in_bytes();
        println!("⬜ \x1b[1m Proof size before PoP compression: {} B \x1b[0m", proof_size);

        println!("************************************************************************");


        let (hat, point) = batchproj.atomic_pop.get_c();
        self.leaf_hat = hat.clone();
        // Keep left challenges for PCS opening; right challenges are also recorded in transcript
        self.leaf_point = point.0.clone();

        // Record the absolute transcript indices for the final exposure (public inputs)
        let mapping = &batchproj.atomic_pop.mapping;
        self.leaf_hat_index = mapping.final_c_hat_index;
        self.leaf_point_index = mapping.final_c_point_index.clone();

    // debug block removed
        

        (hat, point.0)
    }

    // debug helpers removed

    /// Clear large intermediate data structures that are no longer needed
    /// after proof generation and before opening leaf commitment
    pub fn clear_intermediate_memory(&mut self) {
        println!("Clearing intermediate memory before leaf commitment opening...");
        
        // Clear the constraint system builder (contains many intermediate constraints)
        self.cs_builder = ConstraintSystemBuilder::<E::ScalarField>::new();
        
        // Clear the proving key (large circuit data no longer needed after proof generation)
        self.pop_pk = None;
        
        // Clear MockNNProj data (projection data no longer needed)
        self.mock_nn_proj = MockNNProj::<E::ScalarField>::new();
        
        // Note: We cannot clear mock_nn completely as it's needed for open_leaf_commitment
        // to reconstruct the matrix via push_to_mat_container()
        
        println!("Intermediate memory cleared successfully.");
    }

    pub fn open_leaf_commitment(&mut self) {

        println!("========Open Leaf Commitment=========================================");
        let _timer = std::time::Instant::now();

        let leaf_com = self.par_commit.clone() + self.witness_commit.clone();
        let mut leaf_cache = add_vec_g1::<E>(&self.par_commit_cache, &self.witness_commit_cache);

        self.par_commit_cache = Vec::new();
        self.witness_commit_cache = Vec::new();

        self.trans_pcs = self.mock_nn.open_leaf_commitment(
            &self.pcsrs,
            &leaf_com,
            &self.leaf_hat,
            &self.leaf_point,
            &mut leaf_cache,
        );

        println!("🕒 \x1b[1m Open Leaf projection took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("========End Open Leaf projection======================================");
        let pcs_trans_size = self.trans_pcs.serialized_size(Compress::Yes);

        println!("⬜ \x1b[1m PCS Transcript Size: {} B \x1b[0m", pcs_trans_size);

        println!("************************************************************************");

    }

    pub fn verify_leaf_commitment(&self) -> bool {
        // Verify the leaf commitment using the stored values
        println!("========Verify Leaf Commitment=========================================");
        let _timer = std::time::Instant::now();
        
        let leaf_com = self.par_commit.clone() + self.witness_commit.clone();
        let flag = self.mock_nn.verify_leaf_commitment(
            &self.pcsrs,
            &self.trans_pcs,
            leaf_com,
            self.leaf_hat,
            &self.leaf_point,
        );
        println!("😀 \x1b[1m Verify leaf projection result: {} \x1b[0m", flag);
        
        println!("🕒 \x1b[1m Verify Leaf projection took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("========End Verify Leaf projection======================================");

        flag
    }

    pub fn reduce_prover_and_building_pop_circuit(&mut self) {
        let mut trans = Transcript::<E::ScalarField>::default();
        let mut cs_builder = ConstraintSystemBuilder::<E::ScalarField>::new();

        self.reduce_prover_with_constraint_building(&mut trans, &mut cs_builder);
        self.private_trans_reduce = trans;
        self.cs_builder = cs_builder;
    }

    pub fn reduce_prover_with_constraint_building(
        &mut self,
        trans: &mut Transcript<E::ScalarField>,
        cs_builder: &mut ConstraintSystemBuilder<E::ScalarField>
    ) {
        // First, collectively prove activations from all layers
        // The table projections and auxiliary input projctions are final while the input and output vectors are not
        //
        println!("************************************************************************");
        println!("========Backward Proof Reduction=========================================");
        let _timer = std::time::Instant::now();
        let mut activation = Activation::<E::ScalarField>::new();
        activation.reduce_prover_with_constraint_building(
            trans,
            cs_builder,
            &matop::flatten_and_concat(&self.mock_nn.phi_inputs).data[0],
            &matop::flatten_and_concat(&self.mock_nn.layer_outputs).data[0],
            &self.mock_nn.lookup_table,
            &self.mock_nn.lookup_target,
            &self.mock_nn.lookup_table_auxiliary,
            &self.mock_nn.lookup_target_auxiliary,
            &self.mock_nn.range_table_auxiliary,
            &self.mock_nn.range_target_auxiliary,
        );

        let phi_input_vec_proj = PointInfo::<E::ScalarField>::new(
            activation.atomic_pop.phi_input_hat,
            activation.atomic_pop.phi_input_point.clone(),
            activation.atomic_pop.mapping.phi_input_hat_index,
            activation.atomic_pop.mapping.phi_input_point_index.clone(),
        );

        let phi_output_vec_proj = PointInfo::<E::ScalarField>::new(
            activation.atomic_pop.phi_output_hat,
            activation.atomic_pop.phi_output_point.clone(),
            activation.atomic_pop.mapping.phi_output_hat_index,
            activation.atomic_pop.mapping.phi_output_point_index.clone(),
        );

        for i in 0..self.mock_nn.lookup_table.len() {
            self.mock_nn_proj.lookup_table_proj.push(
                PointInfo::<E::ScalarField>::new(
                    activation.atomic_pop.lookup_table_hats[i],
                    activation.atomic_pop.lookup_table_points[i].clone(),
                    activation.atomic_pop.mapping.lookup_table_hats_index[i],
                    activation.atomic_pop.mapping.lookup_table_points_index[i].clone(),
                )
            )
        }

        for i in 0..self.mock_nn.lookup_target.len() {
            self.mock_nn_proj.lookup_target_proj.push(
                PointInfo::<E::ScalarField>::new(
                    activation.atomic_pop.lookup_target_hats[i],
                    activation.atomic_pop.lookup_target_points[i].clone(),
                    activation.atomic_pop.mapping.lookup_target_hats_index[i],
                    activation.atomic_pop.mapping.lookup_target_points_index[i].clone(),
                )
            )
        }

        self.mock_nn_proj.lookup_table_auxiliary_proj =
        PointInfo::<E::ScalarField>::new(
            activation.atomic_pop.lookup_table_auxiliary_hat,
            activation.atomic_pop.lookup_table_auxiliary_points.clone(),
            activation.atomic_pop.mapping.lookup_table_auxiliary_hat_index,
            activation.atomic_pop.mapping.lookup_table_auxiliary_points_index.clone(),
        );

        self.mock_nn_proj.lookup_target_auxiliary_proj =
        PointInfo::<E::ScalarField>::new(
            activation.atomic_pop.lookup_target_auxiliary_hat,
            activation.atomic_pop.lookup_target_auxiliary_points.clone(),
            activation.atomic_pop.mapping.lookup_target_auxiliary_hat_index,
            activation.atomic_pop.mapping.lookup_target_auxiliary_points_index.clone(),
        );

        for i in 0..self.mock_nn.range_table_auxiliary.len() {
            self.mock_nn_proj.range_table_auxiliary_proj.push(
                PointInfo::<E::ScalarField>::new(
                    activation.atomic_pop.range_table_auxiliary_hat[i],
                    activation.atomic_pop.range_table_auxiliary_points[i].clone(),
                    activation.atomic_pop.mapping.range_table_auxiliary_hat_index[i],
                    activation.atomic_pop.mapping.range_table_auxiliary_points_index[i].clone(),
                )
            )
        }

        for i in 0..self.mock_nn.range_target_auxiliary.len() {
            self.mock_nn_proj.range_target_auxiliary_proj.push(
                PointInfo::<E::ScalarField>::new(
                    activation.atomic_pop.range_target_auxiliary_hat[i],
                    activation.atomic_pop.range_target_auxiliary_points[i].clone(),
                    activation.atomic_pop.mapping.range_target_auxiliary_hat_index[i],
                    activation.atomic_pop.mapping.range_target_auxiliary_points_index[i].clone(),
                )
            )
        }

        // Then, we use Concat protocol to transform the projection of phi_input_vec and phi_output_vec to
        // projections of self.phi_inputs and self.layer_outputs

        let mut concat_input = Concat::<E::ScalarField>::new(
            phi_input_vec_proj.hat.clone(),
            phi_input_vec_proj.point.clone(),
            phi_input_vec_proj.hat_index.clone(),
            phi_input_vec_proj.point_index.clone(),
            (self.shape.0, 1),
            self.depth
        );

        let concat_input_mats = self.mock_nn.phi_inputs.iter().map(|mat| {
           matop::myint_to_field_mat::<E::ScalarField>(&mat) 
        }).collect();

        concat_input.set_input(concat_input_mats);
        concat_input.reduce_prover_with_constraint_building(trans, cs_builder);

        for i in 0..self.mock_nn.phi_inputs.len() {
            self.mock_nn_proj.phi_inputs_proj.push(
                PointInfo::<E::ScalarField>::new(
                    concat_input.atomic_pop.hat_inputs[i],
                    concat_input.atomic_pop.point_inputs[i].clone(),
                    concat_input.atomic_pop.mapping.hat_inputs_index[i],
                    concat_input.atomic_pop.mapping.point_inputs_index[i].clone(),
                )
            )
        }

        concat_input.clear(); // Important to release the memory after using the field mats within the protocols


        // Similary for layer_outputs
        let mut concat_output = Concat::<E::ScalarField>::new(
            phi_output_vec_proj.hat.clone(),
            phi_output_vec_proj.point.clone(),
            phi_output_vec_proj.hat_index.clone(),
            phi_output_vec_proj.point_index.clone(),
            (self.shape.0, 1),
            self.depth
        );

        let concat_output_mats = self.mock_nn.layer_outputs.iter().map(|mat| {
           matop::myint_to_field_mat::<E::ScalarField>(&mat) 
        }).collect();

        concat_output.set_input(concat_output_mats);
        concat_output.reduce_prover_with_constraint_building(trans, cs_builder);


        let mut layer_outputs_act_proj = Vec::new();
        for i in 0..self.mock_nn.layer_outputs.len() {
            layer_outputs_act_proj.push(
                PointInfo::<E::ScalarField>::new(
                    concat_output.atomic_pop.hat_inputs[i],
                    concat_output.atomic_pop.point_inputs[i].clone(),
                    concat_output.atomic_pop.mapping.hat_inputs_index[i],
                    concat_output.atomic_pop.mapping.point_inputs_index[i].clone(),
                )
            );
        }


        // ================================================================================================
        // Then, for each layer, run the LinComb and MatMul protocols
        // 
        // Push the fixed coefficients of linear combination to the Transcript for the LinComb protocol to read
        // 
        for i in 0..self.depth {

            let mut lin_comb = LinComb::<E::ScalarField>::new(
                self.mock_nn_proj.phi_inputs_proj[i].hat.clone(),
                self.mock_nn_proj.phi_inputs_proj[i].point.clone(),
                self.mock_nn_proj.phi_inputs_proj[i].hat_index.clone(),
                self.mock_nn_proj.phi_inputs_proj[i].point_index.clone(),
                (self.shape.0, 1),
                2,
                vec![E::ScalarField::one(), E::ScalarField::from(self.mock_nn.get_scale())],
                vec![0, 0],
            );  

            lin_comb.set_input(
                vec![
                    matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.mul_outputs[i]),
                    matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.biases[i]),
                ],
            );

            lin_comb.reduce_prover(trans);
            lin_comb.prepare_atomic_pop_with_constant_coeff(); // The general method is for variable coefficients
            lin_comb.synthesize_atomic_pop_constraints(cs_builder);

            self.mock_nn_proj.mul_outputs_proj.push(
                PointInfo::<E::ScalarField>::new(
                    lin_comb.atomic_pop.hat_inputs[0].clone(),
                    lin_comb.atomic_pop.point_inputs[0].clone(),
                    lin_comb.atomic_pop.mapping.hat_inputs_index[0].clone(),
                    lin_comb.atomic_pop.mapping.point_inputs_index[0].clone(),
                )
            );

            self.mock_nn_proj.biases_proj.push(
                PointInfo::<E::ScalarField>::new(
                    lin_comb.atomic_pop.hat_inputs[1].clone(),
                    lin_comb.atomic_pop.point_inputs[1].clone(),
                    lin_comb.atomic_pop.mapping.hat_inputs_index[1].clone(),
                    lin_comb.atomic_pop.mapping.point_inputs_index[1].clone(),
                )
            );

            lin_comb.clear();

            let mut matmul = MatMul::<E::ScalarField>::new(
                self.mock_nn_proj.mul_outputs_proj[i].hat.clone(),
                self.mock_nn_proj.mul_outputs_proj[i].point.clone(),
                self.mock_nn_proj.mul_outputs_proj[i].hat_index.clone(),
                self.mock_nn_proj.mul_outputs_proj[i].point_index.clone(),
                (self.shape.0, 1),
                (self.shape.0, self.shape.1),
                (self.shape.0, 1),
            );


            let a_mat = matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.weights[i]);
            let b_mat: DenseMatFieldCM<E::ScalarField>;
            if i == 0 {
                b_mat = matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.nn_input);
            } else {
                b_mat = matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.layer_outputs[i-1]);
            }

            matmul.set_input(a_mat, b_mat.clone());
            matmul.reduce_prover_with_constraint_building(
                trans,
                cs_builder
            );

            let (a_hat, a_point) = matmul.atomic_pop.get_a();
            let (a_hat_index, a_point_index) = matmul.atomic_pop.get_a_index();
            let (b_hat, b_point) = matmul.atomic_pop.get_b();
            let (b_hat_index, b_point_index) = matmul.atomic_pop.get_b_index();

            self.mock_nn_proj.weights_proj.push(
                PointInfo::<E::ScalarField>::new(
                    a_hat,
                    a_point,
                    a_hat_index,
                    a_point_index,
                )
            );
            if i == 0 {
                self.mock_nn_proj.nn_input_proj = PointInfo::<E::ScalarField>::new(
                    b_hat,
                    b_point,
                    b_hat_index,
                    b_point_index,
                );
            } else {
                // Batch the projections of the same matrix layer_outputs[i] with that from the activation protocol
                let mut batchpoint = BatchPoint::new(
                    vec![b_hat, layer_outputs_act_proj[i-1].hat.clone()],
                    vec![b_point, layer_outputs_act_proj[i-1].point.clone()],
                    vec![b_hat_index, layer_outputs_act_proj[i-1].hat_index.clone()],
                    vec![b_point_index, layer_outputs_act_proj[i-1].point_index.clone()],
                );
                batchpoint.set_input(b_mat);
                batchpoint.reduce_prover_with_constraint_building(
                    trans,
                    cs_builder
                );

                self.mock_nn_proj.layer_outputs_proj.push(
                    PointInfo::<E::ScalarField>::new(
                        batchpoint.atomic_pop.c_hat.clone(),
                        batchpoint.atomic_pop.c_point.clone(),
                        batchpoint.atomic_pop.mapping.c_hat_index.clone(),
                        batchpoint.atomic_pop.mapping.c_point_index.clone(),
                    )
                );
            }
        } // end of For loop

        // The last layer output equals the nn_output
        let mut mateq = MatEq::new((self.shape.0, 1));
        let a_mat = matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.nn_output);
        let b_mat = matop::myint_to_field_mat::<E::ScalarField>(&self.mock_nn.layer_outputs[self.depth-1]);
        mateq.set_input(a_mat, b_mat.clone());
        mateq.reduce_prover_with_constraint_building(
            trans,
            cs_builder
        );

        let (a_hat, a_point) = mateq.atomic_pop.get_a();
        let (a_hat_index, a_point_index) = mateq.atomic_pop.get_a_index();
        let (b_hat, b_point) = mateq.atomic_pop.get_b();
        let (b_hat_index, b_point_index) = mateq.atomic_pop.get_b_index();

        self.mock_nn_proj.nn_output_proj = PointInfo::<E::ScalarField>::new(
            a_hat,
            a_point,
            a_hat_index,
            a_point_index,
        );

        let mut batchpoint = BatchPoint::new(
            vec![b_hat, layer_outputs_act_proj[self.depth-1].hat.clone()],
            vec![b_point, layer_outputs_act_proj[self.depth-1].point.clone()],
            vec![b_hat_index, layer_outputs_act_proj[self.depth-1].hat_index.clone()],
            vec![b_point_index, layer_outputs_act_proj[self.depth-1].point_index.clone()],
        );
        batchpoint.set_input(b_mat);
        batchpoint.reduce_prover_with_constraint_building(
            trans,
            cs_builder
        );

        self.mock_nn_proj.layer_outputs_proj.push(
            PointInfo::<E::ScalarField>::new(
                batchpoint.atomic_pop.c_hat.clone(),
                batchpoint.atomic_pop.c_point.clone(),
                batchpoint.atomic_pop.mapping.c_hat_index.clone(),
                batchpoint.atomic_pop.mapping.c_point_index.clone(),
            )
        );


    // debug prints removed
    
    
        // ======================================================================================
        // We have obtained projections of all leaf NN nodes
        // Now we need to batch them into a single projection for the entire NN
        //
        println!("🕒 \x1b[1m Reduce prover took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        self.batch_leaves_proj(trans, cs_builder);
        // debug prints removed
    
    }

    pub fn commit_to_pop_circuits(&mut self) {
         // The commitment to the pop circuit is the vk of the Groth16 circuit
        println!("Committing to pop circuit (representing NN structure)..");
        // IMPORTANT: Groth16 setup clones the builder and immediately calls generate_constraints.
        // Our ConstraintSystemBuilder::generate_constraints currently returns Unsatisfiable if BOTH
        // num_pub_inputs == 0 AND num_pri_inputs == 0. After reduction we have not yet populated
        // inputs on self.cs_builder; we only do the split in gen_pop_proof previously. So we must
        // pre-populate public & private inputs NOW (using the same split logic as proof generation)
        // so that setup succeeds and the circuit shape is fixed.

        // Derive public inputs (leaf projection) & private inputs.
        // IMPORTANT: Constraints reference transcript values via PriInput indices corresponding to
        // the ORIGINAL full transcript ordering. 
        let _timer = std::time::Instant::now();
        let mut _min_pub_idx = self.leaf_hat_index;
       
        let public_inputs = self.private_trans_reduce.get_trans_seq()[_min_pub_idx..].to_vec();
        let private_inputs = self.private_trans_reduce.get_trans_seq()[.._min_pub_idx].to_vec();
        
        // Populate the builder with concrete inputs so generate_constraints sees non-zero lengths
        self.cs_builder
            .set_public_inputs(public_inputs)
            .set_private_inputs(private_inputs);

        // removed optional debug validation and scanning

        let mut rng = thread_rng();
        let (pk, vk) = Groth16Prover::setup(&self.cs_builder.clone(), &mut rng).expect("groth setup failed");

        self.pop_pk = Some(pk);
        self.pop_circ_commit = Groth16Prover::prepare_verifying_key(&vk);

        println!("📝 \x1b[1m Pop circuit commitment size: {} bytes \x1b[0m", self.pop_circ_commit.serialized_size(Compress::Yes));
        println!("🕒 \x1b[1m Pop circuit commitment took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
    }

    pub fn gen_pop_proof(&mut self) {
        println!("Generating PoP proof via Groth16..");
        let _timer = std::time::Instant::now();

        // Construct the same sparse public vector using recorded indices
        let mut _min_pub_idx = self.leaf_hat_index;

        // Use transcript sequence to preserve PriInput index mapping
        let public_input = self.private_trans_reduce.get_trans_seq()[_min_pub_idx..].to_vec();
        let private_input = self.private_trans_reduce.get_trans_seq()[.._min_pub_idx].to_vec();

        let mut rng = thread_rng();

        // Lazily setup pk if missing to avoid unwrap panic in tests that skip commit_to_pop_circuits
        if self.pop_pk.is_none() {
            // Ensure builder carries consistent inputs before setup
            self.cs_builder
                .set_public_inputs(public_input.clone())
                .set_private_inputs(private_input.clone());
            let (pk, _vk) = Groth16Prover::setup(&self.cs_builder.clone(), &mut rng).expect("groth setup (lazy) failed");
            self.pop_pk = Some(pk);
        }

        let proof = Groth16Prover::prove_with_pub_pri(
            self.pop_pk.as_ref().expect("pop pk missing after setup"),
            self.cs_builder.clone(),
            public_input,
            private_input,
            &mut rng)
        .expect("proof generation failed");

        let proof_size = proof.serialized_size(Compress::Yes);

        println!("📝 \x1b[1m PoP proof size after compression: {} bytes \x1b[0m", proof_size);
        println!("🕒 \x1b[1m PoP proof generation took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());

        self.clear_intermediate_memory();

        self.trans_pop = proof;
    }

    pub fn verify_pop_proof(&self) -> bool {
        // Verify the proof using the stored values

        println!("========Verify PoP Proof=========================================");
        let _timer = std::time::Instant::now();
        // Build the same sparse public vector deterministically from indices and transcript
       
        let mut public_input = vec![self.leaf_hat.clone()];
        for idx in 0..self.leaf_point.len() { public_input.push(self.leaf_point[idx].clone()); }

        let flag = Groth16Prover::verify(
            &self.pop_circ_commit,
            &public_input,
            &self.trans_pop,
        ).expect("verification failed");

        println!("😀 \x1b[1m Verify PoP result: {} \x1b[0m", flag);
        println!("🕒 \x1b[1m PoP proof verification took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        flag
    }


    pub fn get_nn_commitment_size(&self) -> usize {
        let par_commit_size = self.par_commit.serialized_size(Compress::Yes);
        let pop_circ_commit_size = self.pop_circ_commit.serialized_size(Compress::Yes);
        
        par_commit_size + pop_circ_commit_size
    }

    pub fn get_compressed_proof_size(&self) -> usize {

        let witness_com_size = self.witness_commit.serialized_size(Compress::Yes);
        let trans_pop_size = self.trans_pop.serialized_size(Compress::Yes);
        let fs_r1cs_commit_size = self.fsbatch.groth_r1cs_comm.serialized_size(Compress::Yes);
        let fsbatch_proof_size = self.fsbatch.get_trans_size();
        let trans_pcs_size = self.trans_pcs.serialized_size(Compress::Yes);
        witness_com_size + trans_pop_size + fs_r1cs_commit_size + fsbatch_proof_size + trans_pcs_size
    }

    pub fn prove_fs(&mut self) {
        self.mock_nn.clear();
        self.cs_builder = ConstraintSystemBuilder::new();

        println!("Proving the FS transform...");
        let _timer1 = std::time::Instant::now();
        
        let proof_len = self.private_trans_reduce.get_fs_proof_vec().len();
        self.fsbatch = FSBatchGroth::<E>::new(proof_len);
        self.fsbatch.commit_to_r1cs_mat();
        let _ = self.fsbatch.commit_to_transcript(&self.pcsrs, &self.private_trans_reduce);
        self.fsbatch.prove_r1cs_constraints(&self.pcsrs);

        println!("🕒 \x1b[1m FS transform proving took {:.6} seconds \x1b[0m", _timer1.elapsed().as_secs_f64());
    }


    pub fn verify(&mut self) -> bool {
        self.reset_pointer();
        println!("*******************************************************************");
        println!("========Verify NN Protocol=========================================");
        let _timer = std::time::Instant::now();

        let flag_1 = self.verify_leaf_commitment();
        let flag_2 = self.verify_pop_proof();
        self.fsbatch.reset_pointer();
        let flag_3 = self.fsbatch.verify_r1cs_constraints(&self.pcsrs);

        println!("Verify leaf commitment result: {}", flag_1);
        println!("Verify PoP result: {}", flag_2);
        println!("Verify FSBatch result: {}", flag_3);

        println!("***********************************************************************");
        println!("😀 \x1b[1m NN protocol verification result: {} \x1b[0m", flag_1 && flag_2 && flag_3);
        println!("⬜  \x1b[1m Total proof size: {} bytes \x1b[0m", self.get_compressed_proof_size());
        println!("🕒 \x1b[1m NN protocol verification took {:.6} seconds \x1b[0m", _timer.elapsed().as_secs_f64());
        println!("*******************************************************************");

        flag_1 && flag_2 && flag_3

    }

}


/// Create a dummy point container for testing BatchProj
pub fn dummy_point_container<F:PrimeField>(mock_nn: &MockNN<F>) -> PointsContainer<F> {
    fn dummy_point<F:PrimeField>(mat: &DenseMatCM<MyInt,F>) -> (F, (Vec<F>, Vec<F>), usize, (Vec<usize>, Vec<usize>)) {
        let log_m = mat.shape.0.ilog2() as usize;
        let log_n = mat.shape.1.ilog2() as usize;
    let hat = F::from(mat.data[0][0]);
    (hat, (vec![F::zero(); log_m], vec![F::zero(); log_n]), 0, (vec![0; log_m], vec![0; log_n]))
    }

    let mut point_container = PointsContainer::<F>::new();

    for w in &mock_nn.weights { let (h,p,hi,pi)=dummy_point(w); point_container.push(h,p,hi,pi); }
    for b in &mock_nn.biases { let (h,p,hi,pi)=dummy_point(b); point_container.push(h,p,hi,pi); }
    let (h,p,hi,pi)=dummy_point(&mock_nn.nn_input); point_container.push(h,p,hi,pi);
    let (h1,p1,hi1,pi1)=dummy_point(&mock_nn.nn_output); point_container.push(h1,p1,hi1,pi1);

    for val in &mock_nn.lookup_table { let m = DenseMatCM::<MyInt,F>::from_data(vec![val.clone()]); let (h,p,hi,pi)=dummy_point(&m); point_container.push(h,p,hi,pi); }
    // DEBUG: removed length consistency check
    for val in &mock_nn.range_target_auxiliary { let m = DenseMatCM::<MyInt,F>::from_data(vec![val.clone()]); let (h,p,hi,pi)=dummy_point(&m); point_container.push(h,p,hi,pi); }

    point_container
}



#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381 as E;
    use ark_std::ops::Add; // for PairingOutput


    #[test]
    fn test_nnprotocol() {

        let depth = 16;
        let shape = (128, 128);
        let _nndummy: ProtocolNN<E> = ProtocolNN::new(depth, shape);

        // nndummy.reduce_prover_and_building_pop_circuit();
        // // Commit to parameters first
        // nndummy.commit_to_pop_circuits();
        // nndummy.setup_fsbatch();

        
        // let pcsrs = nndummy.pcsrs.clone();
        // let pop_circ_commit = nndummy.pop_circ_commit.clone();
        // let pop_pk = nndummy.pop_pk.clone();
        // let fs_len = nndummy.fsbatch.proof_len.clone();
        // let r1cs_mat_shape = nndummy.fsbatch.r1cs_mat_shape.clone();
        // let groth_r1cs_comm = nndummy.fsbatch.groth_r1cs_comm.clone();
        // let groth_r1cs_pk = nndummy.fsbatch.groth_r1cs_pk.clone();
        


        let mut nn: ProtocolNN<E> = ProtocolNN::new(depth, shape);
        // Capture parameter-only flattened leaves BEFORE filling internals
        let _ = nn.commit_to_pars();

        let _ = nn.commit_to_witness();
    
       

        // nn.pcsrs = pcsrs;
        // nn.pop_circ_commit = pop_circ_commit;
        // nn.pop_pk = pop_pk;
        // nn.fsbatch.proof_len = fs_len;
        // nn.fsbatch.r1cs_mat_shape = r1cs_mat_shape;
        // nn.fsbatch.groth_r1cs_comm = groth_r1cs_comm;
        // nn.fsbatch.groth_r1cs_pk = groth_r1cs_pk;


        let par_com = nn.par_commit.clone();
        let par_commit_cache = nn.par_commit_cache.clone();
        let witness_com = nn.witness_commit.clone();
        let witness_commit_cache = nn.witness_commit_cache.clone();

        let (leaves_com, leaves_com_cache) = nn.mock_nn.commit_to_leaves(&nn.pcsrs);

        
        assert_eq!(par_com.add(&witness_com), leaves_com, "Commit additive property failed");
        assert_eq!(add_vec_g1::<E>(&par_commit_cache, &witness_commit_cache), leaves_com_cache);
        

        nn.reduce_prover_and_building_pop_circuit();
        
        nn.commit_to_pop_circuits();
        nn.gen_pop_proof();

        assert_eq!(nn.depth, depth);
        assert_eq!(nn.shape, shape);

  
        nn.open_leaf_commitment();

        nn.reset_pointer();
        nn.verify_leaf_commitment();
        nn.verify_pop_proof();
     

    }
}