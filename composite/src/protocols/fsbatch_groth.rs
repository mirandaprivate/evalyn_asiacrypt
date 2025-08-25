//! Verify that the committed Transcript is correctly produced from Fiat-Shamir transform
//! 
use ark_ff::{PrimeField, UniformRand, Zero, One};
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::CanonicalSerialize;

use ark_poly_commit::smart_pc::SmartPC;
use ark_poly_commit::smart_pc::data_structures::{
    Trans as PcsTrans,
    UniversalParams as PcsPP,
};

use atomic_proof::{
    MLPCS as GrothPC,
    MLPCSProverKey as GrothProverKey,
    MLPCSCommitment as GrothCommitment,
    MLPCSProof as GrothProof,
};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use atomic_proof::protocols::{BatchPoint, BatchProjField, Hadamard, MatMul, MatSub, EqZero};
use atomic_proof::AtomicMatProtocol;

use fsproof::{BatchConstraints, Transcript};

use mat::DenseMatFieldCM;
 
 


#[derive(Clone)]
pub struct FSBatchGroth<E: Pairing>
where
    E: Pairing,
    E::ScalarField: Absorb + UniformRand + PrimeField,
{
    pub proof_len: usize,
    pub r1cs_mat_shape: (usize, usize),
    pub proof_vec: Vec<E::ScalarField>,
    pub state_vec: Vec<E::ScalarField>,
    pub state_vec_shift: Vec<E::ScalarField>,
    // SmartPC only used for witness; r1cs & vectors use GrothPC
    pub witness_commit: PairingOutput<E>,
    pub r1cs_prover_input: BatchConstraints<E::ScalarField>,
    pub witness_commit_cache: Vec<E::G1>,
    pub reduce_trans: Transcript<E::ScalarField>,
    pub pcs_trans_witness: PcsTrans<E>,
    pub groth_r1cs_pk: Option<GrothProverKey<E>>,
    pub groth_r1cs_comm: GrothCommitment<E>,
    pub groth_r1cs_proof: GrothProof<E>,
    pub groth_r1cs_eval: E::ScalarField,
    pub groth_state_pk: Option<GrothProverKey<E>>,
    pub groth_state_comm: GrothCommitment<E>,
    pub groth_state_proof: GrothProof<E>,
    pub groth_state_eval: E::ScalarField,
    pub groth_state_shift_pk: Option<GrothProverKey<E>>,
    pub groth_state_shift_comm: GrothCommitment<E>,
    pub groth_state_shift_proof: GrothProof<E>,
    pub groth_state_shift_eval: E::ScalarField,
    pub groth_proof_pk: Option<GrothProverKey<E>>,
    pub groth_proof_comm: GrothCommitment<E>,
    pub groth_proof_proof: GrothProof<E>,
    pub groth_proof_eval: E::ScalarField,
    // duplicate block removed
}


impl<E: Pairing> FSBatchGroth<E> 
where 
    E::ScalarField: Absorb + UniformRand + PrimeField,
{

    pub fn new(proof_len: usize) -> Self  {
        FSBatchGroth {
            proof_len,
            r1cs_mat_shape: (0, 0),
            proof_vec: Vec::new(),
            state_vec: Vec::new(),
            state_vec_shift: Vec::new(),
            witness_commit: PairingOutput::default(),
            r1cs_prover_input: BatchConstraints::<E::ScalarField>::new(),
            reduce_trans: Transcript::new(E::ScalarField::zero()),
            pcs_trans_witness: PcsTrans::new(),
            groth_r1cs_proof: GrothProof::new(),
            groth_state_proof: GrothProof::new(),
            groth_proof_proof: GrothProof::new(),
            witness_commit_cache: Vec::new(),
            groth_r1cs_pk: None,
            groth_r1cs_comm: GrothCommitment::new(),
            groth_r1cs_eval: E::ScalarField::zero(),
            groth_state_pk: None,
            groth_state_comm: GrothCommitment::new(),
            groth_state_eval: E::ScalarField::zero(),
            groth_state_shift_pk: None,
            groth_state_shift_comm: GrothCommitment::new(),
            groth_state_shift_proof: GrothProof::new(),
            groth_state_shift_eval: E::ScalarField::zero(),
            groth_proof_pk: None,
            groth_proof_comm: GrothCommitment::new(),
            groth_proof_eval: E::ScalarField::zero(),
        }
    }

    pub fn default() -> Self {
        Self::new(0)
    }

    pub fn set_commitments(
        &mut self,
        _proof_vec_commit: GrothCommitment<E>,
        _state_vec_commit: GrothCommitment<E>,
        _state_vec_shift_commit: GrothCommitment<E>,
        witness_commit: PairingOutput<E>,
    ) {
        self.witness_commit = witness_commit;
    }

    pub fn commit_to_r1cs_mat(&mut self) -> GrothCommitment<E> {
        let (a_mat, b_mat, c_mat) = self.r1cs_prover_input.gen_r1cs_constraints();
        self.r1cs_mat_shape = a_mat.shape.clone();
        
        // GrothPC commit: column-major A||B||C||C padded to power-of-two dimensions
    let cols = a_mat.shape.1; // rows unused
        
    let mut col_concat = Vec::new();
        for c in 0..cols { col_concat.extend(a_mat.data[c].clone()); }
        for c in 0..cols { col_concat.extend(b_mat.data[c].clone()); }
        for c in 0..cols { col_concat.extend(c_mat.data[c].clone()); }
        for c in 0..cols { col_concat.extend(c_mat.data[c].clone()); }
        
        let mut rng = StdRng::seed_from_u64(2024);
        let (pk, comm) = GrothPC::commit::<E,_>(&vec![col_concat], &mut rng).expect("Groth r1cs commit failed");

        // println!("pk l_bits {}, r_bits {}", pk.l_bits, pk.r_bits);

        self.groth_r1cs_pk = Some(pk);
        self.groth_r1cs_comm = comm.clone();

        comm
    }

    pub fn commit_to_transcript(
        &mut self,
        pcsrs: &PcsPP<E>,
        trans: &Transcript<E::ScalarField>
    ) -> (
        GrothCommitment<E>,
        GrothCommitment<E>,
        GrothCommitment<E>,
        PairingOutput<E>
    ) {
        let fs_trans = trans.fs.clone();
        self.r1cs_prover_input.prepare_from_transcript(&fs_trans);

    // Obtain original witness in column-major (collection of column vectors)
        let mut witness = self.r1cs_prover_input.witness_mat.data.clone();
        if !witness.is_empty() {
            let rows = witness[0].len();
            let cols = witness.len();
            fn is_pow2(x: usize) -> bool { x.is_power_of_two() }
            fn next_pow2(x: usize) -> usize { if x<=1 { x } else { x.next_power_of_two() } }
            if !is_pow2(cols) {
                let target = next_pow2(cols);
                println!("[FSBatch::commit_to_transcript] pad witness cols {} -> {} (rows={})", cols, target, rows);
                let zero_col = vec![E::ScalarField::zero(); rows];
                while witness.len() < target { witness.push(zero_col.clone()); }
                // Write back padded columns into BatchConstraints to keep later multiplications consistent
                self.r1cs_prover_input.witness_mat.data = witness.clone();
                self.r1cs_prover_input.witness_mat.shape = (rows, target);
            }
        }

    // Debug info and dimension check (avoid panic inside SmartPC::commit_full)
        let w_n = witness.len();
        let w_m = if w_n > 0 { witness[0].len() } else { 0 };
        println!("[FSBatch::commit_to_transcript] witness dims n={}, m={}, q={} (qlog≈{:.0})", w_n, w_m, pcsrs.q, (pcsrs.q as f64).log2());
        if w_n * w_m >= pcsrs.q * pcsrs.q {
            panic!(
                "Witness dimensions (n={}, m={}) exceed PCS bound q={} (increase qlog to >= 2^{} / current qlog≈{:.0})",
                w_n, w_m, pcsrs.q, (w_n * w_m ).ilog2()/2 + 1, (pcsrs.q as f64).log2()
            );
        }

        let proof_vec_raw = trans.get_fs_proof_vec();
        let state_vec_raw = trans.get_fs_state_vec();

    // Compute next power-of-two sizes for vectors
        fn next_pow2_u(x: usize) -> usize { if x <= 1 { x } else { x.next_power_of_two() } }
        let target_pv = next_pow2_u(proof_vec_raw.len());
        let target_sv = next_pow2_u(state_vec_raw.len());

        let mut proof_vec = proof_vec_raw.clone();
        let mut state_vec = state_vec_raw.clone();
        let initial_state = trans.fs.get_initial_state();
        let mut state_vec_shift = vec![initial_state];
        state_vec_shift.extend(state_vec_raw[..state_vec_raw.len() - 1].to_vec());


        if target_pv > proof_vec.len() { proof_vec.resize(target_pv, E::ScalarField::zero()); }
        if target_sv > state_vec.len() {
            state_vec.resize(target_sv, E::ScalarField::zero());
            state_vec_shift.resize(target_sv, E::ScalarField::zero());
        }
        // if target_pv != proof_vec_raw.len() { println!("[FSBatch::commit_to_transcript] pad proof_vec {} -> {}", proof_vec_raw.len(), target_pv); }
        // if target_sv != state_vec_raw.len() { println!("[FSBatch::commit_to_transcript] pad state_vec {} -> {}", state_vec_raw.len(), target_sv); }

    // Update proof_len: use padded length (including initial zero element)
        self.proof_len = proof_vec.len();
        self.proof_vec = proof_vec.clone();
        self.state_vec = state_vec.clone();
        self.state_vec_shift = state_vec_shift.clone();

        assert_eq!(proof_vec.len(), w_n, "Proof Length and witness length does not match");
        // Removed SmartPC combined commitment for (state_vec, state_vec_shift, proof_vec) – replaced by GrothPC individual commits
        // For vector -> column-major single-column matrix
        let mut rng = StdRng::seed_from_u64(2025);
        let (pk_state, comm_state) = GrothPC::commit::<E,_>(
            &vec![self.state_vec.clone()],
            &mut rng).expect("Groth commit state_vec failed");
        let (pk_shift, comm_shift) = GrothPC::commit::<E,_>(
            &vec![self.state_vec_shift.clone()],
            &mut rng
        ).expect("Groth commit state_vec_shift failed");
        let (pk_proof, comm_proof) = GrothPC::commit::<E,_>(
            &vec![self.proof_vec.clone()],
            &mut rng
        ).expect("Groth commit proof_vec failed");
        self.groth_state_pk = Some(pk_state);
        self.groth_state_comm = comm_state;
        self.groth_state_shift_pk = Some(pk_shift);
        self.groth_state_shift_comm = comm_shift;
        self.groth_proof_pk = Some(pk_proof);
        self.groth_proof_comm = comm_proof;

        let (witness_com, witness_com_cache) = SmartPC::<E>::commit_square_full(
            pcsrs, &witness, E::ScalarField::zero()
        ).expect("commit_full failed for witness");

        self.witness_commit = witness_com;
        self.witness_commit_cache = witness_com_cache;


    (
        self.groth_state_comm.clone(),
        self.groth_state_shift_comm.clone(),
        self.groth_proof_comm.clone(),
        witness_com,
    )
    
    }

    // Prove the R1CS constraints  (Aw) ∘ (Bw) = Cw
    // using the sparse-dense multiplication and Hadamard product helper functions
    pub fn prove_r1cs_constraints(
        &mut self,
        pcsrs: &PcsPP<E>,
    ) -> bool {
        if !self.r1cs_prover_input.is_ready() {
            panic!("R1CS prover input is not ready in FSBatch");
        }

        let w = self.r1cs_prover_input.witness_mat.clone();

        let aw = self.r1cs_prover_input.a_r1cs.par_mul(&w);
        let bw = self.r1cs_prover_input.b_r1cs.par_mul(&w);
        let cw = self.r1cs_prover_input.c_r1cs.par_mul(&w);

        let mut eqzero = EqZero::new(cw.shape);
        eqzero.reduce_prover(&mut self.reduce_trans);
        let (c_hat, c_point) = eqzero.atomic_pop.get_a();
        let (c_hat_index, c_point_index) = eqzero.atomic_pop.get_a_index();

        // Pass their hat values and point values of (Aw) ∘ (Bw) - Cw and their indices to the sub protocol
        let mut sub = MatSub::new(
            c_hat.clone(),
            c_point.clone(),
            c_hat_index,
            c_point_index.clone(),
            cw.shape.clone(),
            cw.shape.clone(),
            cw.shape.clone(),
        );
        
        // Compute (Aw) ∘ (Bw) for the subtraction
        let hadamard_result = aw.hadamard_prod(&bw);
        
        sub.set_input(hadamard_result.clone(), cw.clone());
        sub.reduce_prover(&mut self.reduce_trans);
        sub.clear();
        let (c_hat_hadamard, c_point_hadamard) = sub.atomic_pop.get_a();
        let (c_hat_index_hadamard, c_point_index_hadamard) = sub.atomic_pop.get_a_index();
        let (c_hat_mul_c, c_point_mul_c) = sub.atomic_pop.get_b();
        let (c_hat_index_mul_c, c_point_index_mul_c) = sub.atomic_pop.get_b_index();

        // Pass the hat values and point values of (Aw) ∘ (Bw) and their indices to the Hadamard protocol
        let mut hadamard = Hadamard::new(
            c_hat_hadamard,
            c_point_hadamard,
            c_hat_index_hadamard,
            c_point_index_hadamard,
            hadamard_result.shape.clone(),
            aw.shape.clone(),
            bw.shape.clone(),
        );
        hadamard.set_input(aw.clone(), bw.clone());
        hadamard.reduce_prover(&mut self.reduce_trans);
        hadamard.clear();
        let (c_hat_mul_a, c_point_mul_a) = hadamard.atomic_pop.get_a();
        let (c_hat_index_mul_a, c_point_index_mul_a) = hadamard.atomic_pop.get_a_index();
        let (c_hat_mul_b, c_point_mul_b) = hadamard.atomic_pop.get_b();
        let (c_hat_index_mul_b, c_point_index_mul_b) = hadamard.atomic_pop.get_b_index();

        // Pass the hat values and point values of (Aw), (Bw), (Cw) and their indices to the multiplication step
        let mut mul_a = MatMul::new(
            c_hat_mul_a,
            c_point_mul_a,
            c_hat_index_mul_a,
            c_point_index_mul_a,
            aw.shape.clone(),
            self.r1cs_prover_input.a_r1cs.shape.clone(),
            self.r1cs_prover_input.witness_mat.shape.clone(),
        );
        mul_a.set_input(self.r1cs_prover_input.a_r1cs.clone(), w.clone());
        mul_a.reduce_prover(&mut self.reduce_trans);
        mul_a.clear();
        let (a_hat, a_point) = mul_a.atomic_pop.get_a();
        let (a_hat_index, a_point_index) = mul_a.atomic_pop.get_a_index();
        let (w_hat_1, w_point_1) = mul_a.atomic_pop.get_b();
        let (w_hat_index_1, w_point_index_1) = mul_a.atomic_pop.get_b_index();

        let mut mul_b = MatMul::new(
            c_hat_mul_b,
            c_point_mul_b,
            c_hat_index_mul_b,
            c_point_index_mul_b,
            bw.shape.clone(),
            self.r1cs_prover_input.b_r1cs.shape.clone(),
            self.r1cs_prover_input.witness_mat.shape.clone(),
        );
        mul_b.set_input(self.r1cs_prover_input.b_r1cs.clone(), w.clone());
        mul_b.reduce_prover(&mut self.reduce_trans);
        mul_b.clear();
        let (b_hat, b_point) = mul_b.atomic_pop.get_a();
        let (b_hat_index, b_point_index) = mul_b.atomic_pop.get_a_index();
        let (w_hat_2, w_point_2) = mul_b.atomic_pop.get_b();
        let (w_hat_index_2, w_point_index_2) = mul_b.atomic_pop.get_b_index();

        let mut mul_c = MatMul::new(
            c_hat_mul_c,
            c_point_mul_c,
            c_hat_index_mul_c,
            c_point_index_mul_c,
            cw.shape.clone(),
            self.r1cs_prover_input.c_r1cs.shape.clone(),
            self.r1cs_prover_input.witness_mat.shape.clone(),
        );
        mul_c.set_input(self.r1cs_prover_input.c_r1cs.clone(), w.clone());
        mul_c.reduce_prover(&mut self.reduce_trans);
        mul_c.clear();
        let (c_hat_final, c_point_final) = mul_c.atomic_pop.get_a();
        let (c_hat_index_final, c_point_index_final) = mul_c.atomic_pop.get_a_index();
        let (w_hat_3, w_point_3) = mul_c.atomic_pop.get_b();
        let (w_hat_index_3, w_point_index_3) = mul_c.atomic_pop.get_b_index();


        // Prove the following matrix multiplication
        //
        //        |  state_vec          |         |  0 1 0 0 ... 0  |       
        //        |  proof_vec          |    =    |  0 0 1 0 ... 0  |   *   W
        //        |  state_vec_shifted  |         |  0 0 0 1 ... 0  |       
        //        |  0                  |         |  0 0 0 0 ... 0  |
        //
        let num_rows_witness = self.r1cs_prover_input.witness_mat.shape.0;
        let mut selection_mat_data = vec![vec![E::ScalarField::zero(); 4]; num_rows_witness];
        selection_mat_data[1][0] = E::ScalarField::one();
        selection_mat_data[2][1] = E::ScalarField::one();
        selection_mat_data[3][2] = E::ScalarField::one();
        let mut selection_mat = DenseMatFieldCM::<E::ScalarField>::new(4,num_rows_witness);
        selection_mat.data = selection_mat_data;

        let zero_vec = vec![E::ScalarField::zero(); self.proof_vec.len()];
        let proof_state_mat_data = vec![self.state_vec.clone(), self.state_vec_shift.clone(), self.proof_vec.clone(), zero_vec];
        let mut proof_state_mat = DenseMatFieldCM::<E::ScalarField>::new(self.proof_vec.len(),4);
        proof_state_mat.data = proof_state_mat_data;

        let mut eqzero_proof_state = EqZero::new(proof_state_mat.shape);
        eqzero_proof_state.reduce_prover(&mut self.reduce_trans);
        let (proof_state_hat_zero, proof_state_point_zero) = eqzero_proof_state.atomic_pop.get_a();
        let (proof_state_hat_index_zero, proof_state_point_index_zero) = eqzero_proof_state.atomic_pop.get_a_index();

        let mut sub_selection = MatSub::new(
            proof_state_hat_zero,
            proof_state_point_zero,
            proof_state_hat_index_zero,
            proof_state_point_index_zero,
            proof_state_mat.shape.clone(),
            proof_state_mat.shape.clone(),
            proof_state_mat.shape.clone(),
        );
        sub_selection.set_input(
            proof_state_mat.clone(),
            proof_state_mat.clone()
        );
        sub_selection.reduce_prover(
            &mut self.reduce_trans
        );
        let (proof_state_hat, proof_state_point_raw) = sub_selection.atomic_pop.get_a();
        let (proof_state_hat_index, proof_state_point_index_raw) = sub_selection.atomic_pop.get_a_index();
        
        let proof_state_point = (
            proof_state_point_raw.1.clone(),
            proof_state_point_raw.0.clone()
        );
        let proof_state_point_index = (
            proof_state_point_index_raw.1.clone(),
            proof_state_point_index_raw.0.clone()
        );

        let mut mul_selection = MatMul::new(
            proof_state_hat,
            proof_state_point.clone(),
            proof_state_hat_index,
            proof_state_point_index,
            (selection_mat.shape.0, self.r1cs_prover_input.witness_mat.shape.1),
            selection_mat.shape.clone(),
            self.r1cs_prover_input.witness_mat.shape.clone(),
        );
        mul_selection.set_input(selection_mat, self.r1cs_prover_input.witness_mat.clone());
        mul_selection.reduce_prover(&mut self.reduce_trans);
        let (selection_hat, selection_point) = mul_selection.atomic_pop.get_a();
        let (w_hat_selection, w_point_selection) = mul_selection.atomic_pop.get_b();
        let (w_hat_selection_index, w_point_selection_index) = mul_selection.atomic_pop.get_b_index();

        let xl1 = selection_point.0[selection_point.0.len()-1];
        let xl2 = selection_point.0[selection_point.0.len()-2];
        let xr1 = selection_point.1[selection_point.1.len()-1];
        let xr2 = selection_point.1[selection_point.1.len()-2];
        let selection_hat_expected = xr1 + xl1 * xr2 + xl2 * xr1 * xr2;
        assert_eq!(selection_hat, selection_hat_expected, "!! Selection matrix projection failed!!");

        // (Groth openings deferred until after r1cs batch_proj)
        // Pass the hat values of point values of w to the batchpoints step to yield a single projection of w
        let mut batch_points = BatchPoint::new(
            vec![w_hat_1, w_hat_2, w_hat_3, w_hat_selection],
            vec![w_point_1, w_point_2, w_point_3, w_point_selection],
            vec![w_hat_index_1, w_hat_index_2, w_hat_index_3, w_hat_selection_index],
            vec![w_point_index_1, w_point_index_2, w_point_index_3, w_point_selection_index],
        );
        batch_points.set_input(w.clone());
        batch_points.reduce_prover(&mut self.reduce_trans);
        batch_points.clear();
        let (hat_w, point_w) = (batch_points.atomic_pop.c_hat, batch_points.atomic_pop.c_point.clone());
        let (_hat_w_index, _point_w_index) = (
            batch_points.atomic_pop.mapping.c_hat_index,
            batch_points.atomic_pop.mapping.c_point_index.clone()
        );

        // // DEBUG
        // // DEBUG
        // //
        // let a_hat_expected = self.r1cs_prover_input.a_r1cs.proj_lr_challenges(&a_point.0, &a_point.1);
        // assert_eq!(a_hat, a_hat_expected, "a_hat does not match expected value");
        // let b_hat_expected = self.r1cs_prover_input.b_r1cs.proj_lr_challenges(&b_point.0, &b_point.1);
        // assert_eq!(b_hat, b_hat_expected, "b_hat does not match expected value");
        // let c_hat_final_expected = self.r1cs_prover_input.c_r1cs.proj_lr_challenges(&c_point_final.0, &c_point_final.1);
        // assert_eq!(c_hat_final, c_hat_final_expected, "c_hat_final does not match expected value");
        // // End DEBUG

        // Pass the hat values and point values of a, b, c to the batchproj step to yield a single projection of (a || b || c)
        //
        // Explicit type annotation to avoid compiler inferring legacy BatchProj type (we expect DenseMatCM variant)
        let mut batch_proj: BatchProjField<E::ScalarField> = BatchProjField::new(
            vec![a_hat, b_hat, c_hat_final.clone(), c_hat_final],
            vec![a_point, b_point, c_point_final.clone(), c_point_final],
            vec![a_hat_index, b_hat_index, c_hat_index_final.clone(), c_hat_index_final],
            vec![a_point_index, b_point_index, c_point_index_final.clone(), c_point_index_final],
        );
        batch_proj.set_input(vec![
            self.r1cs_prover_input.a_r1cs.clone(), 
            self.r1cs_prover_input.b_r1cs.clone(), 
            self.r1cs_prover_input.c_r1cs.clone(),
            self.r1cs_prover_input.c_r1cs.clone(),
        ]);
        batch_proj.reduce_prover(&mut self.reduce_trans);
        batch_proj.clear();
        let (r1cs_hat, r1cs_point) = (batch_proj.atomic_pop.c_hat, batch_proj.atomic_pop.c_point.clone());
        let (_r1cs_hat_index, _r1cs_point_index) = (batch_proj.atomic_pop.mapping.c_hat_index, batch_proj.atomic_pop.mapping.c_point_index.clone());

        // ===============================================================================
        // ====================  PCS opening in the following ============================
        // ===============================================================================
        // Groth openings now (need r1cs_hat and selection challenges captured earlier)
        let mut rng_g = StdRng::seed_from_u64(8888);
        // r1cs open: reconstruct padded column-major matrix (A||B||C||C)
        if let Some(pk) = &self.groth_r1cs_pk { // commit_to_r1cs_mat must have been called
            let cols = self.r1cs_mat_shape.1; // rows & total_cols not needed explicitly
            let a_mat = &self.r1cs_prover_input.a_r1cs;
            let b_mat = &self.r1cs_prover_input.b_r1cs;
            let c_mat = &self.r1cs_prover_input.c_r1cs;
            
            let mut col_concat = Vec::new();
            for c in 0..cols { col_concat.extend(a_mat.data[c].clone()); }
            for c in 0..cols { col_concat.extend(b_mat.data[c].clone()); }
            for c in 0..cols { col_concat.extend(c_mat.data[c].clone()); }
            for c in 0..cols { col_concat.extend(c_mat.data[c].clone()); }

        
            let r1cs_point_flatten = [r1cs_point.1.as_slice(), r1cs_point.0.as_slice()].concat();
            println!("R1CS point flatten length {:?}", r1cs_point_flatten.len());

            let r1cs_proof = GrothPC::open::<E,_>(pk, &vec![col_concat], &r1cs_point_flatten, &Vec::new(), &mut rng_g)
                .expect("open r1cs groth failed");
            self.groth_r1cs_eval = r1cs_proof.eval; 
            self.groth_r1cs_proof = r1cs_proof; 
            assert_eq!(r1cs_hat, self.groth_r1cs_eval, "r1cs groth eval mismatch");
        }


        // vectors open (treated as column matrices of width 1) xr empty
        let empty: Vec<E::ScalarField> = vec![];
        
        if !self.state_vec.is_empty() { if let Some(pk)= &self.groth_state_pk {
            let state_proof = GrothPC::open::<E,_>(pk, &vec![self.state_vec.clone()], &proof_state_point_raw.0, &empty, &mut rng_g)
                .expect("open state_vec groth failed");
            self.groth_state_eval = state_proof.eval;
            self.groth_state_proof = state_proof;
        }}
        
        if !self.state_vec_shift.is_empty() { if let Some(pk)= &self.groth_state_shift_pk {
            let shift_proof = GrothPC::open::<E,_>(pk, &vec![self.state_vec_shift.clone()], &proof_state_point_raw.0, &empty, &mut rng_g)
                .expect("open state_vec_shift groth failed");
            self.groth_state_shift_eval = shift_proof.eval;
            self.groth_state_shift_proof = shift_proof;
        }}
        
        if !self.proof_vec.is_empty() { if let Some(pk)= &self.groth_proof_pk {
            let proof_proof = GrothPC::open::<E,_>(pk, &vec![self.proof_vec.clone()], &proof_state_point_raw.0, &empty, &mut rng_g)
                .expect("open proof_vec groth failed");
            self.groth_proof_eval = proof_proof.eval;
            self.groth_proof_proof = proof_proof;
        }}
        // relation
        let xr1 = proof_state_point_raw.1[proof_state_point_raw.1.len()-1];
        let xr2 = proof_state_point_raw.1[proof_state_point_raw.1.len()-2];
        assert_eq!(
            proof_state_hat,
            self.groth_state_eval + xr1 * self.groth_state_shift_eval + xr2 * self.groth_proof_eval,
            "Proof_state_hat relation failed"
        );
       
        // witness directly uses column-major data (consistent with commit_to_transcript)
        let witness_formatted: Vec<Vec<E::ScalarField>> = self.r1cs_prover_input.witness_mat.data.clone();

        // use the open algorithm of zkSMART to prove the projection of w and (a || b || c || c)
        // Removed SmartPC r1cs open (GrothPC handles r1cs)
        let hat_w_com = pcsrs.u * hat_w;

        let pcs_trans_witness_result = SmartPC::<E>::open_square (
            pcsrs,
            &witness_formatted,
            &point_w.0,
            &point_w.1,
            hat_w_com,
            self.witness_commit,
            &self.witness_commit_cache,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        );
        // Ensure we don't proceed with an empty transcript; propagate error up
        match pcs_trans_witness_result {
            Ok(trans) => {
                self.pcs_trans_witness = trans;
            }
            Err(e) => {
                eprintln!("SmartPC::open_square failed: {:?}", e);
                return false;
            }
        }

        // Removed SmartPC open for proof_state_mat (replaced by Groth vectors)

        true
    }

    pub fn set_trans(
        &mut self,
        reduce_trans: Transcript<E::ScalarField>,
        trans_witness: PcsTrans<E>,
    ) {
        self.reduce_trans = reduce_trans;
        self.reduce_trans.reset_pointer();
        self.pcs_trans_witness = trans_witness;
    }

    pub fn reset_pointer(&mut self) {
        self.reduce_trans.reset_pointer();
    }

    pub fn verify_r1cs_constraints(&mut self, pcsrs: &PcsPP<E>) -> bool {

        let mut eqzero = EqZero::new((self.r1cs_mat_shape.0, self.proof_len));
        eqzero.verify_as_subprotocol(&mut self.reduce_trans);
        let (c_hat, c_point) = eqzero.atomic_pop.get_a();
        let (c_hat_index, c_point_index) = eqzero.atomic_pop.get_a_index();

      
        // Pass their hat values and point values of (Aw) ∘ (Bw) - Cw and their indices to the sub protocol
        let mut sub = MatSub::new(
            c_hat.clone(),
            c_point.clone(),
            c_hat_index,
            c_point_index.clone(),
            (self.r1cs_mat_shape.0, self.proof_len),
            (self.r1cs_mat_shape.0, self.proof_len),
            (self.r1cs_mat_shape.0, self.proof_len),
        );
        
       
        // sub.set_input(hadamard_result.clone(), cw.clone());
        sub.verify_as_subprotocol(&mut self.reduce_trans);
    
        let (c_hat_hadamard, c_point_hadamard) = sub.atomic_pop.get_a();
        let (c_hat_index_hadamard, c_point_index_hadamard) = sub.atomic_pop.get_a_index();
        let (c_hat_mul_c, c_point_mul_c) = sub.atomic_pop.get_b();
        let (c_hat_index_mul_c, c_point_index_mul_c) = sub.atomic_pop.get_b_index();

        println!("================ hadamard shape: {:?}, {:?}", self.r1cs_mat_shape.0, self.proof_len);

        // Pass the hat values and point values of (Aw) ∘ (Bw) and their indices to the Hadamard protocol
        let mut hadamard = Hadamard::new(
            c_hat_hadamard,
            c_point_hadamard,
            c_hat_index_hadamard,
            c_point_index_hadamard,
            (self.r1cs_mat_shape.0, self.proof_len),
            (self.r1cs_mat_shape.0, self.proof_len),
            (self.r1cs_mat_shape.0, self.proof_len),
        );
        hadamard.verify_as_subprotocol(&mut self.reduce_trans);
        
        let (c_hat_mul_a, c_point_mul_a) = hadamard.atomic_pop.get_a();
        let (c_hat_index_mul_a, c_point_index_mul_a) = hadamard.atomic_pop.get_a_index();
        let (c_hat_mul_b, c_point_mul_b) = hadamard.atomic_pop.get_b();
        let (c_hat_index_mul_b, c_point_index_mul_b) = hadamard.atomic_pop.get_b_index();

        // Pass the hat values and point values of (Aw), (Bw), (Cw) and their indices to the multiplication step
        let mut mul_a = MatMul::new(
            c_hat_mul_a,
            c_point_mul_a,
            c_hat_index_mul_a,
            c_point_index_mul_a,
            (self.r1cs_mat_shape.0, self.proof_len),
            self.r1cs_mat_shape,
            (self.r1cs_mat_shape.1, self.proof_len),
        );
        mul_a.verify_as_subprotocol(&mut self.reduce_trans);
        
        let (a_hat, a_point) = mul_a.atomic_pop.get_a();
        let (a_hat_index, a_point_index) = mul_a.atomic_pop.get_a_index();
        let (w_hat_1, w_point_1) = mul_a.atomic_pop.get_b();
        let (w_hat_index_1, w_point_index_1) = mul_a.atomic_pop.get_b_index();

        let mut mul_b = MatMul::new(
            c_hat_mul_b,
            c_point_mul_b,
            c_hat_index_mul_b,
            c_point_index_mul_b,
            (self.r1cs_mat_shape.0, self.proof_len),
            self.r1cs_mat_shape,
            (self.r1cs_mat_shape.1, self.proof_len),
        );
        mul_b.verify_as_subprotocol(&mut self.reduce_trans);
       
        let (b_hat, b_point) = mul_b.atomic_pop.get_a();
        let (b_hat_index, b_point_index) = mul_b.atomic_pop.get_a_index();
        let (w_hat_2, w_point_2) = mul_b.atomic_pop.get_b();
        let (w_hat_index_2, w_point_index_2) = mul_b.atomic_pop.get_b_index();

        let mut mul_c = MatMul::new(
            c_hat_mul_c,
            c_point_mul_c,
            c_hat_index_mul_c,
            c_point_index_mul_c,
            (self.r1cs_mat_shape.0, self.proof_len),
            self.r1cs_mat_shape,
            (self.r1cs_mat_shape.1, self.proof_len),
        );
        mul_c.verify_as_subprotocol(&mut self.reduce_trans);
        let (c_hat_final, c_point_final) = mul_c.atomic_pop.get_a();
        let (c_hat_index_final, c_point_index_final) = mul_c.atomic_pop.get_a_index();
        let (w_hat_3, w_point_3) = mul_c.atomic_pop.get_b();
        let (w_hat_index_3, w_point_index_3) = mul_c.atomic_pop.get_b_index();

        let mut eqzero_proof_state = EqZero::new((self.proof_len, 4));
        eqzero_proof_state.verify_as_subprotocol(&mut self.reduce_trans);
        let (proof_state_hat_zero, proof_state_point_zero) = eqzero_proof_state.atomic_pop.get_a();
        let (proof_state_hat_index_zero, proof_state_point_index_zero) = eqzero_proof_state.atomic_pop.get_a_index();

        let mut sub_selection = MatSub::new(
            proof_state_hat_zero,
            proof_state_point_zero,
            proof_state_hat_index_zero,
            proof_state_point_index_zero,
            (self.proof_len, 4),
            (self.proof_len, 4),
            (self.proof_len, 4),
        );
        sub_selection.verify_as_subprotocol(&mut self.reduce_trans);
        let (proof_state_hat, proof_state_point_raw) = sub_selection.atomic_pop.get_a();
        let (proof_state_hat_index, proof_state_point_index_raw) = sub_selection.atomic_pop.get_a_index();
        
        let proof_state_point = (
            proof_state_point_raw.1.clone(),
            proof_state_point_raw.0.clone()
        );
        let proof_state_point_index = (
            proof_state_point_index_raw.1,
            proof_state_point_index_raw.0
        );

        let mut mul_selection = MatMul::new(
            proof_state_hat,
            proof_state_point,
            proof_state_hat_index,
            proof_state_point_index,
            (4, self.proof_len),
            (4, self.r1cs_mat_shape.1),
            (self.r1cs_mat_shape.1, self.proof_len),
        );
        mul_selection.verify_as_subprotocol(&mut self.reduce_trans);
        let (selection_hat, selection_point) = mul_selection.atomic_pop.get_a();
        let (w_hat_selection, w_point_selection) = mul_selection.atomic_pop.get_b();
        let (w_hat_selection_index, w_point_selection_index) = mul_selection.atomic_pop.get_b_index();

        let xl1 = selection_point.0[selection_point.0.len()-1];
        let xl2 = selection_point.0[selection_point.0.len()-2];
        let xr1 = selection_point.1[selection_point.1.len()-1];
        let xr2 = selection_point.1[selection_point.1.len()-2];
        let selection_hat_expected = xr1 + xl1 * xr2 + xl2 * xr1 * xr2;
        let flag4 = selection_hat == selection_hat_expected;

        // Pass the hat values of point values of w to the batchpoints step to yield a single projection of w
        let mut batch_points = BatchPoint::new(
            vec![w_hat_1, w_hat_2, w_hat_3, w_hat_selection],
            vec![w_point_1, w_point_2, w_point_3, w_point_selection],
            vec![w_hat_index_1, w_hat_index_2, w_hat_index_3, w_hat_selection_index],
            vec![w_point_index_1, w_point_index_2, w_point_index_3, w_point_selection_index],
        );
        batch_points.verify(&mut self.reduce_trans);
        
        // BatchPoint uses different atomic_pop structure, directly access fields
        let hat_w = batch_points.atomic_pop.c_hat;
        let point_w = batch_points.atomic_pop.c_point.clone();
        let _hat_w_index = batch_points.atomic_pop.mapping.c_hat_index;
        let _point_w_index = batch_points.atomic_pop.mapping.c_point_index.clone();

        // Pass the hat values and point values of a, b, c to the batchproj step to yield a single projection of (a || b || c)
        // 
        let mut batch_proj = BatchProjField::new(
            vec![a_hat, b_hat, c_hat_final.clone(), c_hat_final],
            vec![a_point, b_point, c_point_final.clone(), c_point_final],
            vec![a_hat_index, b_hat_index, c_hat_index_final.clone(), c_hat_index_final],
            vec![a_point_index, b_point_index, c_point_index_final.clone(), c_point_index_final],
        );
        batch_proj.verify(&mut self.reduce_trans);
        // BatchProjField also uses different atomic_pop structure, directly access fields
        let r1cs_hat = batch_proj.atomic_pop.c_hat;
        let r1cs_point = batch_proj.atomic_pop.c_point.clone();
        let _r1cs_hat_index = batch_proj.atomic_pop.mapping.c_hat_index;
        let _r1cs_point_index = batch_proj.atomic_pop.mapping.c_point_index.clone();

        // ===================================================================================
        // =====================  PCS verification in the following ==========================
        // ===================================================================================


        // GrothPC verification only for r1cs now
        let r1cs_point_flatten = [r1cs_point.1.as_slice(), r1cs_point.0.as_slice()].concat();
        let r1cs_xr_empty: Vec<E::ScalarField> = Vec::new();
        let flag1 = GrothPC::verify::<E>(
            &self.groth_r1cs_comm,
            r1cs_hat,
            &r1cs_point_flatten,
            &r1cs_xr_empty,
            &self.groth_r1cs_proof
        ).unwrap_or(false);

        let hat_w_com = pcsrs.u * hat_w;

        let flag2 = SmartPC::<E>::verify_square (
            pcsrs,
            self.witness_commit,
            hat_w_com,
            &point_w.0,
            &point_w.1,
            &self.pcs_trans_witness,
        ).unwrap_or(false);
    
        // removed old SmartPC multi-vector commitment pairing check
        // Groth vector verifications + relation (reuse selection challenges)
        let xl_vec = proof_state_point_raw.0.clone();
        let xr_empty: Vec<E::ScalarField> = vec![];
        let ok_state = GrothPC::verify::<E>(
            &self.groth_state_comm, self.groth_state_eval, &xl_vec, &xr_empty, &self.groth_state_proof
        ).unwrap_or(false);
        let ok_shift = GrothPC::verify::<E>(
            &self.groth_state_shift_comm, self.groth_state_shift_eval, &xl_vec, &xr_empty, &self.groth_state_shift_proof
        ).unwrap_or(false);
        let ok_proof = GrothPC::verify::<E>(
            &self.groth_proof_comm, self.groth_proof_eval, &xl_vec, &xr_empty, &self.groth_proof_proof
        ).unwrap_or(false);
        let xr1 = proof_state_point_raw.1[proof_state_point_raw.1.len()-1];
        let xr2 = proof_state_point_raw.1[proof_state_point_raw.1.len()-2];

        let rel_ok = proof_state_hat == self.groth_state_eval + xr1*self.groth_state_shift_eval + xr2*self.groth_proof_eval;
        println!("  Groth vec evals verify: state={} shift={} proof={} rel_ok={}", ok_state, ok_shift, ok_proof, rel_ok);
        let flag3 = ok_state && ok_shift && ok_proof && rel_ok;

        println!("[FSBatchGroth::verify_r1cs_constraints] flag1(r1cs)={} flag2(witness)={} flag3(vectors+rel)={} flag4(selection)={}", flag1, flag2, flag3, flag4);
        // println!("  r1cs_point sizes: L={}, R={} | witness_point sizes: L={}, R={}", r1cs_point.0.len(), r1cs_point.1.len(), point_w.0.len(), point_w.1.len());
    
        flag1 && flag2 && flag3 && flag4
    }

    pub fn get_trans_size(&self) -> usize {
        use ark_serialize::Compress;
        // 1. reduction transcript (stored as field elements sequence)
        let size_reduce = self.reduce_trans.trans_seq.len() * std::mem::size_of::<E::ScalarField>();
        // 2. SmartPC transcripts we still retain (witness only).
        let size_r1cs_smart = 0; // SmartPC r1cs removed
        let size_witness_smart = self.pcs_trans_witness.serialized_size(Compress::Yes);
        // 3. Groth commitments & proofs (r1cs + 3 vectors)
        let size_field = std::mem::size_of::<E::ScalarField>();
        let size_evals = 4 * size_field; // r1cs + 3 vector eval scalars
        let size_groth_r1cs_comm = self.groth_r1cs_comm.serialized_size(Compress::Yes);
        let size_groth_r1cs_proof = self.groth_r1cs_proof.serialized_size(Compress::Yes);
        let size_groth_state_comm = self.groth_state_comm.serialized_size(Compress::Yes);
        let size_groth_state_proof = self.groth_state_proof.serialized_size(Compress::Yes);
        let size_groth_state_shift_comm = self.groth_state_shift_comm.serialized_size(Compress::Yes);
        let size_groth_state_shift_proof = self.groth_state_shift_proof.serialized_size(Compress::Yes);
        let size_groth_proof_comm = self.groth_proof_comm.serialized_size(Compress::Yes);
        let size_groth_proof_proof = self.groth_proof_proof.serialized_size(Compress::Yes);
        let size_groth_total = size_evals
            + size_groth_r1cs_comm + size_groth_r1cs_proof
            + size_groth_state_comm + size_groth_state_proof
            + size_groth_state_shift_comm + size_groth_state_shift_proof
            + size_groth_proof_comm + size_groth_proof_proof;
        let total = size_reduce + size_r1cs_smart + size_witness_smart + size_groth_total;
        println!("======================");
        println!("[FSBatchGroth::get_trans_size] reduce={} smart_r1cs={} smart_witness={} groth_total={} -> {} bytes", 
            size_reduce, size_r1cs_smart, size_witness_smart, size_groth_total, total);
        println!("  groth detail: r1cs(c={},p={}) state(c={},p={}) shift(c={},p={}) proof(c={},p={}) evals={}B", 
            size_groth_r1cs_comm, size_groth_r1cs_proof,
            size_groth_state_comm, size_groth_state_proof,
            size_groth_state_shift_comm, size_groth_state_shift_proof,
            size_groth_proof_comm, size_groth_proof_proof,
            size_evals);
        total
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr}; 
    use std::fs;
    
    #[test]
    fn test_fsbatchgroth_end_to_end() {
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
        // 1. PCS setup (small qlog)
        let mut rng = ark_std::test_rng();
        // qlog >= padded dim log2
        let qlog = 16usize; 
        let trans_len = 32000usize;
        let pcs_pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).expect("pcs setup failed");

        // 2. Construct transcript
        let mut trans = Transcript::<Fr>::new(Fr::from(0u64));
        for i in 1..=trans_len { trans.push_response(Fr::from(i as u64)); let _ = trans.gen_challenge(); }

        // 3. Create FSBatch
        // Use Groth version batch proof structure
        let mut fsb: FSBatchGroth<Bls12_381> = FSBatchGroth::new(trans.get_fs_proof_vec().len());
        let _ = fsb.commit_to_r1cs_mat();


        let timer = std::time::Instant::now();
        let _ = fsb.commit_to_transcript(&pcs_pp, &trans);
        fsb.prove_r1cs_constraints(&pcs_pp);

        let duration_prover = timer.elapsed().as_secs_f64();

        // Set transcript (third SmartPC transcript no longer used, kept as placeholder)
        fsb.reset_pointer();

        // 7. Verify function current return
        let timer = std::time::Instant::now();
        let flag = fsb.verify_r1cs_constraints(&pcs_pp);
        let duration_verifier = timer.elapsed().as_secs_f64();

        let trans_size = fsb.get_trans_size();
        println!("=============== Experiment Result ==================");
        println!("====Input transcript size before PoP: {} bytes", trans_len);
        println!("====Prove fs time: {}s", duration_prover);
        println!("====Verify fs time: {}s", duration_verifier);
        println!("====Transcript size (with Groth parts): {} bytes", trans_size);
        let peak_kb = *max_memory.lock().unwrap();
        println!("====Peak RSS (approx): {} KB", peak_kb);

        // detach monitor thread (it will exit with process); no join to avoid prolonging test

        assert!(flag);
        assert!(trans_size > 0, "transcript size should be positive");
    }

    fn get_memory_usage() -> Option<u64> {
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