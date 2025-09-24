//! Verify that the committed Transcript is correctly produced from Fiat-Shamir
// 
use ark_ff::{PrimeField, UniformRand, Zero, One};
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::CanonicalSerialize;

use ark_poly_commit::smart_pc::SmartPC;
use ark_poly_commit::smart_pc::data_structures::{
    Trans as PcsTrans,
    UniversalParams as PcsPP,
};



use atomic_proof::protocols::{BatchPoint, BatchProjField, Hadamard, MatMul, MatSub, EqZero};
use atomic_proof::AtomicMatProtocol;

use fsproof::{BatchConstraints, Transcript};

use mat::DenseMatFieldCM;


#[derive(Clone, Debug)]
pub struct FSBatch<E: Pairing>
where
    E: Pairing,
    E::ScalarField: Absorb + UniformRand + PrimeField,
{
    proof_len: usize,
    r1cs_mat_shape: (usize, usize),
    proof_vec_commit: E::G1,
    state_vec_commit: E::G1,
    state_vec_shift_commit: E::G1,
    proof_vec: Vec<E::ScalarField>,
    state_vec: Vec<E::ScalarField>,
    state_vec_shift: Vec<E::ScalarField>,
    r1cs_mats_commit: PairingOutput<E>,
    witness_commit: PairingOutput<E>,
    r1cs_prover_input: BatchConstraints<E::ScalarField>,
    r1cs_mat_commit_cache: Vec<E::G1>,
    witness_commit_cache: Vec<E::G1>,
    // The total transcrip contains three sub-transcripts
    reduce_trans: Transcript<E::ScalarField>,
    pcs_trans_1: PcsTrans<E>,
    pcs_trans_2: PcsTrans<E>,
    pcs_trans_3: PcsTrans<E>,
}


impl<E: Pairing> FSBatch<E> 
where 
    E::ScalarField: Absorb + UniformRand + PrimeField,
{
    fn default_trans() -> PcsTrans<E> {
        PcsTrans {
            vec_l_tilde: Vec::new(),
            vec_r_tilde: Vec::new(),
            com_rhs_tilde: PairingOutput::default(),
            v_g: E::G1::default(),
            v_h: E::G2::default(),
            v_g_prime: E::G1::default(),
            v_h_prime: E::G2::default(),
            w_g: E::G1::default(),
            w_h: E::G1::default(),
            schnorr_1_f: PairingOutput::default(),
            schnorr_1_z: E::ScalarField::zero(),
            schnorr_2_f: PairingOutput::default(),
            schnorr_2_z_1: E::ScalarField::zero(),
            schnorr_2_z_2: E::ScalarField::zero(),
        }
    }

    pub fn new(proof_len: usize) -> Self  {
        FSBatch {
            proof_len,
            r1cs_mat_shape: (0, 0),
            proof_vec_commit: E::G1::default(),
            state_vec_commit: E::G1::default(),
            state_vec_shift_commit: E::G1::default(),
            proof_vec: Vec::new(),
            state_vec: Vec::new(),
            state_vec_shift: Vec::new(),
            r1cs_mats_commit: PairingOutput::default(),
            witness_commit: PairingOutput::default(),
            r1cs_prover_input: BatchConstraints::<E::ScalarField>::new(),
            r1cs_mat_commit_cache: Vec::new(),
            witness_commit_cache: Vec::new(),
            reduce_trans: Transcript::new(E::ScalarField::zero()),
            pcs_trans_1: Self::default_trans(),
            pcs_trans_2: Self::default_trans(),
            pcs_trans_3: Self::default_trans(),
        }
    }

    pub fn default() -> Self {
        Self::new(0)
    }

    pub fn set_commitments(
        &mut self,
        r1cs_mats_commit: PairingOutput<E>,
        proof_vec_commit: E::G1,
        state_vec_commit: E::G1,
        state_vec_shift_commit: E::G1,
        witness_commit: PairingOutput<E>,
    ) {
        self.r1cs_mats_commit = r1cs_mats_commit;
        self.proof_vec_commit = proof_vec_commit;
        self.state_vec_commit = state_vec_commit;
        self.state_vec_shift_commit = state_vec_shift_commit;
        self.witness_commit = witness_commit;
    }

    pub fn commit_to_r1cs_mat(
        &mut self,
        pcsrs: &PcsPP<E>
    ) -> PairingOutput<E> {
        let (a_mat, b_mat, c_mat) = self.r1cs_prover_input.gen_r1cs_constraints();
        // In column-major order (data is already a collection of column vectors),
        // directly concatenate the columns of the three matrices, then pad with the c matrix again (a||b||c||c).
        let mut r1cs_mat_concat = a_mat.to_vec();
        r1cs_mat_concat.extend(b_mat.to_vec());
        r1cs_mat_concat.extend(c_mat.to_vec());
        r1cs_mat_concat.extend(c_mat.to_vec());

        let (r1cs_mats_com, r1cs_mats_cache) = SmartPC::<E>::commit_square_full(
            pcsrs,
            &vec![r1cs_mat_concat],
            E::ScalarField::zero(),
        ).expect("commit_full failed for r1cs mats");

        self.r1cs_mat_shape = a_mat.shape.clone();

        self.r1cs_mats_commit = r1cs_mats_com;
        self.r1cs_mat_commit_cache = r1cs_mats_cache;

        r1cs_mats_com
    }

    pub fn commit_to_transcript(
        &mut self,
        pcsrs: &PcsPP<E>,
        trans: &Transcript<E::ScalarField>
    ) -> (
        E::G1,
        E::G1,
        E::G1,
        PairingOutput<E>
    ) {
        let fs_trans = trans.fs.clone();
        self.r1cs_prover_input.prepare_from_transcript(&fs_trans);

        // Obtain the original witness in column-major order (collection of column vectors)
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
                // Write it back into BatchConstraints so that later multiplications use the padded columns
                self.r1cs_prover_input.witness_mat.data = witness.clone();
                self.r1cs_prover_input.witness_mat.shape = (rows, target);
            }
        }

        // Debug info and dimension check (to avoid panic inside SmartPC::commit_full)
        let w_n = witness.len();
        let w_m = if w_n > 0 { witness[0].len() } else { 0 };
        println!("[FSBatch::commit_to_transcript] witness dims n={}, m={}, q={} (qlog≈{:.0})", w_n, w_m, pcsrs.q, (pcsrs.q as f64).log2());
        if w_n > pcsrs.q || w_m > pcsrs.q {
            panic!(
                "Witness dimensions (n={}, m={}) exceed PCS bound q={} (increase qlog to >= {} / current qlog≈{:.0})",
                w_n, w_m, pcsrs.q, w_n.max(w_m), (pcsrs.q as f64).log2()
            );
        }

        let proof_vec_raw = trans.get_fs_proof_vec();
        let state_vec_raw = trans.get_fs_state_vec();

        // Compute padding length up to next power of two
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

        // Dimension check (after padding)
        println!("[FSBatch::commit_to_transcript] proof_vec len={}, state_vec len={}, q={} ", proof_vec.len(), state_vec.len(), pcsrs.q);
        if proof_vec.len() > pcsrs.q || state_vec.len() > pcsrs.q {
            panic!("Padded proof/state len exceed q (pv={}, sv={}, q={})", proof_vec.len(), state_vec.len(), pcsrs.q);
        }

        // Update proof_len: use the fully padded length of proof_vec
        self.proof_len = proof_vec.len();
        self.proof_vec = proof_vec.clone();
        self.state_vec = state_vec.clone();
        self.state_vec_shift = state_vec_shift.clone();

        assert_eq!(proof_vec.len(), w_n, "Proof Length and witness length does not match");

        let zero_vec = vec![E::ScalarField::zero(); proof_vec.len()];
        let witness_upper_trans = vec![state_vec, state_vec_shift, proof_vec, zero_vec];
        let (com_proof_state, witness_upper_tier_1) = SmartPC::<E>::commit_full(
            pcsrs, &witness_upper_trans, E::ScalarField::zero()
        ).expect("commit_full failed for witness upper tier 1");

        let state_vec_com = witness_upper_tier_1[0].clone();
        let state_vec_shift_com = witness_upper_tier_1[1].clone();
        let proof_vec_com = witness_upper_tier_1[2].clone();
 

        self.proof_vec_commit = proof_vec_com;
        self.state_vec_commit = state_vec_com;
        self.state_vec_shift_commit = state_vec_shift_com;

        let com_expected = E::pairing(state_vec_com, pcsrs.vec_h[0])
            + E::pairing(state_vec_shift_com, pcsrs.vec_h[1])
            + E::pairing(proof_vec_com, pcsrs.vec_h[2]);

        assert_eq!(com_proof_state, com_expected);

        let (witness_com, witness_com_cache) = SmartPC::<E>::commit_square_full(
            pcsrs, &witness, E::ScalarField::zero()
        ).expect("commit_full failed for witness");

        self.witness_commit = witness_com;
        self.witness_commit_cache = witness_com_cache;


        (proof_vec_com, state_vec_com, state_vec_shift_com, witness_com)
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
        sub_selection.set_input(proof_state_mat.clone(), proof_state_mat.clone());
        sub_selection.reduce_prover(&mut self.reduce_trans);
        let (proof_state_hat, proof_state_point_raw) = sub_selection.atomic_pop.get_a();
        let (proof_state_hat_index, proof_state_point_index_raw) = sub_selection.atomic_pop.get_a_index();
        
        let proof_state_point = (proof_state_point_raw.1.clone(), proof_state_point_raw.0.clone());
        let proof_state_point_index = (proof_state_point_index_raw.1.clone(), proof_state_point_index_raw.0.clone());

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
        let (_hat_w_index, _point_w_index) = (batch_points.atomic_pop.mapping.c_hat_index, batch_points.atomic_pop.mapping.c_point_index.clone());


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
        let mut batch_proj = BatchProjField::new(
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


       
        let mut r1cs_mat_concat = self.r1cs_prover_input.a_r1cs.to_vec();
        r1cs_mat_concat.extend(self.r1cs_prover_input.b_r1cs.to_vec());
        r1cs_mat_concat.extend(self.r1cs_prover_input.c_r1cs.to_vec());
        r1cs_mat_concat.extend(self.r1cs_prover_input.c_r1cs.to_vec());
  
        // // START DEBUG
        // let mut r1cs_mat_formatted = DenseMatFieldCM::<E::ScalarField>::new(self.r1cs_mat_shape.0, self.r1cs_mat_shape.1);
        // r1cs_mat_formatted.data = vec![r1cs_mat_concat.clone()];
        // let expected_r1cs_hat = r1cs_mat_formatted.proj_lr_challenges(&r1cs_point.0, &r1cs_point.1);
        // assert_eq!(r1cs_hat, expected_r1cs_hat, "r1cs_hat does not match expected value");
        // // END DEBUG


        // Witness uses column-major data directly (consistent with commit_to_transcript)
        let witness_formatted: Vec<Vec<E::ScalarField>> = self.r1cs_prover_input.witness_mat.data.clone();


        // Use the zkSMART open algorithm to prove the projection of w and (a || b || c || c)
        let hat_r1cs_com = pcsrs.u * r1cs_hat;
        let pcs_trans_1_result = SmartPC::<E>::open_square (
            pcsrs,
            &vec![r1cs_mat_concat],
            &r1cs_point.0,
            &r1cs_point.1,
            hat_r1cs_com,
            self.r1cs_mats_commit,
            &self.r1cs_mat_commit_cache,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        );
        
        if let Ok(trans) = pcs_trans_1_result {
            self.pcs_trans_1 = trans;
        }

        let hat_w_com = pcsrs.u * hat_w;

        let pcs_trans_2_result = SmartPC::<E>::open_square (
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
        
        if let Ok(trans) = pcs_trans_2_result {
            self.pcs_trans_2 = trans;
        }

        let hat_proof_state_commitment = pcsrs.u * proof_state_hat;
        let proof_state_mat_com = E::pairing(self.state_vec_commit, pcsrs.vec_h[0])
            + E::pairing(self.state_vec_shift_commit, pcsrs.vec_h[1])
            + E::pairing(self.proof_vec_commit, pcsrs.vec_h[2]);

    
        let pcs_trans_3_result = SmartPC::<E>::open (
            pcsrs,
            &proof_state_mat.data,
            &proof_state_point_raw.0,
            &proof_state_point_raw.1,
            hat_proof_state_commitment,
            proof_state_mat_com,
            &vec![self.state_vec_commit.clone(), self.state_vec_shift_commit.clone(), self.proof_vec_commit.clone(), E::G1::zero()],
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        );

        if let Ok(trans) = pcs_trans_3_result {
            self.pcs_trans_3 = trans;
        }

        true
    }

    pub fn set_trans(&mut self, reduce_trans: Transcript<E::ScalarField>, trans_r1cs_mats: PcsTrans<E>, trans_witness: PcsTrans<E>, trans_proof_state: PcsTrans<E>) {
        self.reduce_trans = reduce_trans;
        self.reduce_trans.reset_pointer();
        self.pcs_trans_1 = trans_r1cs_mats;
        self.pcs_trans_2 = trans_witness;
        self.pcs_trans_3 = trans_proof_state;
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
        
        // Note: In the verifier we cannot reuse hadamard_result and cw directly since they are not available.
        // They should be reconstructed from the transcript or recomputed if needed.
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
        
        let proof_state_point = (proof_state_point_raw.1.clone(), proof_state_point_raw.0.clone());
        let proof_state_point_index = (proof_state_point_index_raw.1, proof_state_point_index_raw.0);

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
        
        // BatchPoint uses a different atomic_pop structure; access fields directly
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
        // BatchProjField also uses a different atomic_pop structure; access fields directly
        let r1cs_hat = batch_proj.atomic_pop.c_hat;
        let r1cs_point = batch_proj.atomic_pop.c_point.clone();
        let _r1cs_hat_index = batch_proj.atomic_pop.mapping.c_hat_index;
        let _r1cs_point_index = batch_proj.atomic_pop.mapping.c_point_index.clone();



        let hat_r1cs_com = pcsrs.u * r1cs_hat;


        // Use the zkSMART open algorithm to prove the projection of w and (a || b || c)
        let flag1 = SmartPC::<E>::verify_square (
            pcsrs,
            self.r1cs_mats_commit,
            hat_r1cs_com,
            &r1cs_point.0,
            &r1cs_point.1,
            &self.pcs_trans_1,
        ).unwrap_or(false);
        

        let hat_w_com = pcsrs.u * hat_w;

        let flag2 = SmartPC::<E>::verify_square (
            pcsrs,
            self.witness_commit,
            hat_w_com,
            &point_w.0,
            &point_w.1,
            &self.pcs_trans_2,
        ).unwrap_or(false);
    
        let hat_proof_state_commitment = pcsrs.u * proof_state_hat;
        let proof_state_mat_com = E::pairing(self.state_vec_commit, pcsrs.vec_h[0])
            + E::pairing(self.state_vec_shift_commit, pcsrs.vec_h[1])
            + E::pairing(self.proof_vec_commit, pcsrs.vec_h[2]);

    
        let flag3 = SmartPC::<E>::verify (
            pcsrs,
            proof_state_mat_com,
            hat_proof_state_commitment,
            &proof_state_point_raw.0,
            &proof_state_point_raw.1,
            &self.pcs_trans_3,
        ).unwrap_or(false);
      
    
        println!("[FSBatch::verify_r1cs_constraints] flag1={} flag2={} flag3={} flag4={}", flag1, flag2, flag3, flag4);
        // println!("  r1cs_point sizes: L={}, R={} | witness_point sizes: L={}, R={}", r1cs_point.0.len(), r1cs_point.1.len(), point_w.0.len(), point_w.1.len());
    
        flag1 && flag2 && flag3 && flag4
    }

    pub fn get_trans_size(&self) -> usize {
        let size1 = self.reduce_trans.trans_seq.len() * std::mem::size_of::<E::ScalarField>();
        let size2 = self.pcs_trans_1.serialized_size(ark_serialize::Compress::Yes);
        let size3 = self.pcs_trans_2.serialized_size(ark_serialize::Compress::Yes);
        let size4 = self.pcs_trans_3.serialized_size(ark_serialize::Compress::Yes);
        println!("======================");
        println!("[FSBatchL::fs proof transcript size: {} bytes", size1 + size2 + size3 + size4);
        println!("[FSBatch::get_trans_size] size1={} bytes size2={} bytes size3={} bytes size4={} bytes", size1, size2, size3, size4);
        size1 + size2 + size3 + size4
    }


}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr}; 
    

    #[test]
    fn test_fsbatch_end_to_end() {
        // 1. PCS setup (choose qlog large enough for padded dimensions)
        let mut rng = ark_std::test_rng();
        // qlog must be >= log2 of padded dimension; observed around 256 -> 2^8
        let qlog = 12usize; 
        let pcs_pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).expect("pcs setup failed");

        // 2. Build transcript
        let mut trans = Transcript::<Fr>::new(Fr::from(0u64));
        for i in 1..128u64 { trans.push_response(Fr::from(i)); let _ = trans.gen_challenge(); }

        // 3. Create FSBatch
        let mut fsb: FSBatch<Bls12_381> = FSBatch::new(trans.get_fs_proof_vec().len());
        let _ = fsb.commit_to_r1cs_mat(&pcs_pp);


        let timer = std::time::Instant::now();
        let _ = fsb.commit_to_transcript(&pcs_pp, &trans);
        fsb.prove_r1cs_constraints(&pcs_pp);

        let duration_prover = timer.elapsed().as_secs_f64();

        fsb.set_trans(fsb.reduce_trans.clone(), fsb.pcs_trans_1.clone(), fsb.pcs_trans_2.clone(), fsb.pcs_trans_3.clone());

        // 7. Run verifier
        let timer = std::time::Instant::now();
        let flag = fsb.verify_r1cs_constraints(&pcs_pp);
        let duration_verifier = timer.elapsed().as_secs_f64();

        let trans_size = fsb.get_trans_size();
        println!("====Prove fs time: {}s", duration_prover);
        println!("====Verify fs time: {}s", duration_verifier);
        println!("====Transcript size: {}bytes", trans_size);

        assert!(flag);
    }
}       