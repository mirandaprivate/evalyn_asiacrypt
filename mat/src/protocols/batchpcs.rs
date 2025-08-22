//! Implement the Batch PCS based on SMART-PC
//!
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use ark_ec::PrimeGroup;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::{
    One,
    Zero,
    UniformRand,
};

use ark_poly_commit::smart_pc::SmartPC;
use ark_poly_commit::smart_pc::utils as pcutils;
use ark_poly_commit::smart_pc::data_structures::{
    Trans as PcsTrans,
    UniversalParams as PcsPP,
};


use crate::data_structures::ZkSRS;

use crate::utils::zktr::{ZkTranSeq, TranSeq, TranElem};
use crate::utils::linear;
use crate::utils::xi;

use crate::utils::matdef::{
    ShortInt,
    DenseMatCM,
    DenseBlockMat,
    MatOps,
    RotationMatIndexFormat,
};

use super::litebullet::ZkLiteBullet;
use super::proj::ZkProj;
use super::scalars::ZkSchnorr;

use ark_std::marker::PhantomData;
use derivative::Derivative;

use crate::MyInt;

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct MatContainer<F>
where
    F: PrimeField + From<MyInt>,
{
    pub dense_myint: Vec<DenseMatCM<MyInt, F>>,
    pub dense_bool: Vec<DenseMatCM<bool, F>>,
    pub dense_block_myint: Vec<DenseBlockMat<MyInt, F>>,
    pub dense_block_bool: Vec<DenseBlockMat<bool, F>>,
    pub square_myint: Vec<DenseMatCM<MyInt, F>>,
    pub square_bool: Vec<DenseMatCM<bool, F>>,
    pub k: usize,
}

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct ComContainer<E: Pairing>
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{
    pub dense_myint: Vec<(PairingOutput<E>, Vec<E::G1>, E::ScalarField)>,
    pub dense_bool: Vec<(PairingOutput<E>, Vec<E::G1>, E::ScalarField)>,
    pub dense_block_myint: Vec<(PairingOutput<E>, Vec<E::G1>, E::ScalarField)>,
    pub dense_block_bool: Vec<(PairingOutput<E>, Vec<E::G1>, E::ScalarField)>,
    pub flattened: Vec<(PairingOutput<E>, Vec<E::G1>, E::ScalarField)>,
}

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct EvalContainer<F, G>
where
    G: PrimeGroup<ScalarField = F>,
    F: PrimeField,
{
    pub dense_myint: Vec<(G, F)>,
    pub dense_bool: Vec<(G, F)>,
    pub dense_block_myint: Vec<(G, F)>,
    pub dense_block_bool: Vec<(G, F)>,
    pub v_coms: Vec<G>,
    pub v_tildes: Vec<F>,
}

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct PointsContainer<F:PrimeField>
{
    pub dense_myint: Vec<(Vec<F>, Vec<F>)>,
    pub dense_bool: Vec<(Vec<F>, Vec<F>)>,
    pub dense_block_myint: Vec<(Vec<F>, Vec<F>)>,
    pub dense_block_bool: Vec<(Vec<F>, Vec<F>)>,
    pub flattened: Vec<(Vec<F>, Vec<F>)>,
}


pub fn commit_mats<E: Pairing> (
    pp: &PcsPP<E>,
    mat_container: &MatContainer<E::ScalarField>,
    com_container: &mut ComContainer<E>,
)
where
    E: Pairing + ark_ec::pairing::Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{

    let rng = &mut ark_std::rand::thread_rng();

    for idx in 0..mat_container.dense_myint.len() {
        let mat =
        &mat_container.dense_myint[idx].to_square_mat().data;


        let com_tilde =
        E::ScalarField::rand(rng);   

        let (com, cache) =
        SmartPC::commit_short(
            &pp,
            &mat,
            com_tilde,
            mat_container.k,
        ).unwrap();

        com_container.dense_myint.push((com, cache, com_tilde));
    }

    for idx in 0..mat_container.dense_block_myint.len() {
        let mat =
        &mat_container.dense_block_myint[idx].to_square_mat().data;

        let com_tilde =
        E::ScalarField::rand(rng);    

      
        let (com, cache) =
        SmartPC::commit_short(
            &pp,
            &mat,
            com_tilde,
            mat_container.k,
        ).unwrap();

        com_container.dense_block_myint.push((com, cache, com_tilde));
    }

    for idx in 0..mat_container.dense_bool.len() {
        let mat =
        &mat_container.dense_bool[idx].to_square_mat().data;


        let com_tilde =
        E::ScalarField::rand(rng);    


        let (com, cache) =
        SmartPC::commit_boolean(
            &pp,
            &mat,
            com_tilde,
        ).unwrap();

        com_container.dense_bool.push((com, cache.clone(), com_tilde));
    }


    for idx in 0..mat_container.dense_block_bool.len() {
        let mat =
        &mat_container.dense_block_bool[idx].to_square_mat().data;

        let com_tilde =
        E::ScalarField::rand(rng);

        let (com, cache) =
        SmartPC::commit_boolean(
            &pp,
            &mat,
            com_tilde,
        ).unwrap();

        com_container.dense_block_bool.push((com, cache, com_tilde));
    }

    com_container.flatten();
}


pub fn open_pc_mats<E: Pairing> (
    zksrs: &ZkSRS<E::ScalarField, E::G1>,
    pp: &PcsPP<E>,
    zk_trans_seq: &mut ZkTranSeq<E::ScalarField, E::G1>,
    mat_container: &mut MatContainer<E::ScalarField>,
    com_container: &mut ComContainer<E>,
    eval_container: &mut EvalContainer<E::ScalarField, E::G1>,
    points_container: &mut PointsContainer<E::ScalarField>,
) -> PcsTrans<E> {

    com_container.flatten();
    mat_container.flatten();

    mat_container.dense_myint = Vec::new();
    mat_container.dense_block_myint = Vec::new();
    mat_container.dense_bool = Vec::new();
    mat_container.dense_block_bool = Vec::new();
    

    eval_container.flatten();
    points_container.flatten();

    let mut protocol =
    BatchMulMats::<MyInt, bool, E::ScalarField, E::G1>::new();

    let num1 = mat_container.square_myint.len();
    let num2 = mat_container.square_bool.len();

    for idx in 0..(num1 + num2) {

        protocol.push_point(
            eval_container.v_coms[idx].clone(),
            points_container.flattened[idx].clone(),
        );
    }

    let (
        z,
        mat_sum,
        hat_a_com,
        hat_a_tilde,
        (xl_prime, xr_prime),
    ) = protocol.reduce_prover(
        zksrs,
        zk_trans_seq,
        &mat_container.square_myint,
        &mat_container.square_bool,
        &eval_container.v_tildes,
    );

    let hat_a_com_gt = 
    E::pairing(hat_a_com, pp.h_hat);

    let n = mat_sum.len();

    let mut factor = E::ScalarField::one();

    let mut cache_sum = vec![E::G1::zero(); n];

    let mut mat_com_sum = PairingOutput::<E>::zero();
    let mut mat_tilde_sum = E::ScalarField::zero();
    
    for idx in 0..com_container.flattened.len() {
        let (com, cache, com_tilde) =
        &com_container.flattened[idx];

        cache_sum = pcutils::add_vec_g1::<E>(
            &cache_sum,
            &pcutils::scalar_mul_vec_g1::<E>(
                &cache,
                &factor,
            ),
        );

        mat_com_sum = mat_com_sum + (* com) * factor;
        mat_tilde_sum = mat_tilde_sum + (* com_tilde) * factor;

        factor = factor * z;
    }

    
    let pc_proof = SmartPC::open(
        &pp,
        &mat_sum,
        &xl_prime,
        &xr_prime,
        hat_a_com_gt,
        mat_com_sum,
        &cache_sum,
        hat_a_tilde,
        mat_tilde_sum,
    ).unwrap();

  
    pc_proof

}



pub fn verify_pc_mats<E: Pairing> (
    zksrs: &ZkSRS<E::ScalarField, E::G1>,
    pp: &PcsPP<E>,
    trans_seq: &mut TranSeq<E::ScalarField, E::G1>,
    pc_proof: &PcsTrans<E>,
    com_container: &mut ComContainer<E>,
    eval_container: &mut EvalContainer<E::ScalarField, E::G1>,
    points_container: &mut PointsContainer<E::ScalarField>,
    num_myint: usize,
) -> bool {
    com_container.flatten();
    eval_container.flatten();
    points_container.flatten();

    let mut protocol = BatchMulMats::<MyInt, bool, E::ScalarField, E::G1>::new();

    for idx in 0..points_container.flattened.len() {
        protocol.push_point(
            eval_container.v_coms[idx].clone(),
            points_container.flattened[idx].clone(),
        );
    }

    let (
        check1,
        z,
        hat_a_com_sum,
        (xl_prime, xr_prime),
    ) = protocol.verify_as_subprotocol(
        zksrs,
        trans_seq,
        num_myint,
    );

    let hat_a_com_sum_gt =
    E::pairing(hat_a_com_sum, pp.h_hat);

    let mut factor = E::ScalarField::one();

    let mut mat_com_sum = PairingOutput::<E>::zero();
    
    for idx in 0..com_container.flattened.len() {
        let (com, _, _) =
        &com_container.flattened[idx];

        mat_com_sum = mat_com_sum + (*com) * factor;

        factor = factor * z;
    }

    let check2 = SmartPC::verify(
        pp,
        mat_com_sum,
        hat_a_com_sum_gt,
        &xl_prime,
        &xr_prime,
        &pc_proof,
    ).unwrap();


    // let mut proof_writer = Vec::new();
    // pc_proof.serialize_compressed(&mut proof_writer).unwrap();
    // let proof_size = proof_writer.len();
    // println!("PC proof size: {}B", proof_size);

    // let mut proof_writer2 = Vec::new();
    // trans_seq.serialize_compressed(&mut proof_writer2).unwrap();
    // let proof_size2 = proof_writer2.len();
    // println!("Reduce proof size: {}B", proof_size2);
    // println!("Total proof size: {}B", proof_size + proof_size2);

    println!("Batch opening PCS: check1: {}, check2: {}", check1, check2);
    check1 && check2

}

impl<F> MatContainer<F> 
where
    F: PrimeField + From<MyInt>,
{
    pub fn new(k: usize) -> Self {
        Self {
            dense_myint: Vec::new(),
            dense_bool: Vec::new(),
            dense_block_myint: Vec::new(),
            dense_block_bool: Vec::new(),
            square_myint: Vec::new(),
            square_bool: Vec::new(),
            k: k,
        }
    }

    pub fn flatten(&mut self)
    {
        
        let mut result_myint = Vec::new();

        for idx in 0..self.dense_myint.len() {
            result_myint.push(
                self.dense_myint[idx].to_square_mat()
            );
        }

        self.dense_myint = Vec::new();


        for idx in 0..self.dense_block_myint.len() {
            result_myint.push(
                self.dense_block_myint[idx].to_square_mat()
            );
        }

        self.dense_block_myint = Vec::new();

        let mut result_bool = Vec::new();

        for idx in 0..self.dense_bool.len() {
            result_bool.push(
                self.dense_bool[idx].to_square_mat()
            );
        }

        self.dense_bool = Vec::new();

        for idx in 0..self.dense_block_bool.len() {
            result_bool.push(
                self.dense_block_bool[idx].to_square_mat()
            );
        }

        self.dense_block_bool = Vec::new();

        self.square_myint = result_myint;
        self.square_bool = result_bool;

    }

}

impl<F> PointsContainer<F> 
where
    F: PrimeField + From<MyInt>,
{
    pub fn new() -> Self {
        Self {
            dense_myint: Vec::new(),
            dense_bool: Vec::new(),
            dense_block_myint: Vec::new(),
            dense_block_bool: Vec::new(),
            flattened: Vec::new(),
        }
    }

    pub fn flatten(&mut self) {
        
        let mut result = Vec::new();

        for idx in 0..self.dense_myint.len() {
            result.push(linear::reshape_points_keep_projection(
                &self.dense_myint[idx]))
        }



        for idx in 0..self.dense_block_myint.len() {
            result.push(linear::reshape_points_keep_projection(
                &self.dense_block_myint[idx]));
        }

        for idx in 0..self.dense_bool.len() {
            result.push(linear::reshape_points_keep_projection(
                &self.dense_bool[idx]))
        }

        for idx in 0..self.dense_block_bool.len() {
            result.push(linear::reshape_points_keep_projection(
                &self.dense_block_bool[idx]));
        }

        self.flattened = result;
    }

}

impl<E> ComContainer<E> 
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{
    pub fn new() -> Self {
        Self {
            dense_myint: Vec::new(),
            dense_bool: Vec::new(),
            dense_block_myint: Vec::new(),
            dense_block_bool: Vec::new(),
            flattened: Vec::new(),
        }
    }

    pub fn flatten(&mut self) {
        
        let mut result = Vec::new();

        for idx in 0..self.dense_myint.len() {
            result.push(
                self.dense_myint[idx].clone()
            );
        }


        for idx in 0..self.dense_block_myint.len() {
            result.push(
                self.dense_block_myint[idx].clone()
            );
        }

        for idx in 0..self.dense_bool.len() {
            result.push(
                self.dense_bool[idx].clone()
            );
        }

        for idx in 0..self.dense_block_bool.len() {
            result.push(
                self.dense_block_bool[idx].clone()
            );
        }

        self.flattened = result;
    }

}

impl<F, G> EvalContainer<F, G>
where
    G: PrimeGroup<ScalarField = F>,
    F: PrimeField,
{
    pub fn new() -> Self {
        Self {
            dense_myint: Vec::new(),
            dense_bool: Vec::new(),
            dense_block_myint: Vec::new(),
            dense_block_bool: Vec::new(),
            v_coms: Vec::new(),
            v_tildes: Vec::new(),
        }
    }

    pub fn flatten(&mut self) {
        
        let mut v_com = Vec::new();
        let mut v_tilde = Vec::new();

        for idx in 0..self.dense_myint.len() {
            v_com.push(self.dense_myint[idx].0);
            v_tilde.push(self.dense_myint[idx].1);
        }


        for idx in 0..self.dense_block_myint.len() {
            v_com.push(self.dense_block_myint[idx].0);
            v_tilde.push(self.dense_block_myint[idx].1);
        }


        for idx in 0..self.dense_bool.len() {
            v_com.push(self.dense_bool[idx].0);
            v_tilde.push(self.dense_bool[idx].1);
        }

        for idx in 0..self.dense_block_bool.len() {
            v_com.push(self.dense_block_bool[idx].0);
            v_tilde.push(self.dense_block_bool[idx].1);
        }

        self.v_coms = v_com;
        self.v_tildes = v_tilde;

    }

}



pub struct BatchPoints<I, F, G>
where 
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_coms: Vec<G>,
    pub points: Vec<(Vec<F>,Vec<F>)>,
    _marker: PhantomData<I>,
}


pub struct BatchPointsRotation<I, F, G>
where 
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_coms: Vec<G>,
    pub points: Vec<(Vec<F>,Vec<F>)>,
    _marker: PhantomData<I>,
}

pub struct BatchMulMats<I1, I2, F, G>
where 
    I1: ShortInt,
    I2: ShortInt,
    F: PrimeField + From<I1> + From<I2>,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_coms: Vec<G>,
    pub points: Vec<(Vec<F>, Vec<F>)>,
    _marker: PhantomData<(I1, I2)>,
}

impl<I, F, G> BatchPointsRotation<I, F, G>
where 
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new() -> Self {
        Self {
            v_coms: Vec::new(),
            points: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push_point(
        &mut self,
        v_com: G,
        point: (Vec<F>, Vec<F>)
    ) {
        self.v_coms.push(v_com);
        self.points.push(point);
    }

    pub fn reduce_prover(
        &self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &RotationMatIndexFormat<I, F>,
        v_tildes: Vec<F>,
    ) -> (G, F, (Vec<F>, Vec<F>))
    {
        // println!("\n BatchPointsRotation: reduce_prover");

        let num = self.points.len();
        if self.v_coms.len() != num
            || num == 0
        {
            panic!("BatchPointsRotation: invalid input");
        }

        let log_m = self.points[0].0.len();
        let log_n = self.points[0].1.len();
        // let m = 1 << log_m;
        // let n = 1 << log_n;

        let mut subprotocols = Vec::new();

        let mut challenges_n = Vec::new();
        let mut challenges_m = Vec::new();
        let mut challenges_inv_n = Vec::new();
        let mut challenges_inv_m = Vec::new();
        

        for idx in 0..num {
            let (xl, xr) = &self.points[idx];

            let v_com_cur = self.v_coms[idx];

            // println!("v_com_cur: {:?}", v_com_cur);

            let v_tilde_cur = v_tildes[idx];

            let mut curprotocol =
            ZkProj::<I, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            curprotocol.prover_prepare_n(
                zk_trans_seq,
                mat,
                v_tilde_cur,
            );

            subprotocols.push(curprotocol);
        }

        for j in 0..log_n {
            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();
            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);
            
            for idx in 0..num {
                    subprotocols[idx].prover_j_in_n_iteration(
                        zk_trans_seq,
                        j,
                        x_j,
                    );                
                }
        }

        for idx in 0..num {
            subprotocols[idx].prover_prepare_m(
            zk_trans_seq,
            mat,
            );
        }


        for j in 0..log_m {
            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();
            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);

            for idx in 0..num {

                subprotocols[idx].prover_j_in_m_iteration(
                    zk_trans_seq,
                    j,
                    x_j,
                );
            }
        }


        for idx in 0..num {
            subprotocols[idx].prover_conclude(
                srs,
                zk_trans_seq,
            );
        }

        let a_hat_com
        = subprotocols[0].prover_intermediate.a_reduce_blind;

        let a_hat_tilde
        = subprotocols[0].prover_intermediate.a_reduce_tilde;


        (
            a_hat_com,
            a_hat_tilde,
            (
                challenges_inv_m.clone(),
                challenges_inv_n.clone(),
            )
        )

    }

    pub fn verify_as_subprotocol(
        &mut self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {

        let num = self.points.len();
        if self.v_coms.len() != num
            || num == 0
        {
            panic!("BatchPoints Prover: invalid num");
        }

        let log_m = self.points[0].0.len();
        let log_n = self.points[0].1.len();

        let mut subprotocols = Vec::new();
        
        let mut challenges_n = Vec::new();
        let mut challenges_m = Vec::new();
        let mut challenges_inv_n = Vec::new();
        let mut challenges_inv_m = Vec::new();
        

        for idx in 0..num {
            let (xl, xr) = &self.points[idx];
            
            let v_com_cur = self.v_coms[idx];


            let mut curprotocol =
            ZkProj::<I, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            assert_eq!(TranElem::Group(v_com_cur), trans_seq.data[trans_seq.pointer].clone());

            curprotocol.verifer_prepare(
                trans_seq,
            );

            subprotocols.push(curprotocol);
        }



        for _ in 0..log_n {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                let x_j_inv = x_j.inverse().unwrap();
                challenges_n.push(x_j);
                challenges_inv_n.push(x_j_inv);

                for idx in 0..num {
                    // println!("idx: {}, j: {}", idx, j);

                    subprotocols[idx].verifier_j_in_n_iteration(
                        trans_seq,
                        x_j,
                        false,
                    );
                }

            } else {
                println!("!! Invalid transcript when verifying BatchMulMats");
            }
        }

        // println!("log_m_max: {}", log_m_max);

        for j in 0..log_m {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                let last_j =
                j == log_m - 1;

                let x_j_inv = x_j.inverse().unwrap();
                challenges_m.push(x_j);
                challenges_inv_m.push(x_j_inv);

                for idx in 0..num {
                    // println!("idx: {}, jm: {}", idx, j);

                    subprotocols[idx].verifier_j_in_m_iteration(
                        trans_seq,
                        x_j,
                        last_j,
                    );
                }
                
            } else {
                println!("!! Invalid transcript when verifying BatchMulMats");
            }

            
        }

        for idx in 0..num {
            subprotocols[idx].verifier_conclude(
                srs,
                trans_seq,
            );
        }


        let mut flag = true;
        let a_reduce_com =
        subprotocols[0].verifier_intermediate.a_reduce_blind;

        for idx in 0..num {
            let cur_flag = subprotocols[idx].verifier_intermediate.flag;
            // println!("idx: {}, cur_flag: {}", idx, cur_flag);

            flag = flag && cur_flag;            
        }


        (
            flag,
            a_reduce_com,
            (
                challenges_inv_m.clone(),
                challenges_inv_n.clone()
            )
        )

    }

}


impl<I, F, G> BatchPoints<I, F, G>
where 
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new() -> Self {
        Self {
            v_coms: Vec::new(),
            points: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push_point(
        &mut self,
        v_com: G,
        point: (Vec<F>, Vec<F>)
    ) {
        self.v_coms.push(v_com);
        self.points.push(point);
        
    }

    pub fn reduce_prover<M>(
        &self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &M,
        v_tildes: Vec<F>,
    ) -> (G, F, (Vec<F>, Vec<F>))
    where
        M: MatOps<I,F>,
    {
        let num = self.points.len();
        if self.v_coms.len() != num
            || num == 0
        {
            panic!("BatchPoints: invalid input");
        }

        let log_m = self.points[0].0.len();
        let log_n = self.points[0].1.len();
        let m = 1 << log_m;
        let n = 1 << log_n;
        let len = m * n;

        let mut b_vec = vec![F::zero(); len];
        let mut v_com = G::zero();
        let mut v_tilde = F::zero();

        let z = zk_trans_seq.gen_challenge();
        let mut factor = F::one();

        for i in 0..num {
            let (xl, xr) = &self.points[i];
            let v_com_cur = &self.v_coms[i];
            let v_tilde_cur = &v_tildes[i];

            if (xl.len() != log_m) || (xr.len() != log_n) {
                panic!("BatchPoints: point size inconsistent");
            }

            let mut xx = xl.clone();
            xx.extend(xr.iter().cloned());

            let xi = xi::xi_from_challenges(&xx);

            b_vec = (0..xi.len()).into_par_iter()
            .map(|i| {
                b_vec[i] + xi[i] * factor
            }).collect();
 
            v_tilde = v_tilde + v_tilde_cur.mul(factor);
            v_com = v_com + v_com_cur.mul(factor);

            factor = factor * z;
        }

        let a_vec =
        linear::reshape_mat_cm_to_field_vec_keep_projection_short::<I, F>(
                    &(mat.to_dense().data));

        let litebullet =
        ZkLiteBullet::new(v_com, len);

        let (
            a_reduce_blind,
            _,
            a_reduce_tilde,
            b_reduce_tilde,
            challenges_inv,
            _,
        ) = litebullet.reduce_prover(
            srs,
            zk_trans_seq,
            &a_vec,
            &b_vec,
            v_tilde,
        );


        let b_tilde_com = srs.blind_base.mul(b_reduce_tilde);

        let schnorr =
        ZkSchnorr::new(
            srs,
            b_tilde_com,
        );

        schnorr.prove(zk_trans_seq, b_reduce_tilde);

        let xl_prime = challenges_inv[0..log_m].to_vec();
        let xr_prime = challenges_inv[log_m..(log_m + log_n)].to_vec();

        // let a_reduce = mat.proj_lr(&xl_prime, &xr_prime);

        // let xi = xi::xi_from_challenges(&challenges_inv);
        // let a_reduce_vec =
        // linear::inner_product(&a_vec, &xi);

        // assert_eq!(a_reduce_vec, a_reduce);

        // let a_reduce_com_check =
        // srs.com_base.mul(a_reduce) + srs.blind_base.mul(a_reduce_tilde);
        // assert_eq!(a_reduce_com_check, a_reduce_blind);


        (a_reduce_blind, a_reduce_tilde, (xl_prime, xr_prime))

    }

    pub fn verify_as_subprotocol(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {
        let num = self.points.len();
        if self.v_coms.len() != num
            || num == 0
        {
            panic!("BatchPoints: invalid input");
        }

        let log_m = self.points[0].0.len();
        let log_n = self.points[0].1.len();
        let m = 1 << log_m;
        let n = 1 << log_n;
        let len = m * n;


        let z: F;
        
        if let (
            TranElem::Coin(z_value),
        ) = (
            trans_seq.data[trans_seq.pointer].clone(),
        ) {
            trans_seq.pointer += 1;
            z = z_value;
        } else {
            panic!("BatchPoints: invalid input");
        }


        let mut factor = F::one();

        let mut v_com = G::zero();

        for i in 0..num {
            let v_com_cur = &self.v_coms[i];
            v_com = v_com + v_com_cur.mul(factor);
            factor = factor * z;
        }

        let litebullet =
        ZkLiteBullet::new(v_com, len);
        
        let (
            check1,
            a_reduce_com,
            b_reduce_com,
            challenges_inv,
            challenges,  
        ) = litebullet.verify_as_subprotocol(
            srs,
            trans_seq,
        );


        let xl_prime = challenges_inv[0..log_m].to_vec();
        let xr_prime = challenges_inv[log_m..(log_m + log_n)].to_vec();

        let mut factor = F::one();

        let mut hat_b = F::zero();

        for i in 0..num {
            let (xl, xr) = &self.points[i];
           
            if (xl.len() != log_m) || (xr.len() != log_n) {
                panic!("BatchPoints: point size inconsistent");
            }

            let mut xx = xl.clone();
            xx.extend(xr.iter().cloned());

            hat_b = hat_b
            + factor * xi::xi_ip_from_challenges(&xx, &challenges);

            factor = factor * z;
        }

        let b_tilde_com = b_reduce_com - srs.com_base.mul(hat_b);

    

        let schnorr =
        ZkSchnorr::new(
            srs,
            b_tilde_com,
        );

        let check2 = schnorr.verify_as_subprotocol(trans_seq);

        // println!("check1: {}, check2: {}", check1, check2);
        
        let check = check1 && check2;

        (check, a_reduce_com, (xl_prime, xr_prime))
    
    }

    pub fn verify(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>
    ) -> bool {

        if trans_seq.check_fiat_shamir() == false {
            println!("!! Fiat shamir check failed when verifying BatchPoints");
            return false;
        }

        return self.verify_as_subprotocol(srs, trans_seq).0;
    }

}


impl<I1, I2, F, G> BatchMulMats<I1, I2, F, G>
where 
    I1: ShortInt,
    I2: ShortInt,
    F: PrimeField + From<I1> + From<I2>,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new() -> Self {
        Self {
            v_coms: Vec::new(),
            points: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push_point(
        &mut self,
        v_com: G,
        point: (Vec<F>, Vec<F>)
    ) {
        self.v_coms.push(v_com);
        self.points.push(point);
    }

    pub fn reduce_prover(
        &self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mats1: &Vec<DenseMatCM<I1, F>>,
        mats2: &Vec<DenseMatCM<I2, F>>,
        v_tildes: &Vec<F>,
    ) -> (F, Vec<Vec<F>>, G, F, (Vec<F>, Vec<F>))
    {
        let num = self.points.len();
        if self.v_coms.len() != num
            || mats1.len() + mats2.len() != num
            || v_tildes.len() != num
            || num == 0
        {
            panic!("BatchPoints Prover: invalid num");
        }

        let num1 = mats1.len();
        let num2 = mats2.len();

        let mut log_m_max = 0;
        let mut log_n_max = 0;

        let mut shape_vec = Vec::new();

        let mut subprotocols1 = Vec::new();
        let mut subprotocols2 = Vec::new();

        let mut challenges_n = Vec::new();
        let mut challenges_m = Vec::new();
        let mut challenges_inv_n = Vec::new();
        let mut challenges_inv_m = Vec::new();
        

        for idx in 0..num1 {
            let (xl, xr) = &self.points[idx];
            let log_m_cur = xl.len();
            let log_n_cur = xr.len();
            
            shape_vec.push((log_m_cur, log_n_cur));

            if log_m_cur > log_m_max {
                log_m_max = log_m_cur;
            }

            if log_n_cur > log_n_max {
                log_n_max = log_n_cur;
            }

            let v_com_cur = self.v_coms[idx];

            let v_tilde_cur = v_tildes[idx];

            let mut curprotocol =
            ZkProj::<I1, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            let mat_cur = &mats1[idx];

            curprotocol.prover_prepare_n(
                zk_trans_seq,
                mat_cur,
                v_tilde_cur,
            );

            subprotocols1.push(curprotocol);
        }

        for idx in 0..num2 {
            let (xl, xr) = &self.points[num1 + idx];
            let log_m_cur = xl.len();
            let log_n_cur = xr.len();
            
            shape_vec.push((log_m_cur, log_n_cur));

            if log_m_cur > log_m_max {
                log_m_max = log_m_cur;
            }

            if log_n_cur > log_n_max {
                log_n_max = log_n_cur;
            }

            let v_com_cur = self.v_coms[num1 + idx];

            let v_tilde_cur = v_tildes[num1 + idx];

            let mut curprotocol =
            ZkProj::<I2, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            let mat_cur = &mats2[idx];

            curprotocol.prover_prepare_n(
                zk_trans_seq,
                mat_cur,
                v_tilde_cur,
            );

            subprotocols2.push(curprotocol);
        }

        for j in 0..log_n_max {
            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();
            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);
            
            for idx in 0..num1 {
                
                let log_n_cur = shape_vec[idx].1;

                let j_cur =
                j as MyInt - (log_n_max - log_n_cur) as MyInt;

                if j_cur >=0 {
                    let j_cur = j_cur.abs() as usize;
                    subprotocols1[idx].prover_j_in_n_iteration(
                        zk_trans_seq,
                        j_cur,
                        x_j,
                    );
                }
            }

            for idx in 0..num2 {
                let log_n_cur = shape_vec[num1 + idx].1;

                let j_cur =
                j as MyInt - (log_n_max - log_n_cur) as MyInt;

                if j_cur >=0 {
                    let j_cur = j_cur.abs() as usize;
                    subprotocols2[idx].prover_j_in_n_iteration(
                        zk_trans_seq,
                        j_cur,
                        x_j,
                    );
                }
            }
        }

        for idx in 0..num1 {
            let mat_cur = &mats1[idx];
            subprotocols1[idx].prover_prepare_m(
            zk_trans_seq,
            mat_cur,
            );
        }

        for idx in 0..num2 {
            let mat_cur = &mats2[idx];
            subprotocols2[idx].prover_prepare_m(
                zk_trans_seq,
                mat_cur,
            )
        }

        for j in 0..log_m_max {
            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();
            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);

            for idx in 0..num1 {
                let log_m_cur = shape_vec[idx].0;
                let j_cur =
                j as MyInt - (log_m_max - log_m_cur) as MyInt;

                if j_cur >=0 {
                    let j_cur = j_cur.abs() as usize;

                    subprotocols1[idx].prover_j_in_m_iteration(
                        zk_trans_seq,
                        j_cur,
                        x_j,
                    );
                }
            }

            for idx in 0..num2 {
                let log_m_cur = shape_vec[num1 + idx].0;
                let j_cur =
                j as MyInt - (log_m_max - log_m_cur) as MyInt;

                if j_cur >=0 {
                    let j_cur = j_cur.abs() as usize;

                    subprotocols2[idx].prover_j_in_m_iteration(
                        zk_trans_seq,
                        j_cur,
                        x_j,
                    );
                }
            }
        }

        for idx in 0..num1 {
            subprotocols1[idx].prover_conclude(
                srs,
                zk_trans_seq,
            );
        }

        for idx in 0..num2 {
            subprotocols2[idx].prover_conclude(
                srs,
                zk_trans_seq,
            );
        }

        // z for batch proof
        let z = zk_trans_seq.gen_challenge();
        let mut factor = F::one();

        let mut hat_a_com_sum = G::zero();
        let mut hat_a_tilde_sum = F::zero();

        let max_m = 1 << log_m_max;
        let max_n = 1 << log_n_max;

        let mut mat_sum = vec![vec![F::zero(); max_m]; max_n];


        for idx in 0..num1 {
            let hat_a_com_cur =
            subprotocols1[idx].prover_intermediate.a_reduce_blind;

            let hat_a_tilde_cur =
            subprotocols1[idx].prover_intermediate.a_reduce_tilde;

            hat_a_com_sum = hat_a_com_sum + hat_a_com_cur.mul(factor);
            hat_a_tilde_sum = hat_a_tilde_sum + hat_a_tilde_cur.mul(factor);
          
            mat_sum = linear::mat_scalar_addition::<I1,F>(
                &mat_sum,
                &mats1[idx].data,
                factor,
            );

            factor = factor * z;
        }

        for idx in 0..num2 {
            let hat_a_com_cur =
            subprotocols2[idx].prover_intermediate.a_reduce_blind;

            let hat_a_tilde_cur =
            subprotocols2[idx].prover_intermediate.a_reduce_tilde;

            hat_a_com_sum = hat_a_com_sum + hat_a_com_cur.mul(factor);
            hat_a_tilde_sum = hat_a_tilde_sum + hat_a_tilde_cur.mul(factor);
          
            mat_sum = linear::mat_scalar_addition::<I2,F>(
                &mat_sum,
                &mats2[idx].data,
                factor,
            );

            factor = factor * z;
        }


        (
            z,
            mat_sum,
            hat_a_com_sum,
            hat_a_tilde_sum,
            (
                challenges_inv_m.clone(),
                challenges_inv_n.clone(),
            )
        )

    }

    pub fn verify_as_subprotocol(
        &mut self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
        num1 : usize,
    ) -> (bool, F, G, (Vec<F>, Vec<F>)) {

        let num = self.points.len();
        if self.v_coms.len() != num
            || num == 0
        {
            panic!("BatchPoints Prover: invalid num");
        }

        let num2 = num - num1;

        let mut log_m_max = 0;
        let mut log_n_max = 0;

        let mut shape_vec = Vec::new();

        let mut subprotocols1 = Vec::new();
        let mut subprotocols2 = Vec::new();

        let mut challenges_n = Vec::new();
        let mut challenges_m = Vec::new();
        let mut challenges_inv_n = Vec::new();
        let mut challenges_inv_m = Vec::new();
        

        for idx in 0..num1 {
            let (xl, xr) = &self.points[idx];
            let log_m_cur = xl.len();
            let log_n_cur = xr.len();
            
            shape_vec.push((log_m_cur, log_n_cur));

            if log_m_cur > log_m_max {
                log_m_max = log_m_cur;
            }

            if log_n_cur > log_n_max {
                log_n_max = log_n_cur;
            }

            let v_com_cur = self.v_coms[idx];


            let mut curprotocol =
            ZkProj::<I1, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            curprotocol.verifer_prepare(
                trans_seq,
            );

            subprotocols1.push(curprotocol);
        }


        for idx in 0..num2 {
            let (xl, xr) = &self.points[num1 + idx];
            let log_m_cur = xl.len();
            let log_n_cur = xr.len();
            
            shape_vec.push((log_m_cur, log_n_cur));

            if log_m_cur > log_m_max {
                log_m_max = log_m_cur;
            }

            if log_n_cur > log_n_max {
                log_n_max = log_n_cur;
            }

            let v_com_cur = self.v_coms[num1 + idx];


            let mut curprotocol =
            ZkProj::<I2, F, G>::new(
                v_com_cur, 
                xl.clone(), 
                xr.clone()
            );

            curprotocol.verifer_prepare(
                trans_seq,
            );

            subprotocols2.push(curprotocol);
        }


        for j in 0..log_n_max {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                let x_j_inv = x_j.inverse().unwrap();
                challenges_n.push(x_j);
                challenges_inv_n.push(x_j_inv);

                for idx in 0..num1 {
                    // println!("idx: {}, j: {}", idx, j);

                    let log_n_cur = shape_vec[idx].1;
                    let j_cur =
                    j as MyInt - (log_n_max - log_n_cur) as MyInt;

                    if j_cur >=0 {
                        subprotocols1[idx].verifier_j_in_n_iteration(
                            trans_seq,
                            x_j,
                            false,
                        );
                    }
                }



                for idx in 0..num2 {
                    // println!("idx: {}, j: {}", idx, j);

                    let log_n_cur = shape_vec[num1 + idx].1;
                    let j_cur =
                    j as MyInt - (log_n_max - log_n_cur) as MyInt;

                    if j_cur >=0 {
                        subprotocols2[idx].verifier_j_in_n_iteration(
                            trans_seq,
                            x_j,
                            false,
                        );
                    }
                }
            } else {
                println!("!! Invalid transcript when verifying BatchMulMats");
            }
        }

        // println!("log_m_max: {}", log_m_max);

        for j in 0..log_m_max {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                let last_j =
                j == log_m_max - 1;

                let x_j_inv = x_j.inverse().unwrap();
                challenges_m.push(x_j);
                challenges_inv_m.push(x_j_inv);

                for idx in 0..num1 {
                    // println!("idx: {}, jm: {}", idx, j);

                    let log_m_cur = shape_vec[idx].0;
                    let j_cur =
                    j as MyInt - (log_m_max - log_m_cur) as MyInt;

                    // println!("j_cur: {}", j_cur);

                    if j_cur >=0 {

                        subprotocols1[idx].verifier_j_in_m_iteration(
                            trans_seq,
                            x_j,
                            last_j,
                        );
                    }
                }

                for idx in 0..num2 {
                    // println!("idx: {}, jm: {}", idx, j);

                    let log_m_cur = shape_vec[num1 + idx].0;
                    let j_cur =
                    j as MyInt - (log_m_max - log_m_cur) as MyInt;

                    // println!("j_cur: {}", j_cur);

                    if j_cur >=0 {

                        subprotocols2[idx].verifier_j_in_m_iteration(
                            trans_seq,
                            x_j,
                            last_j,
                        );
                    }
                }
                
            } else {
                println!("!! Invalid transcript when verifying BatchMulMats");
            }

            
        }

        for idx in 0..num1 {
            subprotocols1[idx].verifier_conclude(
                srs,
                trans_seq,
            );
        }

        for idx in 0..num2 {
            subprotocols2[idx].verifier_conclude(
                srs,
                trans_seq,
            );
        }

        let mut hat_a_com_sum = G::zero();

        let mut z: F = F::zero();

        if let (
            TranElem::Coin(z_value),
        ) = (
            trans_seq.data[trans_seq.pointer].clone(),
        ) {
            trans_seq.pointer += 1;

            z = z_value;

            let mut factor = F::one();

            for idx in 0..num1 {
                let hat_a_com_cur =
                subprotocols1[idx].verifier_intermediate.a_reduce_blind;
                hat_a_com_sum = hat_a_com_sum + hat_a_com_cur.mul(factor);
                factor = factor * z;
            }

            for idx in 0..num2 {
                let hat_a_com_cur =
                subprotocols2[idx].verifier_intermediate.a_reduce_blind;
                hat_a_com_sum = hat_a_com_sum + hat_a_com_cur.mul(factor);
                factor = factor * z;
            }

        } else {
            println!("!! Invalid transcript when verifying BatchMulMats");
        }

        let mut flag = true;

        for idx in 0..num1 {
            // let cur_flag = subprotocols[idx].verifier_intermediate.flag;
            // println!("idx: {}, cur_flag: {}", idx, cur_flag);

            flag = flag && subprotocols1[idx].verifier_intermediate.flag;
        }

        for idx in 0..num2 {
            // let cur_flag = subprotocols[idx].verifier_intermediate.flag;
            // println!("idx: {}, cur_flag: {}", idx, cur_flag);

            flag = flag && subprotocols2[idx].verifier_intermediate.flag;
        }

        (
            flag,
            z,
            hat_a_com_sum,
            (
                challenges_inv_m.clone(),
                challenges_inv_n.clone()
            )
        )

    }

}
#[cfg(test)]
mod tests {
    
    use crate::{data_structures::SRS, utils::matdef::DenseMatCM};

    use super::*;

    #[test]
    fn test_batch_mulpoints() {

        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;
        

        let rng = &mut ark_std::rand::thread_rng();

        let logm = 4 as usize;
        let m = 1 << logm;

        let logn = 3 as usize;
        let n = 1 << logn;

        let mut mat =
        DenseMatCM::<MyInt, F>::new(m, n);
        mat.gen_rand(8);

        let base = G::rand(rng);
        let blind_base = G::rand(rng);

        let zksrs =
        &ZkSRS::new(base, blind_base);

        let mut v_tilde_vec = Vec::new();

        let mut protocol =
        BatchPoints::<MyInt, F, G>::new();

        for _ in 0..3 {
            let xl = (0..logm).map(|_| F::rand(rng)).collect();
            let xr = (0..logn).map(|_| F::rand(rng)).collect();

            let v =
            mat.proj_lr(&xl, &xr);
            
            let v_tilde =
            <E as Pairing>::ScalarField::rand(rng);
    
            let v_com = base.mul(v) + blind_base.mul(v_tilde);

            v_tilde_vec.push(v_tilde);    
            
            protocol.push_point(v_com, (xl, xr));
        
        }


        let zk_trans_seq =
        &mut ZkTranSeq::new(zksrs);


        let (
            _,
            a_reduce_tilde,
            (_, _),
        ) = protocol.reduce_prover(
            zksrs,
            zk_trans_seq,
            &mat,
            v_tilde_vec,
        );

        let mut trans_seq =
        zk_trans_seq.publish_trans();
        
        let (
            flag,
            hat_a_com,
            (xl_prime, xr_prime),
        ) = protocol.verify_as_subprotocol(
            zksrs,
            &mut trans_seq,
        );

        assert_eq!(flag, true);

        let hat_a =
        mat.proj_lr(&xl_prime, &xr_prime);

        let hat_a_com_check = 
        base.mul(hat_a) + blind_base.mul(a_reduce_tilde);

        assert_eq!(hat_a_com, hat_a_com_check);
     
        println!(" * Verification of BatchPoints passed");

        assert_eq!(trans_seq.pointer, trans_seq.data.len());

    }

    #[test]
    fn test_batch_proj() {

        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;
        

        let rng = &mut ark_std::rand::thread_rng();

        let logm = 4 as usize;
        let m = 1 << logm;

        let logn = 5 as usize;
        let n = 1 << logn;

        let mut mat =
        RotationMatIndexFormat::<MyInt, F>::new(m, n);
        mat.gen_rand(8);

        let base = G::rand(rng);
        let blind_base = G::rand(rng);

        let zksrs =
        &ZkSRS::new(base, blind_base);

        let mut v_tilde_vec = Vec::new();

        let mut protocol =
        BatchPointsRotation::<MyInt, F, G>::new();

        for _ in 0..3 {
            let xl = (0..logm).map(|_| F::rand(rng)).collect();
            let xr = (0..logn).map(|_| F::rand(rng)).collect();

            let v =
            mat.proj_lr(&xl, &xr);
            
            let v_tilde =
            <E as Pairing>::ScalarField::rand(rng);
    
            let v_com = base.mul(v) + blind_base.mul(v_tilde);

            v_tilde_vec.push(v_tilde);    
            
            protocol.push_point(v_com, (xl, xr));
        
        }


        let zk_trans_seq =
        &mut ZkTranSeq::new(zksrs);


        let (
            _,
            a_reduce_tilde,
            (_, _),
        ) = protocol.reduce_prover(
            zksrs,
            zk_trans_seq,
            &mat,
            v_tilde_vec,
        );

        let mut trans_seq =
        zk_trans_seq.publish_trans();
        
        let (
            flag,
            hat_a_com,
            (xl_prime, xr_prime),
        ) = protocol.verify_as_subprotocol(
            zksrs,
            &mut trans_seq,
        );

        assert_eq!(flag, true);

        let hat_a =
        mat.proj_lr(&xl_prime, &xr_prime);

        let hat_a_com_check = 
        base.mul(hat_a) + blind_base.mul(a_reduce_tilde);

        assert_eq!(hat_a_com, hat_a_com_check);
     
        println!(" * Verification of BatchPoints passed");

        assert_eq!(trans_seq.pointer, trans_seq.data.len());

    }

    #[test]
    fn test_batch_mats() {

        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;
        

        let rng = &mut ark_std::rand::thread_rng();

        let logm = 4 as usize;
        let m = 1 << logm;

        let logn = 3 as usize;
        let n = 1 << logn;

        let mut mat =
        DenseMatCM::<MyInt, F>::new(m, n);
        mat.gen_rand(8);

        let base = G::rand(rng);
        let blind_base = G::rand(rng);

        let zksrs =
        &ZkSRS::new(base, blind_base);

        let xl = (0..logm).map(|_| F::rand(rng)).collect();
        let xr = (0..logn).map(|_| F::rand(rng)).collect();
        let l_vec = xi::xi_from_challenges(&xl);
        let r_vec = xi::xi_from_challenges(&xr);

        let v =
        mat.proj_lr(&xl, &xr);

        let la = mat.proj_left(&l_vec);
        let v1 = linear::inner_product(&la, &r_vec);
        assert_eq!(v, v1);
        
        let v_tilde =
        <E as Pairing>::ScalarField::rand(rng);

        let v_com = base.mul(v) + blind_base.mul(v_tilde);

        let mut ip =
        BatchMulMats::<MyInt, bool, F, G>::new();

        ip.push_point(v_com, (xl, xr));


        let logm1 = 2 as usize;
        let m1 = 1 << logm1;

        let logn1 = 5 as usize;
        let n1 = 1 << logn1;

        let mut mat1 =
        DenseMatCM::<bool, F>::new(m1, n1);
        mat1.gen_rand(8);

        let xl1 = (0..logm1).map(|_| F::rand(rng)).collect();
        let xr1 = (0..logn1).map(|_| F::rand(rng)).collect();
        let l_vec1 = xi::xi_from_challenges(&xl1);
        let r_vec1 = xi::xi_from_challenges(&xr1);

        let v1 =
        mat1.proj_lr(&xl1, &xr1);

        // println!("we are here");

        let la1 = mat1.proj_left(&l_vec1);
        let v11 = linear::inner_product(&la1, &r_vec1);
        assert_eq!(v1, v11);
        
        let v_tilde1 =
        <E as Pairing>::ScalarField::rand(rng);

        let v_com1 = base.mul(v1) + blind_base.mul(v_tilde1);

        ip.push_point(v_com1, (xl1, xr1));


        let zk_trans_seq =
        &mut ZkTranSeq::new(zksrs);

        let mats1 =
        vec![mat.clone()];
        let mats2 =
        vec![mat1.clone()];
        let v_tildes =
        vec![v_tilde.clone(),v_tilde1.clone()];

        let (
            z,
            mat_sum,
            hat_a_com_sum,
            hat_a_tilde_sum,
            (xl_prime, xr_prime),
        ) = ip.reduce_prover(
            zksrs,
            zk_trans_seq,
            &mats1,
            &mats2,
            &v_tildes,
        );

        let trans_seq =
        &mut zk_trans_seq.publish_trans();

        let (
            flag,
            z_check,
            hat_a_com_sum_check,
            (xl_prime_check, xr_prime_check),
        ) = ip.verify_as_subprotocol(
            zksrs,
            trans_seq,
            1,
        );
        assert_eq!(trans_seq.pointer, trans_seq.data.len());
        assert_eq!(flag, true);


        let l_vec_prime =
        xi::xi_from_challenges(&xl_prime);
        let la_sum_check =
        linear::proj_left_cm(&mat_sum, &l_vec_prime);
        let r_vec_prime =
        xi::xi_from_challenges(&xr_prime);
        let hat_sum_a_check =
        linear::inner_product(&la_sum_check, &r_vec_prime);

        let hat_a =
        mat.proj_lr(&xl_prime, &xr_prime);
        let hat_a_1 =
        mat1.proj_lr(&xl_prime, &xr_prime);

        let hat_a = hat_a + hat_a_1 * z;

        assert_eq!(hat_a, hat_sum_a_check);


        let tilde_hat_a_sum = hat_a_tilde_sum;
        let hat_a_com_check_sum = 
        base.mul(hat_a) + blind_base.mul(tilde_hat_a_sum);

        // println!("hat_a: {:?}", hat_a);
        // println!("hat_a_tilde: {:?}", tilde_hat_a_sum);

      

        assert_eq!(z, z_check);
        assert_eq!(hat_a_com_sum, hat_a_com_sum_check);
        assert_eq!(xl_prime, xl_prime_check);
        assert_eq!(xr_prime, xr_prime_check);

        assert_eq!(hat_a_com_sum, hat_a_com_check_sum);

     
        println!(" * Verification of BatchMats passed");

    }


    #[test]
    fn test_batch() {

        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        let srs =
        SRS::<E>::setup(5);

        let pcspp =
        srs.pc_pp;

        let zksrs =
        srs.zksrs;

        let base = zksrs.com_base;
        let blind_base = zksrs.blind_base;

        let mut mat_container =
        MatContainer::<F>::new(8);

        let mut com_container =
        ComContainer::<E>::new();

        let mut points_container =
        PointsContainer::<F>::new();

        let mut eval_container =
        EvalContainer::<F, G>::new();


        let logm1 = 4 as usize;
        let m1 = 1 << logm1;

        let logn1 = 3 as usize;
        let n1 = 1 << logn1;

        let mut mat1 =
        DenseMatCM::<MyInt, F>::new(m1, n1);
        mat1.gen_rand(8);

        let xl1 = (0..logm1).map(|_| F::rand(rng)).collect();
        let xr1 = (0..logn1).map(|_| F::rand(rng)).collect();
        let l_vec1 = xi::xi_from_challenges(&xl1);
        let r_vec1 = xi::xi_from_challenges(&xr1);

        let v1 =
        mat1.proj_lr(&xl1, &xr1);

        let la1 = mat1.proj_left(&l_vec1);
        let v11 = linear::inner_product(&la1, &r_vec1);
        assert_eq!(v1, v11);
        
        let v_tilde1 =
        <E as Pairing>::ScalarField::rand(rng);

        let v_com1 = base.mul(v1) + blind_base.mul(v_tilde1);

        mat_container.dense_myint.push(mat1.clone());
        eval_container.dense_myint.push((v_com1, v_tilde1));
        points_container.dense_myint.push((xl1, xr1));


        let logm2 = 2 as usize;
        let m2 = 1 << logm2;

        let logn2 = 5 as usize;
        let n2 = 1 << logn2;

        let mut mat2 =
        DenseMatCM::<bool, F>::new(m2, n2);
        mat2.gen_rand(8);

        let xl2 = (0..logm2).map(|_| F::rand(rng)).collect();
        let xr2 = (0..logn2).map(|_| F::rand(rng)).collect();
        let l_vec2 = xi::xi_from_challenges(&xl2);
        let r_vec2 = xi::xi_from_challenges(&xr2);

        let v2 =
        mat2.proj_lr(&xl2, &xr2);

        let la2 = mat2.proj_left(&l_vec2);
        let v12 = linear::inner_product(&la2, &r_vec2);
        assert_eq!(v2, v12);
        
        let v_tilde2 =
        <E as Pairing>::ScalarField::rand(rng);

        let v_com2 = base.mul(v2) + blind_base.mul(v_tilde2);

        mat_container.dense_bool.push(mat2.clone());
        eval_container.dense_bool.push((v_com2, v_tilde2));
        points_container.dense_bool.push((xl2, xr2));

        commit_mats(
            &pcspp,
            &mat_container,
            &mut com_container
        );

        let zk_trans_seq =
        &mut ZkTranSeq::new(&zksrs);

        let pcs_proof = open_pc_mats(
            &zksrs,
            &pcspp,
            zk_trans_seq,
            &mut mat_container,
            &mut com_container,
            &mut eval_container,
            &mut points_container
        );

        let trans_seq =
        &mut zk_trans_seq.publish_trans();

        let check = verify_pc_mats(
            &zksrs,
            &pcspp,
            trans_seq,
            &pcs_proof,
            &mut com_container,
            &mut eval_container,
            &mut points_container,
            1,
        );

        assert_eq!(check, true);

        println!(" * Verification of BatchMats passed");

    }
    
}