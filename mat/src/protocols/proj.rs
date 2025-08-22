// Implement the litebullet protocol
//
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::protocols::scalars::ZkSemiMulScalar;
use crate::utils::matdef::{MatOps, ShortInt};
use crate::utils::zktr::{ZkTranSeq, TranSeq, TranElem};
use crate::utils::linear;
use crate::utils::xi;

// Implementation of the protocol to
// prove the projection of a matrix 
// on public vectors
// Reduce to MLPC of the witness matrix
// 
pub struct ZkProj<I, F, G>
where 
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_com: G,
    pub xl: Vec<F>,
    pub xr: Vec<F>,
    pub prover_intermediate: ZkProjProverIntermediate<F, G>,
    pub verifier_intermediate: ZkProjVerifierIntermediate<F, G>,
    _marker: PhantomData<I>,
}

pub struct ZkProjProverIntermediate<F, G>
where 
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub m: usize,
    pub n: usize,
    pub log_m: usize,
    pub log_n: usize,
    pub challenges_n: Vec<F>,
    pub challenges_inv_n: Vec<F>,
    pub challenges_m: Vec<F>,
    pub challenges_inv_m: Vec<F>,
    pub a_reduce_blind: G,
    pub a_reduce_tilde: F,
    pub vec_u_current: Vec<F>,
    pub vec_r_current: Vec<F>,
    pub vec_a_current: Vec<F>,
    pub vec_l_current: Vec<F>,
    pub lhs_tilde: F,
    pub lhs_com: G,
    pub l_reduce: F,
    pub r_reduce: F,
    pub a_reduce: F,
    pub previous_l_com: G,
    pub previous_r_com: G,
    pub previous_l_tilde: F,
    pub previous_r_tilde: F,
    pub previous_lh1: Vec<F>,
    pub previous_rh1: Vec<F>,
    pub previous_lh2: Vec<F>,
    pub previous_rh2: Vec<F>,
}


pub struct ZkProjVerifierIntermediate<F, G>
where 
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub m: usize,
    pub n: usize,
    pub log_m: usize,
    pub log_n: usize,
    pub challenges_n: Vec<F>,
    pub challenges_inv_n: Vec<F>,
    pub challenges_m: Vec<F>,
    pub challenges_inv_m: Vec<F>,
    pub a_reduce_blind: G,
    pub lhs_com: G,
    pub previous_l_com: G,
    pub previous_r_com: G,
    pub flag: bool,
}

impl<F, G> ZkProjProverIntermediate<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new() -> Self {
        Self {
            m: 0,
            n: 0,
            log_m: 0,
            log_n: 0,
            challenges_n: Vec::new(),
            challenges_inv_n: Vec::new(),
            challenges_m: Vec::new(),
            challenges_inv_m: Vec::new(),
            a_reduce_blind: G::zero(),
            a_reduce_tilde: F::zero(),
            vec_u_current: Vec::new(),
            vec_r_current: Vec::new(),
            vec_a_current: Vec::new(),
            vec_l_current: Vec::new(),
            lhs_tilde: F::zero(),
            lhs_com: G::zero(),
            l_reduce: F::zero(),
            r_reduce: F::zero(),
            a_reduce: F::zero(),
            previous_l_com: G::zero(),
            previous_r_com: G::zero(),
            previous_l_tilde: F::zero(),
            previous_r_tilde: F::zero(),
            previous_lh1: Vec::new(),
            previous_rh1: Vec::new(),
            previous_lh2: Vec::new(),
            previous_rh2: Vec::new(),
        }
    }
}


impl<F,G> ZkProjVerifierIntermediate<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new() -> Self {
        Self {
            m: 0,
            n: 0,
            log_m: 0,
            log_n: 0,
            challenges_n: Vec::new(),
            challenges_inv_n: Vec::new(),
            challenges_m: Vec::new(),
            challenges_inv_m: Vec::new(),
            a_reduce_blind: G::zero(),
            lhs_com: G::zero(),
            previous_l_com: G::zero(),
            previous_r_com: G::zero(),
            flag: false,
        }
    }
}

impl<I, F, G> ZkProj<I, F, G> 
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(
        v_com_value: G,
        xl_value: Vec<F>,
        xr_value: Vec<F>,
    ) -> Self {
        Self {
            v_com: v_com_value,
            xl: xl_value,
            xr: xr_value,
            prover_intermediate: ZkProjProverIntermediate::new(),
            verifier_intermediate: ZkProjVerifierIntermediate::new(),
            _marker: PhantomData,
        }
    }

    pub fn prover_prepare_n<M>(
        &mut self,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &M,
        tilde_v: F,
    )
    where
        M: MatOps<I,F>,
    {
        zk_trans_seq.push_com(self.v_com);

        let log_m = self.xl.len();
        let log_n = self.xr.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        if m & (m - 1) != 0 
        || n & (n - 1) != 0 {
            panic!("Length of the projection vector is not a power of 2");
        }
        
        if m != mat.get_shape().0 
        || n != mat.get_shape().1
        {
            panic!("Length of the projection vector does not match the matrix");
        }
        

        self.prover_intermediate.m = m;
        self.prover_intermediate.n = n;
        self.prover_intermediate.log_m = (m as u64).ilog2() as usize;
        self.prover_intermediate.log_n = (n as u64).ilog2() as usize;
        
        self.prover_intermediate.vec_u_current =
        mat.proj_left_challenges(&self.xl);
        self.prover_intermediate.vec_r_current =
        xi::xi_from_challenges::<F>(&self.xr);

        self.prover_intermediate.challenges_n = Vec::new();
        self.prover_intermediate.challenges_inv_n = Vec::new();

        self.prover_intermediate.lhs_tilde = tilde_v;
        self.prover_intermediate.lhs_com = self.v_com.clone();

        let current_len = self.prover_intermediate.n;
            
        let u_left = 
        self.prover_intermediate.vec_u_current[0..current_len/2].to_vec();
        let u_right = 
        self.prover_intermediate.vec_u_current[current_len/2..current_len].to_vec();
        
        let r_left = 
        self.prover_intermediate.vec_r_current[0..current_len/2].to_vec();
        let r_right = 
        self.prover_intermediate.vec_r_current[current_len/2..current_len].to_vec();
        
        let l_tr = 
            linear::inner_product(&u_left, &r_right);
        let r_tr = 
            linear::inner_product(&u_right, &r_left);
            

        let (l_com,l_tilde) = zk_trans_seq
        .push_gen_blinding(l_tr);
        let (r_com,r_tilde) = zk_trans_seq
        .push_gen_blinding(r_tr);

        self.prover_intermediate.previous_l_com = l_com;
        self.prover_intermediate.previous_r_com = r_com;
        self.prover_intermediate.previous_l_tilde = l_tilde;
        self.prover_intermediate.previous_r_tilde = r_tilde;
        self.prover_intermediate.previous_lh1 = u_left.clone();
        self.prover_intermediate.previous_rh1 = u_right.clone();
        self.prover_intermediate.previous_lh2 = r_left.clone();
        self.prover_intermediate.previous_rh2 = r_right.clone();

    }

    pub fn prover_j_in_n_iteration(
        &mut self,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        j: usize,
        x_j: F,
    ) {

        let x_j_inv = x_j.inverse().unwrap();

        self.prover_intermediate.challenges_n.push(x_j);
        self.prover_intermediate.challenges_inv_n.push(x_j_inv);

        self.prover_intermediate.lhs_tilde =
        self.prover_intermediate.lhs_tilde
        +  self.prover_intermediate.previous_l_tilde * x_j
        + self.prover_intermediate.previous_r_tilde * x_j_inv;
        self.prover_intermediate.lhs_com =
        self.prover_intermediate.lhs_com
        + self.prover_intermediate.previous_l_com.mul(&x_j)
        + self.prover_intermediate.previous_r_com.mul(&x_j_inv);


        self.prover_intermediate.vec_u_current = linear::vec_addition(
            &self.prover_intermediate.previous_lh1,
            &linear::vec_scalar_mul(
                &self.prover_intermediate.previous_rh1, &x_j_inv),
        );

        self.prover_intermediate.vec_r_current = linear::vec_addition(
            &self.prover_intermediate.previous_lh2,
            &linear::vec_scalar_mul(
                &self.prover_intermediate.previous_rh2, &x_j),
        );

        if j < self.prover_intermediate.log_n - 1 {
            
        let current_len = self.prover_intermediate.n / 2usize.pow((j + 1) as u32);
            
        let u_left = 
        self.prover_intermediate.vec_u_current[0..current_len/2].to_vec();
        let u_right = 
        self.prover_intermediate.vec_u_current[current_len/2..current_len].to_vec();
        
        let r_left = 
        self.prover_intermediate.vec_r_current[0..current_len/2].to_vec();
        let r_right = 
        self.prover_intermediate.vec_r_current[current_len/2..current_len].to_vec();
        
        let l_tr = 
            linear::inner_product(&u_left, &r_right);
        let r_tr = 
            linear::inner_product(&u_right, &r_left);
            

        let (l_com,l_tilde) = zk_trans_seq
        .push_gen_blinding(l_tr);
        let (r_com,r_tilde) = zk_trans_seq
        .push_gen_blinding(r_tr);

        self.prover_intermediate.previous_l_com = l_com;
        self.prover_intermediate.previous_r_com = r_com;
        self.prover_intermediate.previous_l_tilde = l_tilde;
        self.prover_intermediate.previous_r_tilde = r_tilde;
        self.prover_intermediate.previous_lh1 = u_left.clone();
        self.prover_intermediate.previous_rh1 = u_right.clone();
        self.prover_intermediate.previous_lh2 = r_left.clone();
        self.prover_intermediate.previous_rh2 = r_right.clone();

        }
    }

    pub fn prover_prepare_m<M>(
        &mut self,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &M,
    )
    where
        M: MatOps<I,F>,
    {
        self.prover_intermediate.r_reduce = self.prover_intermediate.vec_r_current[0];

        let a_xi_inv = mat.proj_right_challenges(&self.prover_intermediate.challenges_inv_n);
        
        let l_vec =
        xi::xi_from_challenges::<F>(&self.xl);

        self.prover_intermediate.vec_a_current= a_xi_inv.to_vec();
        self.prover_intermediate.vec_l_current = l_vec[0..self.prover_intermediate.m].to_vec();
        
        self.prover_intermediate.challenges_m= Vec::new();
        self.prover_intermediate.challenges_inv_m= Vec::new();

        let current_len = self.prover_intermediate.m;
            
        let a_left = 
        self.prover_intermediate.vec_a_current[0..current_len/2].to_vec();
        let a_right = 
        self.prover_intermediate.vec_a_current[current_len/2..current_len].to_vec();
        
        let l_left = 
        self.prover_intermediate.vec_l_current[0..current_len/2].to_vec();
        let l_right = 
        self.prover_intermediate.vec_l_current[current_len/2..current_len].to_vec();
        
        let l_tr = 
            linear::inner_product(&a_left, &l_right) * self.prover_intermediate.r_reduce;
        let r_tr = 
            linear::inner_product(&a_right, &l_left) * self.prover_intermediate.r_reduce;
            

        let (l_com,l_tilde) = zk_trans_seq
        .push_gen_blinding(l_tr);
        let (r_com,r_tilde) = zk_trans_seq
        .push_gen_blinding(r_tr);

        self.prover_intermediate.previous_l_com = l_com;
        self.prover_intermediate.previous_r_com = r_com;
        self.prover_intermediate.previous_l_tilde = l_tilde;
        self.prover_intermediate.previous_r_tilde = r_tilde;
        self.prover_intermediate.previous_lh1 = a_left.clone();
        self.prover_intermediate.previous_rh1 = a_right.clone();
        self.prover_intermediate.previous_lh2 = l_left.clone();
        self.prover_intermediate.previous_rh2 = l_right.clone();
    }

    pub fn prover_j_in_m_iteration(
        &mut self,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        j: usize,
        x_j: F,
    ) {
        let x_j_inv = x_j.inverse().unwrap();

        self.prover_intermediate.challenges_m.push(x_j);
        self.prover_intermediate.challenges_inv_m.push(x_j_inv);

        self.prover_intermediate.lhs_tilde =
        self.prover_intermediate.lhs_tilde
        +  self.prover_intermediate.previous_l_tilde * x_j
        + self.prover_intermediate.previous_r_tilde * x_j_inv;
        self.prover_intermediate.lhs_com =
        self.prover_intermediate.lhs_com
        + self.prover_intermediate.previous_l_com.mul(&x_j)
        + self.prover_intermediate.previous_r_com.mul(&x_j_inv);

        self.prover_intermediate.vec_a_current = linear::vec_addition(
            &self.prover_intermediate.previous_lh1,
            &linear::vec_scalar_mul(
                &self.prover_intermediate.previous_rh1, &x_j_inv),
        );

        self.prover_intermediate.vec_l_current = linear::vec_addition(
            &self.prover_intermediate.previous_lh2,
            &linear::vec_scalar_mul(
                &self.prover_intermediate.previous_rh2, &x_j),
        );

        if j < self.prover_intermediate.log_m - 1 {
            let current_len = self.prover_intermediate.m / 2usize.pow((j+1) as u32);
            
            let a_left = 
            self.prover_intermediate.vec_a_current[0..current_len/2].to_vec();
            let a_right = 
            self.prover_intermediate.vec_a_current[current_len/2..current_len].to_vec();
            
            let l_left = 
            self.prover_intermediate.vec_l_current[0..current_len/2].to_vec();
            let l_right = 
            self.prover_intermediate.vec_l_current[current_len/2..current_len].to_vec();
            
            let l_tr = 
                linear::inner_product(&a_left, &l_right)
                * self.prover_intermediate.r_reduce;
            let r_tr = 
                linear::inner_product(&a_right, &l_left)
                * self.prover_intermediate.r_reduce;
                
    
            let (l_com,l_tilde) = zk_trans_seq
            .push_gen_blinding(l_tr);
            let (r_com,r_tilde) = zk_trans_seq
            .push_gen_blinding(r_tr);

            self.prover_intermediate.previous_l_com = l_com;
            self.prover_intermediate.previous_r_com = r_com;
            self.prover_intermediate.previous_l_tilde = l_tilde;
            self.prover_intermediate.previous_r_tilde = r_tilde;
            self.prover_intermediate.previous_lh1 = a_left.clone();
            self.prover_intermediate.previous_rh1 = a_right.clone();
            self.prover_intermediate.previous_lh2 = l_left.clone();
            self.prover_intermediate.previous_rh2 = l_right.clone();
        }

    }

    pub fn prover_conclude (
        &mut self,
        srs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
    ) {

        self.prover_intermediate.a_reduce =
        self.prover_intermediate.vec_a_current[0];
        self.prover_intermediate.l_reduce =
        self.prover_intermediate.vec_l_current[0];


        let (a_reduce_blind, a_reduce_tilde) =
            zk_trans_seq.push_gen_blinding(
                self.prover_intermediate.a_reduce);
        

        let hat_b = self.prover_intermediate.l_reduce * self.prover_intermediate.r_reduce;
        
        let zk_semi_mul = ZkSemiMulScalar::new(
            srs,
            self.prover_intermediate.lhs_com,
            a_reduce_blind,
            hat_b,
        );

        self.prover_intermediate.a_reduce_blind = a_reduce_blind;
        self.prover_intermediate.a_reduce_tilde = a_reduce_tilde;

        // println!("Concluding a_reduce {}", self.prover_intermediate.a_reduce);
        // println!("Concluding a_tilde {}", self.prover_intermediate.a_reduce_tilde);

        zk_semi_mul.prove(
            zk_trans_seq,
            self.prover_intermediate.a_reduce,
            self.prover_intermediate.lhs_tilde,
            a_reduce_tilde,
        );

    }

    pub fn reduce_prover_split<M>(
        &mut self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &M,
        tilde_v: F,
    ) -> (G, F, Vec<F>, Vec<F>)
    where
        M: MatOps<I,F>,
    {
        self.prover_prepare_n(
            zk_trans_seq,
            mat,
            tilde_v,
        );

        for j in 0..self.prover_intermediate.log_n {
            let x_j = zk_trans_seq.gen_challenge();
            self.prover_j_in_n_iteration(
                zk_trans_seq,
                j,
                x_j,
            );
        }

        self.prover_prepare_m(
            zk_trans_seq,
            mat,
        );

        for j in 0..self.prover_intermediate.log_m {
            let x_j = zk_trans_seq.gen_challenge();
            self.prover_j_in_m_iteration(
                zk_trans_seq,
                j,
                x_j,
            );
        }

        self.prover_conclude(
            srs,
            zk_trans_seq,
        );

        (
            self.prover_intermediate.a_reduce_blind,
            self.prover_intermediate.a_reduce_tilde,
            self.prover_intermediate.challenges_inv_m.clone(),
            self.prover_intermediate.challenges_inv_n.clone(),
        )
    }

    pub fn reduce_prover<M>(
        &self,
        srs: &ZkSRS<F,G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat: &M,
        tilde_v: F,
    ) -> (G, F, Vec<F>, Vec<F>)
    where
        M: MatOps<I,F>,
    {
        zk_trans_seq.push_com(self.v_com);

        let log_m = self.xl.len();
        let log_n = self.xr.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        if m & (m - 1) != 0 
        || n & (n - 1) != 0 {
            panic!("Length of the projection vector is not a power of 2");
        } 
        
        let log_m = (m as u64).ilog2() as usize;
        let log_n = (n as u64).ilog2() as usize;
        
        let la = mat.proj_left_challenges(&self.xl);

        let mut vec_u_current = la.clone();
        let mut vec_r_current =
        xi::xi_from_challenges(&self.xr);
        
        let mut challenges_n: Vec<F> = Vec::new();
        let mut challenges_inv_n: Vec<F> = Vec::new();

        let mut lhs_tilde = tilde_v;
        let mut lhs_com = self.v_com.clone();


        for j in 0..log_n {
            let current_len = n / 2usize.pow(j as u32);
            
            let u_left = 
                vec_u_current[0..current_len/2].to_vec();
            let u_right = 
                vec_u_current[current_len/2..current_len].to_vec();
            
            let r_left = 
                vec_r_current[0..current_len/2].to_vec();
            let r_right = 
                vec_r_current[current_len/2..current_len].to_vec();
            
            let l_tr = 
                linear::inner_product(&u_left, &r_right);
            let r_tr = 
                linear::inner_product(&u_right, &r_left);
                

            let (l_com,l_tilde) = zk_trans_seq
            .push_gen_blinding(l_tr);
            let (r_com,r_tilde) = zk_trans_seq
            .push_gen_blinding(r_tr);

            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_n.push(x_j);
            challenges_inv_n.push(x_j_inv);

            lhs_tilde = lhs_tilde +  l_tilde * x_j + r_tilde * x_j_inv;
            lhs_com = lhs_com + l_com.mul(&x_j) + r_com.mul(&x_j_inv);


            vec_u_current = linear::vec_addition(
                &u_left,
                &linear::vec_scalar_mul(
                    &u_right, &x_j_inv),
            );

            vec_r_current = linear::vec_addition(
                &r_left,
                &linear::vec_scalar_mul(
                    &r_right, &x_j),
            );

        }

        let r_reduce = vec_r_current[0];

        let a_xi_inv = mat.proj_right_challenges(&challenges_inv_n);
        
        let mut vec_a_current= a_xi_inv.to_vec();

        let l_vec =
        xi::xi_from_challenges::<F>(&self.xl);

        let mut vec_l_current = l_vec[0..m].to_vec();
        
        let mut challenges_m= Vec::new();
        let mut challenges_inv_m= Vec::new();

        
        for j in 0..log_m {
            let current_len = m / 2usize.pow(j as u32);
            
            let a_left = 
                vec_a_current[0..current_len/2].to_vec();
            let a_right = 
                vec_a_current[current_len/2..current_len].to_vec();
            
            let l_left = 
                vec_l_current[0..current_len/2].to_vec();
            let l_right = 
                vec_l_current[current_len/2..current_len].to_vec();
            
            let l_tr = 
                linear::inner_product(&a_left, &l_right) * r_reduce;
            let r_tr = 
                linear::inner_product(&a_right, &l_left) * r_reduce;
                

            let (l_com,l_tilde) = zk_trans_seq
            .push_gen_blinding(l_tr);
            let (r_com,r_tilde) = zk_trans_seq
            .push_gen_blinding(r_tr);

            let x_j = zk_trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();

            challenges_m.push(x_j);
            challenges_inv_m.push(x_j_inv);

            lhs_tilde = lhs_tilde +  l_tilde * x_j + r_tilde * x_j_inv;
            lhs_com = lhs_com + l_com.mul(&x_j) + r_com.mul(&x_j_inv);

            vec_a_current = linear::vec_addition(
                &a_left,
                &linear::vec_scalar_mul(
                    &a_right, &x_j_inv),
            );

            vec_l_current = linear::vec_addition(
                &l_left,
                &linear::vec_scalar_mul(
                    &l_right, &x_j),
            );

        }

        let a_reduce = vec_a_current[0];
        let l_reduce = vec_l_current[0];



        let (a_reduce_blind, a_reduce_tilde) =
            zk_trans_seq.push_gen_blinding(a_reduce);
        
     
        let hat_b = l_reduce * r_reduce;
        
        let zk_semi_mul = ZkSemiMulScalar::new(
            srs,
            lhs_com,
            a_reduce_blind,
            hat_b,
        );

        zk_semi_mul.prove(
            zk_trans_seq,
            a_reduce,
            lhs_tilde,
            a_reduce_tilde,
        );

        (
            a_reduce_blind,
            a_reduce_tilde,
            challenges_inv_m,
            challenges_inv_n,
        )

        
    }

    pub fn verify_as_subprotocol(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {


        let mut a_reduce_blind = G::zero();
        
        let mut challenges_n: Vec<F> = Vec::new();
        let mut challenges_inv_n: Vec<F> = Vec::new();

        let mut flag = false;

        let pointer_old = trans_seq.pointer;
        
        if (
            TranElem::Group(self.v_com),
        ) != (
            trans_seq.data[pointer_old].clone(),
        ) {
            println!("{:?}", self.v_com);
            println!("{:?}", trans_seq.data[pointer_old]);
            println!("!! Invalid public input when verifying Proj");
        } 


        let log_m = self.xl.len();
        let log_n = self.xr.len();


        trans_seq.pointer = pointer_old + 3 * log_n + 2;

        let mut current_pointer = pointer_old + 1;
        let mut lhs: G = self.v_com;
        

        for _ in 0..log_n {

            if let (
                TranElem::Group(l_tr),
                TranElem::Group(r_tr),
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[current_pointer].clone(),
                trans_seq.data[current_pointer + 1].clone(),
                trans_seq.data[current_pointer + 2].clone(),
            ) {
                let x_j_inv = x_j.inverse().unwrap();
                lhs = lhs + l_tr.mul(x_j) + r_tr.mul(x_j_inv);
                challenges_n.push(x_j);
                challenges_inv_n.push(x_j_inv);

            } else {
                println!("!!!! Invalid transcript when verifying Proj");
            }

            current_pointer += 3;
        }

        let mut challenges_m = Vec::new();
        let mut challenges_inv_m = Vec::new();

        for _ in 0..log_m {
            if let (
                TranElem::Group(l_tr),
                TranElem::Group(r_tr),
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[current_pointer].clone(),
                trans_seq.data[current_pointer + 1].clone(),
                trans_seq.data[current_pointer + 2].clone(),
            ) {
                let x_j_inv = x_j.inverse().unwrap();
                lhs = lhs + l_tr.mul(x_j) + r_tr.mul(x_j_inv);
                challenges_m.push(x_j);
                challenges_inv_m.push(x_j_inv);

            } else {
                println!("!!! Invalid transcript when verifying Proj");
            }

            current_pointer += 3;
        }

        
        let l_reduce =
        xi::xi_ip_from_challenges::<F>(&self.xl, &challenges_m);
        let r_reduce =
        xi::xi_ip_from_challenges::<F>(&self.xr, &challenges_n);

        let hat_b = l_reduce * r_reduce;

        if let (
            TranElem::Group(a_reduce_com),
        ) = (
            trans_seq.data[current_pointer].clone(),
        ) {
            trans_seq.pointer = current_pointer + 1;

            let zk_semi_mul = ZkSemiMulScalar::new(
                srs,
                lhs,
                a_reduce_com,
                hat_b,
            );
    
    
            let check = zk_semi_mul.verify_as_subprotocol(
                trans_seq,
            );
            flag = check;
            a_reduce_blind = a_reduce_com;
        } 
    
        (flag, a_reduce_blind, (challenges_inv_m, challenges_inv_n))
        
    }

    pub fn verify_as_subprotocol_split(
        &mut self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {

        self.verifer_prepare(
            trans_seq,
        );

        for _ in 0..self.verifier_intermediate.log_n {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                self.verifier_j_in_n_iteration(
                    trans_seq,
                    x_j,
                    false,
                );
            } else {
                println!("!! * Invalid transcript when verifying Proj");
            }
        }

        for j in 0..self.verifier_intermediate.log_m {
            if let (
                TranElem::Coin(x_j),
            ) = (
                trans_seq.data[trans_seq.pointer].clone(),
            ) {
                trans_seq.pointer += 1;

                let last_j =
                j == self.verifier_intermediate.log_m - 1;
                
                self.verifier_j_in_m_iteration(
                    trans_seq,
                    x_j,
                    last_j,
                );
                
            } else {
                println!("!! * Invalid transcript when verifying Proj");
            }
        }

        self.verifier_conclude(
            srs,
            trans_seq,
        );

        (
            self.verifier_intermediate.flag,
            self.verifier_intermediate.a_reduce_blind,
            (
                self.verifier_intermediate.challenges_inv_m.clone(),
                self.verifier_intermediate.challenges_inv_n.clone()
            )
        )

    }

    pub fn verifer_prepare(
        &mut self,
        trans_seq: &mut TranSeq<F, G>,
    ) {

        let log_m = self.xl.len();
        let log_n = self.xr.len();

        let m = 1 << log_m;
        let n = 1 << log_n;
        
        self.verifier_intermediate.m = m;
        self.verifier_intermediate.n = n;
        self.verifier_intermediate.log_m = log_m;
        self.verifier_intermediate.log_n = log_n;

        let pointer_old = trans_seq.pointer;
        
        if (
            TranElem::Group(self.v_com),
        ) != (
            trans_seq.data[pointer_old].clone(),
        ) {
            println!("{:?}", self.v_com);
            println!("{:?}", trans_seq.data[pointer_old]);
            println!("!! Invalid public input when verifying Proj");
        } 


        let current_pointer = pointer_old + 1;

        self.verifier_intermediate.lhs_com = self.v_com;

        if let (
            TranElem::Group(l_tr),
            TranElem::Group(r_tr),
        ) = (
            trans_seq.data[current_pointer].clone(),
            trans_seq.data[current_pointer + 1].clone(),
        ) {
            self.verifier_intermediate.previous_l_com = l_tr;
            self.verifier_intermediate.previous_r_com = r_tr;

        } else {
            println!("l_tr {:?}", trans_seq.data[current_pointer]);
            println!("r_tr {:?}", trans_seq.data[current_pointer + 1]);
            println!("!! Invalid transcript when verifying Proj");
        }

        // println!("Preparing");
        trans_seq.pointer = pointer_old + 3;
        
    }

    pub fn verifier_j_in_m_iteration(
        &mut self,
        trans_seq: &mut TranSeq<F, G>,
        x_j: F,
        last_j: bool,
    ) {
        let x_j_inv = x_j.inverse().unwrap();

        self.verifier_intermediate.challenges_m.push(x_j);
        self.verifier_intermediate.challenges_inv_m.push(x_j_inv);

        self.verifier_intermediate.lhs_com =
        self.verifier_intermediate.lhs_com
        + self.verifier_intermediate.previous_l_com.mul(x_j)
        + self.verifier_intermediate.previous_r_com.mul(x_j_inv);

        let current_pointer = trans_seq.pointer;

        if !last_j{
            if let (
                TranElem::Group(l_tr),
                TranElem::Group(r_tr),
            ) = (
                trans_seq.data[current_pointer].clone(),
                trans_seq.data[current_pointer + 1].clone(),
            ) {
                trans_seq.pointer = current_pointer + 2;

                self.verifier_intermediate.previous_l_com = l_tr;
                self.verifier_intermediate.previous_r_com = r_tr;

            } else {
                // println!("l_tr {:?}", trans_seq.data[current_pointer]);
                // println!("r_tr {:?}", trans_seq.data[current_pointer + 1]);
                println!("!! Invalid transcript when verifying Proj");
            }
        }
    }

    pub fn verifier_j_in_n_iteration(
        &mut self,
        trans_seq: &mut TranSeq<F, G>,
        x_j: F,
        last_j: bool,
    ) {
        let x_j_inv = x_j.inverse().unwrap();

        self.verifier_intermediate.challenges_n.push(x_j);
        self.verifier_intermediate.challenges_inv_n.push(x_j_inv);

        self.verifier_intermediate.lhs_com =
        self.verifier_intermediate.lhs_com
        + self.verifier_intermediate.previous_l_com.mul(x_j)
        + self.verifier_intermediate.previous_r_com.mul(x_j_inv);

        let current_pointer = trans_seq.pointer;


        if !last_j{
            if let (
                TranElem::Group(l_tr),
                TranElem::Group(r_tr),
            ) = (
                trans_seq.data[current_pointer].clone(),
                trans_seq.data[current_pointer + 1].clone(),
            ) {
                trans_seq.pointer = current_pointer + 2;

                self.verifier_intermediate.previous_l_com = l_tr;
                self.verifier_intermediate.previous_r_com = r_tr;

            } else {
                println!("!! Invalid transcript when verifying Proj");
            }
        }
    }

    pub fn verifier_conclude(
        &mut self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>
    ) {

        let l_reduce =
        xi::xi_ip_from_challenges(
            &self.xl, &self.verifier_intermediate.challenges_m);
        let r_reduce =
        xi::xi_ip_from_challenges(
            &self.xr, &self.verifier_intermediate.challenges_n);

        let hat_b = l_reduce * r_reduce;
        
        let current_pointer = trans_seq.pointer;
        if let (
            TranElem::Group(a_reduce_com),
        ) = (
            trans_seq.data[current_pointer].clone(),
        ) {
            trans_seq.pointer = current_pointer + 1;

            let zk_semi_mul = ZkSemiMulScalar::new(
                srs,
                self.verifier_intermediate.lhs_com,
                a_reduce_com,
                hat_b,
            );
    
    
            let check = zk_semi_mul.verify_as_subprotocol(
                trans_seq,
            );
            self.verifier_intermediate.flag = check;
            self.verifier_intermediate.a_reduce_blind = a_reduce_com;
        } 
    }

    pub fn verify(
        &self,
        srs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>
    ) -> bool {

        if trans_seq.check_fiat_shamir() == false {
            println!("!! Fiat shamir check failed when verifying Proj");
            return false;
        }

        return self.verify_as_subprotocol(srs, trans_seq).0;
    }

}


#[cfg(test)]
mod tests {
    
    use crate::utils::matdef::DenseMatCM;

    use super::*;

    use crate::MyInt;


    #[test]
    fn test_proj() {

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
        // let l_vec = xi::xi_from_challenges(&xl);
        let r_vec = xi::xi_from_challenges(&xr);

        let v =
        mat.proj_lr(&xl, &xr);

        let la = mat.proj_left_challenges(&xl);
        let v1 = linear::inner_product(&la, &r_vec);
        assert_eq!(v, v1);
        
        let v_tilde =
        <E as Pairing>::ScalarField::rand(rng);

        let v_com = base.mul(v) + blind_base.mul(v_tilde);

        let ip =
        ZkProj::<MyInt, F, G>::new(
            v_com,
            xl.clone(),
            xr.clone(),
        );

        let zk_trans_seq =
        &mut ZkTranSeq::new(zksrs);

        let (
            hat_a_com_p,
            tilde_hat_a,
            _,
            _
        ) = ip.reduce_prover(
            zksrs,
            zk_trans_seq,
            &mat,
            v_tilde,
        );

        let trans_seq =
        &mut zk_trans_seq.publish_trans();

        let (
            flag,
            hat_a_com,
            (xl_prime, xr_prime),
        ) = ip.verify_as_subprotocol(
            zksrs,
            trans_seq,
        );

        assert_eq!(flag, true);

        let hat_a =
        mat.proj_lr(&xl_prime, &xr_prime);
        let hat_a_com_check = 
        base.mul(hat_a) + blind_base.mul(tilde_hat_a);

        assert_eq!(hat_a_com, hat_a_com_check);
        assert_eq!(hat_a_com, hat_a_com_p);

        let mut ip2 =
        ZkProj::<MyInt, F, G>::new(
            v_com,
            xl,
            xr,
        );

        let zk_trans_seq_split =
        &mut ZkTranSeq::new(zksrs);

        let (
            hat_a_com_p,
            tilde_hat_a,
            _,
            _
        ) = ip2.reduce_prover_split(
            zksrs,
            zk_trans_seq_split,
            &mat,
            v_tilde,
        );

        let trans_seq_split =
        &mut zk_trans_seq_split.publish_trans();

        let (
            flag,
            hat_a_com,
            (xl_prime, xr_prime),
        ) = ip2.verify_as_subprotocol_split(
            zksrs,
            trans_seq_split,
        );

        assert_eq!(flag, true);

        let hat_a =
        mat.proj_lr(&xl_prime, &xr_prime);
        let hat_a_com_check = 
        base.mul(hat_a) + blind_base.mul(tilde_hat_a);

        assert_eq!(hat_a_com, hat_a_com_p);
        assert_eq!(hat_a_com, hat_a_com_check);

        assert_eq!(trans_seq.data.len(), trans_seq.pointer);

     
        println!(" * Verification of ZkProj passed");

    }

    
}

