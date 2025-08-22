// Implement the matmul protocol
//
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::zktr::{ZkTranSeq, TranSeq, TranElem};
use crate::utils::matdef::{
    ShortInt,
    MatOps,
};

use super::scalars::ZkMulScalar;




pub struct ZkKronecker<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<I, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub shape_a: (usize, usize),
    _marker: PhantomData<(I, MA, MB)>,
}

impl<I, F, G, MA, MB> ZkKronecker<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<I, F>,
{
    pub fn new(
        v_c_com_value: G,
        point_value: (Vec<F>, Vec<F>),
        shape_a_value: (usize, usize),
    ) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            shape_a: shape_a_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &MA,
        mat_b: &MB,
        tilde_c: F,
    ) -> (G, G, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {

        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let ma = self.shape_a.0;
        let na = self.shape_a.1;

        let mb = m / ma;
        let nb = n / na;

        
        if (ma, na) != mat_a.get_shape()
        || (mb, nb) != mat_b.get_shape() {
            panic!("Matrix dimension mismatch");
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let log_ma = ma.ilog2() as usize;
        let log_na = na.ilog2() as usize;

        let xl_a = xl[..log_ma].to_vec();
        let xl_b = xl[log_ma..].to_vec();

        let xr_a = xr[..log_na].to_vec();
        let xr_b = xr[log_na..].to_vec();

        let hat_a = mat_a.proj_lr(&xl_a, &xr_a);
        let hat_b = mat_b.proj_lr(&xl_b, &xr_b);

        let (a_hat_com, a_hat_tilde) =
        zk_trans_seq.push_gen_blinding(hat_a);
        let (b_hat_com, b_hat_tilde) =
        zk_trans_seq.push_gen_blinding(hat_b);


        let protocol_mul =
        ZkMulScalar::new(
            zksrs,
            self.v_c_com,
            a_hat_com,
            b_hat_com,
        );

        protocol_mul.prove(
            zk_trans_seq,
            hat_a,
            hat_b,
            tilde_c,
            a_hat_tilde,
            b_hat_tilde,
        );

        (
            a_hat_com,
            b_hat_com,
            a_hat_tilde,
            b_hat_tilde,
            (xl_a, xr_a),
            (xl_b, xr_b),
        )

    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) 
    {

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let ma = self.shape_a.0;
        let na = self.shape_a.1;

        let log_ma = ma.ilog2() as usize;
        let log_na = na.ilog2() as usize;

        let xl_a = xl[..log_ma].to_vec();
        let xl_b = xl[log_ma..].to_vec();
        let xr_a = xr[..log_na].to_vec();
        let xr_b = xr[log_na..].to_vec();

        let pointer_old = trans_seq.pointer;

        let a_hat_com: G;
        let b_hat_com: G;


        if let (
            TranElem::Group(a_hat_com_value),
            TranElem::Group(b_hat_com_value),
        ) = (
            trans_seq.data[pointer_old].clone(),
            trans_seq.data[pointer_old + 1].clone(),
        ) {
            trans_seq.pointer += 2;
            a_hat_com = a_hat_com_value;
            b_hat_com = b_hat_com_value;
        } else {
            panic!("Unexpected type in Verifying Kronecker");
        }


        let protocol_mul =
        ZkMulScalar::new(
            zksrs,
            self.v_c_com,
            a_hat_com,
            b_hat_com,
        );

        let flag =
        protocol_mul.verify_as_subprotocol(trans_seq);
    
        (
            flag,
            a_hat_com,
            b_hat_com,
            (xl_a, xr_a),
            (xl_b, xr_b),
        )
    }

}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::matdef::{DenseMatCM, SparseKronecker};

    #[test]
    fn test_kronecker() {

        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;


        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;
        

        let rng = &mut ark_std::rand::thread_rng();

        let log_ma = 3;
        let log_na = 4;
        let log_mb = 2;
        let log_nb = 5;

        let ma = 1 << log_ma;
        let na = 1 << log_na;
        let mb = 1 << log_mb;
        let nb = 1 << log_nb;

        let mut mat_a =
        DenseMatCM::<i32,Fr>::new(ma, na);
        let mut mat_b =
        DenseMatCM::<i32,Fr>::new(mb, nb);

        mat_a.gen_rand(8);
        mat_b.gen_rand(8);

        let mut mat_c =
        SparseKronecker::<i32,Fr>::new(
            (ma, na), (mb, nb)
        );
        mat_c.set_data(
            (mat_a.data.clone(),
            mat_b.data.clone(),)
        );


        let log_m = log_ma + log_mb;
        let log_n = log_na + log_nb;

        let xl =
        vec![Fr::rand(rng); log_m];
        let xr =
        vec![Fr::rand(rng); log_n];


        let v_c =
        mat_c.proj_lr(&xl, &xr);

        let v_c_tilde = Fr::rand(rng);

        let zksrs =
        ZkSRS::<F, G>::new(
            G::rand(rng),
            G::rand(rng),
        );

        let v_c_com =
        zksrs.commit(v_c, v_c_tilde);



        let protocol =
        ZkKronecker::<i32, F, G, DenseMatCM<i32, F>, DenseMatCM<i32,F>>::new(
            v_c_com,
            (xl, xr),
            (ma, na),
        );

        let zk_trans_seq =
        &mut ZkTranSeq::new(&zksrs);

        let (
            _,
            _,
            hat_a_tilde,
            hat_b_tilde,
            _,
            _,
        ) = protocol.reduce_prover(
            &zksrs,
            zk_trans_seq,
            &mat_a,
            &mat_b,
            v_c_tilde,
        );

        let trans_seq =
        &mut zk_trans_seq.publish_trans();

        let (
            flag,
            a_hat_com,
            b_hat_com,
            (xl_a, xr_a),
            (xl_b, xr_b),
        ) = protocol.verify_as_subprotocol(
            &zksrs,
            trans_seq,
        );

        let hat_a_check =
        mat_a.proj_lr(&xl_a, &xr_a);

        let hat_b_check =
        mat_b.proj_lr(&xl_b, &xr_b);

        let a_hat_com_check =
        zksrs.commit(hat_a_check, hat_a_tilde);
        let b_hat_com_check =
        zksrs.commit(hat_b_check, hat_b_tilde);

        assert_eq!(a_hat_com, a_hat_com_check);
        assert_eq!(b_hat_com, b_hat_com_check);
        assert_eq!(flag, true);


    }
}

