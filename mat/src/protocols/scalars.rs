//! Zero-knowledge protocols for scalar arithmetic
//! 
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;

use crate::data_structures::ZkSRS;
use crate::utils::zktr::{ZkTranSeq, TranSeq, TranElem};


// Prove know a scalar a such that a * base = commit
pub struct ZkSchnorr<F, G>
where 
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub commit: G,
    pub base: G,
}

// Prove holding scalar c, a, b and \tilde{c}, \tilde{a}, \tilde{b} such that:
// c_com = c.toGt() + \tilde{c} * blind_base
// a_com = a.toGt() + \tilde{a} * blind_base
// b_com = b.toGt() + \tilde{b} * blind_base
// c = a * b
// 
pub struct ZkMulScalar<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub c_com: G,
    pub a_com: G,
    pub b_com: G,
    pub com_base: G,
    pub blind_base: G,
}

// Prove holding scalar c, a and \tilde{c}, \tilde{a} such that:
// c_com = c.toGt() + \tilde{c} * blind_base
// a_com = a.toGt() + \tilde{a} * blind_base
// c = a * b
// Here, b is a public scalar
// 
pub struct ZkSemiMulScalar<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub c_com: G,
    pub a_com: G,
    pub b: F,
    pub com_base: G,
    pub blind_base: G,
}

impl<F, G> ZkSchnorr<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    
    pub fn new(
        srs: &ZkSRS<F, G>,
        commit_value: G, 
    ) -> Self {
        Self {
            commit: commit_value,
            base: srs.blind_base,
        }
    }

    pub fn prove(
        &self,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        witness: F
    ) {
        
        zk_trans_seq.trans_seq.push(TranElem::Group(self.commit));
        zk_trans_seq.trans_seq.push(TranElem::Group(self.base));

        let rng = &mut ark_std::rand::thread_rng();
        let r = F::rand(rng);

        let r_com = self.base * r;

        zk_trans_seq.trans_seq.push(TranElem::Group(r_com));

        let challenge = zk_trans_seq.trans_seq.gen_challenge();

        let z = r + challenge * witness;

        zk_trans_seq.trans_seq.push(TranElem::Field(z));
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut TranSeq<F, G>,
    ) -> bool {

        let pointer_old = trans_seq.pointer;

        if (
            TranElem::Group(self.commit),
            TranElem::Group(self.base),
        ) != (
            trans_seq.data[pointer_old].clone(),
            trans_seq.data[pointer_old + 1].clone(),
        ) {
            // println!("schnorr {:?}", trans_seq.data[pointer_old]);
            // println!("schnorr {:?}", trans_seq.data[pointer_old + 1]);
            println!("!! Invalid public input when verifying ZkSchnorr");
            return false;
        } 

        
        if let (
            TranElem::Group(r_com),
            TranElem::Coin(challenge),
            TranElem::Field(z),
        ) = (
            trans_seq.data[pointer_old + 2].clone(),
            trans_seq.data[pointer_old + 3].clone(),
            trans_seq.data[pointer_old + 4].clone(),
        ) {

            trans_seq.pointer = pointer_old + 5;

           if r_com + self.commit.mul(&challenge) == self.base.mul(&z) {
               return true;
           } else {
               println!("!! ZkSchnorr equation check failed when verifying ZkSchnorr");
           } 
        } else {
            println!("!! Type check for transcript elements failed when verifying ZkSchnorr");
        }

        return false;
        
    } 

}


// This is the simplified case of Algorithm 5 in the zkMatrix paper
impl<F, G> ZkMulScalar<F, G>
where 
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    
    pub fn new(
        srs: &ZkSRS<F, G>,
        c_com_value: G, 
        a_com_value: G,
        b_com_value: G,
    ) -> Self {
        Self {
            c_com: c_com_value,
            a_com: a_com_value,
            b_com: b_com_value,
            com_base: srs.com_base,
            blind_base: srs.blind_base,
        }
    }

    pub fn prove(
        &self, 
        zk_trans_seq: &mut ZkTranSeq<F, G>, 
        a: F, b: F,
        c_tilde: F, a_tilde: F, b_tilde: F,
    )
    {
        
        zk_trans_seq.trans_seq.push(TranElem::Group(self.c_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(self.a_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(self.b_com));

        let rng = &mut ark_std::rand::thread_rng();

        let alpha: F = F::rand(rng);
        let beta: F = F::rand(rng);

        let alpha_tilde = F::rand(rng);
        let beta_tilde = F::rand(rng);
        let gamma_tilde = F::rand(rng);
        let delta_tilde = F::rand(rng);

        let blind_base = self.blind_base;
        let com_base = self.com_base;
        let alpha_com 
            = com_base.mul(&alpha) + blind_base.mul(&alpha_tilde);
        let beta_com
            = com_base.mul(&beta) + blind_base.mul(&beta_tilde);
        let gamma_com =
            com_base.mul(alpha * beta) + blind_base.mul(&gamma_tilde);
        let delta_com =
            com_base.mul(alpha * b + a * beta) + blind_base.mul(&delta_tilde);
         
        zk_trans_seq.trans_seq.push(TranElem::Group(alpha_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(beta_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(gamma_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(delta_com));

        let x = zk_trans_seq.gen_challenge();

        let z_a = 
            - alpha_tilde - x * a_tilde;
        let z_b = 
            - beta_tilde - x * b_tilde;
        let z_c = 
            - gamma_tilde - x * delta_tilde - x * x * c_tilde;

        zk_trans_seq.trans_seq.push(TranElem::Field(z_a));
        zk_trans_seq.trans_seq.push(TranElem::Field(z_b));
        zk_trans_seq.trans_seq.push(TranElem::Field(z_c));

        let a_blind: F = 
            (alpha + a * x) * x;
        let b_blind: F = 
            (beta + b * x) * x;
        
        zk_trans_seq.trans_seq.push(TranElem::Field(a_blind));
        zk_trans_seq.trans_seq.push(TranElem::Field(b_blind));

    }


    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut TranSeq<F, G>,
    ) -> bool 
    {

        let pointer_old = trans_seq.pointer;

        if (
            TranElem::Group(self.c_com),
            TranElem::Group(self.a_com),
            TranElem::Group(self.b_com),
        ) != (
            trans_seq.data[pointer_old].clone(),
            trans_seq.data[pointer_old + 1].clone(),
            trans_seq.data[pointer_old + 2].clone(),
        ) {
            println!("!! Invalid public input when verifying ZkMulScalar");
            return false;
        } 

        
        if let (
            TranElem::Group(alpha_com),
            TranElem::Group(beta_com),
            TranElem::Group(gamma_com),
            TranElem::Group(delta_com),
            TranElem::Coin(x),
            TranElem::Field(z_a),
            TranElem::Field(z_b),
            TranElem::Field(z_c),
            TranElem::Field(a_blind),
            TranElem::Field(b_blind),
        ) = (
            trans_seq.data[pointer_old + 3].clone(),
            trans_seq.data[pointer_old + 4].clone(),
            trans_seq.data[pointer_old + 5].clone(),
            trans_seq.data[pointer_old + 6].clone(),
            trans_seq.data[pointer_old + 7].clone(),
            trans_seq.data[pointer_old + 8].clone(),
            trans_seq.data[pointer_old + 9].clone(),
            trans_seq.data[pointer_old + 10].clone(),
            trans_seq.data[pointer_old + 11].clone(),
            trans_seq.data[pointer_old + 12].clone(),
        ) {

            trans_seq.pointer = pointer_old + 13;

            let blind_base = self.blind_base;
            let com_base = self.com_base;

            let p_a = 
                alpha_com.add(self.a_com.mul(&x))
                .add(blind_base.mul(&z_a))
                .mul(&x);
            let p_b = 
                beta_com.add(self.b_com.mul(&x))
                .add(blind_base.mul(&z_b))
                .mul(&x);
            let p_c =
                gamma_com.add(delta_com.mul(&x))
                .add(self.c_com.mul(x * x))
                .add(blind_base.mul(&z_c)).mul(x * x);

            let check1: bool = 
                p_a == com_base.mul(&a_blind);
            let check2: bool =
                p_b == com_base.mul(&b_blind);
            let check3: bool =
                p_c == com_base.mul(a_blind * b_blind);

           if check1 || check2 || check3 {
            //    println!("Check passed");
               return true;
               
           } else {
               println!(
                "!! ZkSchnorr equation check failed when verifying ZkSchnorr"
            );
           } 
        } else {
            println!(
                "!! Type check for transcript elements failed when verifying ZkSchnorr"
            );
        }

        return false;
        
    } 

}

// This is a simplied case of ZkMulScalar
impl <F, G> ZkSemiMulScalar<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(
        srs: &ZkSRS<F, G>,
        c_com_value: G, 
        a_com_value: G,
        b_value: F,
    ) -> Self {
        Self {
            c_com: c_com_value,
            a_com: a_com_value,
            b: b_value,
            com_base: srs.com_base,
            blind_base: srs.blind_base,
        }
    }

    pub fn prove (
        &self, 
        zk_trans_seq: &mut ZkTranSeq<F, G>, 
        a: F, 
        c_tilde: F, a_tilde: F, 
    ) {
        
        zk_trans_seq.trans_seq.push(TranElem::Group(self.c_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(self.a_com));
        zk_trans_seq.trans_seq.push(TranElem::Field(self.b));

        let rng = &mut ark_std::rand::thread_rng();
        
        let alpha = F::rand(rng);
        let gamma = alpha * self.b;

        let alpha_tilde = F::rand(rng);
        let gamma_tilde = F::rand(rng);
     
        let blind_base = self.blind_base;
        let com_base = self.com_base;

        let alpha_com 
            = com_base.mul(&alpha) + blind_base.mul(&alpha_tilde);
        let gamma_com
            = com_base.mul(&gamma) + blind_base.mul(&gamma_tilde);
         
        zk_trans_seq.trans_seq.push(TranElem::Group(alpha_com));
        zk_trans_seq.trans_seq.push(TranElem::Group(gamma_com));
       
        let x = zk_trans_seq.gen_challenge();

        let z_a = 
            - alpha_tilde - x * a_tilde;
        let z_c = 
            - gamma_tilde - x * c_tilde;


        zk_trans_seq.trans_seq.push(TranElem::Field(z_a));
        zk_trans_seq.trans_seq.push(TranElem::Field(z_c));

        let a_blind = 
            alpha + a * x;
        
        zk_trans_seq.trans_seq.push(TranElem::Field(a_blind));
       
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut TranSeq<F, G>,
    ) -> bool 
    {

        let pointer_old = trans_seq.pointer;


        if (
            TranElem::Group(self.c_com),
            TranElem::Group(self.a_com),
            TranElem::Field(self.b),
        ) != (
            trans_seq.data[pointer_old].clone(),
            trans_seq.data[pointer_old + 1].clone(),
            trans_seq.data[pointer_old + 2].clone(),
        ) {
            println!("!! Invalid public input when verifying ZkMulSemiScalar");
            return false;
        } 

        
        if let (
            TranElem::Group(alpha_com),
            TranElem::Group(gamma_com),
            TranElem::Coin(x),
            TranElem::Field(z_a),
            TranElem::Field(z_c),
            TranElem::Field(a_blind),
        ) = (
            trans_seq.data[pointer_old + 3].clone(),
            trans_seq.data[pointer_old + 4].clone(),
            trans_seq.data[pointer_old + 5].clone(),
            trans_seq.data[pointer_old + 6].clone(),
            trans_seq.data[pointer_old + 7].clone(),
            trans_seq.data[pointer_old + 8].clone(),
        ) {

            trans_seq.pointer = pointer_old + 9;
        
            let blind_base = self.blind_base;
            let com_base = self.com_base;
            let p_a = 
                alpha_com + self.a_com.mul(&x) + blind_base.mul(&z_a);
            let p_c = 
                gamma_com + self.c_com.mul(&x) + blind_base.mul(&z_c);
      
            let check1: bool = 
                p_a == com_base.mul(&a_blind);
            let check2: bool =
                p_c == com_base.mul(a_blind * self.b);

           if check1 || check2 {
            //    println!("Check passed");
               return true;
               
           } else {
               println!(
                "!! ZkMulSemiScalar equation check failed"
            );
           } 
        } else {
            println!(
                "!! Type check for transcript elements failed when verifying ZkMulSemiScalar"
            );
        }

        return false;
        
    } 


}


#[cfg(test)]
mod tests {
    
    use super::*;
   
    #[test]
    fn test_zk_scalar() {

        use ark_bls12_381::Fr;
        use ark_bls12_381::G1Projective;
        use ark_std::UniformRand;
        use ark_std::ops::Mul;

        for _ in 0..10 {
        let rng = &mut ark_std::rand::thread_rng();

        let base = G1Projective::rand(rng);
        let blind_base = G1Projective::rand(rng);

        let a = Fr::rand(rng);
        let b = Fr::rand(rng);
        let c = a * b;

        let a_tilde = Fr::rand(rng);
        let b_tilde = Fr::rand(rng);
        let c_tilde = Fr::rand(rng);

        let a_com =
        base.mul(a) + blind_base.mul(a_tilde);
        let b_com =
        base.mul(b) + blind_base.mul(b_tilde);
        let c_com =
        base.mul(c) + blind_base.mul(c_tilde);

        let a_tilde_com = blind_base.mul(a_tilde);


        let srs =
        &ZkSRS::new(base, blind_base);

        let mut zk_trans_seq =
        ZkTranSeq::new(srs);

        let schnorr = ZkSchnorr::new(
            srs, a_tilde_com,
        );

        let mul_scalar = 
        ZkMulScalar::new(
            srs, c_com, a_com, b_com);

        let mul_semi =
        ZkSemiMulScalar::new(
            srs,c_com, a_com, b,
        );


        schnorr.prove(&mut zk_trans_seq, a_tilde);
        mul_scalar.prove(
        &mut zk_trans_seq, a, b, c_tilde, a_tilde, b_tilde
        );
        mul_semi.prove(&mut zk_trans_seq, a, c_tilde, a_tilde);

        let mut trans_seq = zk_trans_seq.publish_trans();

        let check1 = schnorr.verify_as_subprotocol(&mut trans_seq);
        let check2 = 
            mul_scalar.verify_as_subprotocol(
                &mut trans_seq
            );
        let check3 =
            mul_semi.verify_as_subprotocol(
                &mut trans_seq
            );
 
        assert_eq!(check1, true);
        assert_eq!(check2, true);
        assert_eq!(check3, true);
        }
    }

    
}