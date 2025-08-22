//! Define the zk transformation trait.
//! 
//! 
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_serialize::{
    CanonicalSerialize, Compress,
    SerializationError,
};
use ark_std::io::Write;

use crate::data_structures::ZkSRS;
use super::fiat_shamir::FiatShamir;

// Generate random field element
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    Eq(bound = ""),
    PartialEq(bound = ""),
)]
pub enum TranElem<F, G> 
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    Field(F),
    Group(G),
    Size(usize),
    Coin(F),
}

pub struct ZkTranSeq<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub trans_seq: TranSeq<F, G>,
    pub com_base: G,
    pub blind_base: G,
    pub blind_seq: Vec<F>,
}


impl<F, G> ZkTranSeq<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{

    pub fn new(
        srs: &ZkSRS<F, G>,
    ) -> Self {
        let trans_seq = TranSeq::new();
        let blind_seq = Vec::new();
        Self { 
            com_base: srs.com_base,
            blind_base: srs.blind_base,
            trans_seq: trans_seq, 
            blind_seq: blind_seq,
        }
    }

    pub fn gen_challenge(&mut self) -> F {
        self.trans_seq.gen_challenge()
    }

    pub fn push_without_blinding(
        &mut self,
        tr_elem: F,
    ) {
        self.trans_seq.push(TranElem::Field(tr_elem));
    }

    pub fn push_size(
        &mut self,
        tr_elem: usize,
    ) {
        self.trans_seq.push(TranElem::Size(tr_elem));
    }

    pub fn push_com(
        &mut self,
        tr_elem: G,
    ) {
        self.trans_seq.push(TranElem::Group(tr_elem));
    }

    pub fn push_with_blinding  (
        &mut self,
        tr_elem: F,
        blinding_factor: F
    ) -> G {

        let tr_blind = 
            self.com_base.mul(&tr_elem)
            + self.blind_base.mul(&blinding_factor);
    
        self.trans_seq.push(TranElem::Group(tr_blind));
        self.blind_seq.push(blinding_factor);

        tr_blind
    }

    pub fn push_gen_blinding(
        &mut self,
        tr_elem: F
    ) -> (G, F) {

        let rng = &mut ark_std::rand::thread_rng();
        
        let blinding_factor = F::rand(rng);

        let com_val = self.push_with_blinding(
            tr_elem, blinding_factor
        );

        (com_val, blinding_factor)
    }
    
    pub fn publish_trans(&mut self)
    -> TranSeq<F, G> {
        self.trans_seq.clone()
    }

}

// Generate transeq
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
)]
pub struct TranSeq<F, G>
where
    F: PrimeField + CanonicalSerialize,
    G: PrimeGroup<ScalarField = F> + CanonicalSerialize,
{
    pub fs: FiatShamir,
    pub data: Vec<TranElem<F,G>>,
    pub pointer: usize,
}


impl<F, G> TranSeq<F, G>
where
    F: PrimeField + CanonicalSerialize,
    G: PrimeGroup<ScalarField = F> + CanonicalSerialize,
{
 
    pub fn new() -> Self {
        let fs = FiatShamir::new();
    
        Self { 
            fs: fs,
            data: Vec::new(),
            pointer: 0,
        }
    }

    pub fn push(&mut self, elem: TranElem<F, G>) {
        
        match elem {
            TranElem::Field(el) => {
                self.fs.push::<F>(&el);
            },
            TranElem::Group(el) => {
                self.fs.push::<G>(&el);
            },
            TranElem::Size(el) => {
                self.fs.push::<usize>(&el);
            },
            TranElem::Coin(_) => {
                panic!("Coin should not be pushed");
            },
        }

        self.data.push(elem);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn gen_challenge(&mut self)
    -> F {
        let challenge = self.fs.gen_challenge();
        self.data.push(TranElem::Coin(challenge));
        challenge
    }

    pub fn check_fiat_shamir(&self) -> bool {
        let n = self.data.len();
        let mut fs_check = FiatShamir::new();

        
        for i in 0..n {
            let current = self.data[i].clone();
    
            match current{
                TranElem::Field(el) => {
                    fs_check.push::<F>(&el);
                },
                TranElem::Group(el) => {
                    fs_check.push::<G>(&el);
                },
                TranElem::Size(el) => {
                    fs_check.push::<usize>(&el);
                },
    
                TranElem::Coin(coin_value) => {
                    let challenge = fs_check.gen_challenge::<F>();
                    if challenge != coin_value{
                        return false;
                    }
                },
            }
        }
        return true;
    }

    pub fn get_proof_size(&self) -> usize {
        let mut size = 0;
        for el in self.data.iter() {
            size += el.serialized_size(Compress::Yes);
        }
        size
    }
}

impl<F, G> CanonicalSerialize for TranElem<F, G>
where
    F: PrimeField + CanonicalSerialize,
    G: PrimeGroup<ScalarField = F> + CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            TranElem::Field(el) => {
                el.serialize_with_mode(&mut writer, compress)?;
            },
            TranElem::Group(el) => {
                el.serialize_with_mode(&mut writer, compress)?;
            },
            TranElem::Size(el) => {
                el.serialize_with_mode(&mut writer, compress)?;
            },
            TranElem::Coin(el) => {
                el.serialize_with_mode(&mut writer, compress)?;
            },
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            TranElem::Field(el) => {
                el.serialized_size(compress)
            },
            TranElem::Group(el) => {
                el.serialized_size(compress)
            },
            TranElem::Size(el) => {
                el.serialized_size(compress)
            },
            TranElem::Coin(el) => {
                el.serialized_size(compress)
            },
        }
    }
}


impl<F,G> CanonicalSerialize for TranSeq<F, G>
where
    F: PrimeField + CanonicalSerialize,
    G: PrimeGroup<ScalarField = F> + CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.data.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.data.serialized_size(compress)
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_fiat_shamir(){
        use ark_bls12_381::{Fr, G1Projective};
        
        let mut trans =
        TranSeq::<Fr, G1Projective>::new();

        trans.push(TranElem::<Fr, G1Projective>::Field(
            Fr::from(1 as u64))
        );

        trans.gen_challenge();

        trans.push(TranElem::<Fr, G1Projective>::Group(
            G1Projective::generator()
        ) );

        trans.gen_challenge();

        assert_eq!(trans.data.len(), 4);
        assert_eq!(trans.check_fiat_shamir(), true);
        println!("Size: {:?}", trans.get_proof_size());

    }
}