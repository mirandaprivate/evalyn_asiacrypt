//! Define the zk transformation trait.
//! 
//! 
use ark_ec::PrimeGroup;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly_commit::smart_pc::{SmartPC, UniversalParams as PcsPP};
use ark_serialize::{
    CanonicalSerialize, Compress,
    SerializationError,
};
use ark_std::io::Write;

// SRS for the SMART-PC scheme.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct ZkSRS<F, G>
where    
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub com_base: G,
    pub blind_base: G,
}

impl<F, G> ZkSRS<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(
        com_base: G, blind_base: G
    ) -> Self {
        Self { 
            com_base: com_base,
            blind_base: blind_base,
        }
    }

    pub fn commit(
        &self,
        witness: F,
        hiding_factor: F,
    ) -> G {
        self.com_base.mul(witness) + self.blind_base.mul(hiding_factor)
    }
}

pub struct SRS<E: Pairing>
{
    pub zksrs: ZkSRS<E::ScalarField, E::G1>,
    pub pc_pp: PcsPP<E>,
    pub h_hat: E::G2,
}

impl<E: Pairing> SRS<E> {

    pub fn setup(logq: usize) -> Self {
        
        let rng = &mut ark_std::rand::thread_rng();

        let pc_pp =
        SmartPC::<E>::setup(logq, rng).unwrap();
        
        let zksrs =
        ZkSRS::<E::ScalarField, E::G1>::new(
            pc_pp.g_0,
            pc_pp.tilde_g,
        );

        let h_hat = pc_pp.h_hat;

        Self {
            zksrs: zksrs,
            pc_pp: pc_pp,
            h_hat: h_hat,
        }
    }   
}


impl<E> CanonicalSerialize for SRS<E>
where
    E: Pairing,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.zksrs.com_base.serialize_with_mode(&mut writer, compress)?;
        self.zksrs.blind_base.serialize_with_mode(&mut writer, compress)?;
        self.pc_pp.serialize_with_mode(&mut writer, compress)?;
        self.h_hat.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.zksrs.com_base.serialized_size(compress)
        + self.zksrs.blind_base.serialized_size(compress)
        + self.pc_pp.serialized_size(compress)
        + self.h_hat.serialized_size(compress)
    }
}
