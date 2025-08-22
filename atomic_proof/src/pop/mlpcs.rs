//! Multilinear Polynomial Commitment via Groth16
//!
//! Goal: Given a 2^L × 2^R matrix `a` (treated in column-major order as the
//! coefficient vector `vec(a)` of a multilinear polynomial), for challenges
//! (xl ∈ F^L, xr ∈ F^R) we want the evaluation
//!   eval = < vec(a), xi_from_challenges([xr || xl]) >
//! where `xi_from_challenges` follows the expansion rules reused from
//! `arithmetic_expression.rs` (tensor-style expansion in reverse order of the
//! challenges).
//!
//! Idea: Build an R1CS circuit with public inputs [eval, xl..., xr...] enforcing
//!   Σ_i a_i * xi_i  - eval = 0
//! The a_i are embedded as constants. The Groth16 verifying key acts as the
//! commitment to `a`; a Groth16 proof acts as the evaluation opening.
//!
//! Interfaces:
//!  commit(a) -> (ProverKey, Commitment)
//!  open(pk, a, xl, xr) -> (eval, proof)
//!  verify(commitment, eval, xl, xr, proof) -> bool
//!
//! Note: Demonstration implementation; no aggressive optimization or caching.

use ark_ec::pairing::Pairing;
use ark_ff::{PrimeField, Zero, UniformRand};
use ark_groth16::{Groth16, ProvingKey, VerifyingKey, Proof, PreparedVerifyingKey};
use ark_serialize::{CanonicalSerialize,CanonicalDeserialize,Compress};
use ark_snark::CircuitSpecificSetupSNARK;
use ark_std::rand::{RngCore, CryptoRng};
use ark_relations::r1cs::ConstraintSynthesizer;

use super::arithmetic_expression::{ArithmeticExpression as AE, ConstraintSystemBuilder, xi_from_challenges_exprs};

use mat::xi;

use rayon::prelude::*; 

#[derive(Clone, Debug,CanonicalSerialize, CanonicalDeserialize)]
pub struct MLPCSProverKey<E: Pairing> {
    pub pk: ProvingKey<E>,
    pub dims: (usize, usize), // (rows, cols)
    pub l_bits: usize,
    pub r_bits: usize,
}

#[derive(Clone, Debug,CanonicalSerialize, CanonicalDeserialize)]
pub struct MLPCSCommitment<E: Pairing> {
    pub vk: VerifyingKey<E>,
    pub dims: (usize, usize),
    pub l_bits: usize,
    pub r_bits: usize,
}

#[derive(Clone, Debug,CanonicalSerialize, CanonicalDeserialize)]
pub struct MLPCSProof<E: Pairing> {
    pub proof: Proof<E>,
    pub eval: E::ScalarField,
}

impl<E: Pairing> MLPCSCommitment<E> {
    pub fn new() -> Self {
        Self { vk: VerifyingKey::default(), dims: (0, 0), l_bits: 0, r_bits: 0 }
    }

    pub fn get_size(&self) -> usize {
        self.serialized_size(Compress::Yes)
    }

    pub fn default() -> Self {
        Self { vk: VerifyingKey::default(), dims: (0, 0), l_bits: 0, r_bits: 0 }
    }
}

impl<E: Pairing> MLPCSProof<E> {
    pub fn new() -> Self {
        Self { proof: Proof::default(), eval: E::ScalarField::zero() }
    }

    pub fn get_size(&self) -> usize {
        self.serialized_size(Compress::Yes)
    }
}

pub struct MLPCS;

impl MLPCS {
    fn assert_pow2(x: usize, label: &str) -> Result<usize, String> {
    if x == 0 || !x.is_power_of_two() { return Err(format!("{} (= {}) is not a power of two", label, x)); }
        Ok(x.trailing_zeros() as usize)
    }

    // Now interpret 'a' as column-major: a.len() = cols, a[c].len() = rows.
    fn flatten_col_first<F: PrimeField>(a: &Vec<Vec<F>>) -> Result<Vec<F>, String> {
        if a.is_empty() { return Ok(vec![]); }
        let cols = a.len(); let rows = a[0].len();
        for (ci,col) in a.iter().enumerate() {
            if col.len()!=rows { return Err(format!("Inconsistent column length: col {} has {} != {}", ci, col.len(), rows)); }
        }
        let mut v = Vec::with_capacity(rows*cols);
        // Concatenate columns in order
        for c in 0..cols { v.extend_from_slice(&a[c]); }
        Ok(v)
    }


    fn build_circuit<F: PrimeField>(a_flat: &[F], eval: F, xl: &[F], xr: &[F]) -> ConstraintSystemBuilder<F> {
        let l_bits = xl.len();
        let r_bits = xr.len();
     
        // public inputs: [eval, xl..., xr...]
        let mut inputs = Vec::with_capacity(1 + l_bits + r_bits);
        inputs.push(eval); inputs.extend_from_slice(xl); inputs.extend_from_slice(xr);
     
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());
     
        let xl_exprs: Vec<AE<F>> = (0..l_bits).map(|i| AE::x(1+i)).collect();
        let xr_exprs: Vec<AE<F>> = (0..r_bits).map(|j| AE::x(1 + l_bits + j)).collect();
        let mut combined = xr_exprs.clone(); combined.extend(xl_exprs.clone());
     
        let xi_exprs = xi_from_challenges_exprs(&combined);
        assert_eq!(xi_exprs.len(), a_flat.len(), "xi length {} != a_flat {}", xi_exprs.len(), a_flat.len());
     
        let terms: Vec<AE<F>> = a_flat.par_iter().zip(xi_exprs.par_iter())
            .filter_map(|(coeff, xi_e)| if coeff.is_zero() { None } else { Some(AE::mul(AE::Constant(*coeff), xi_e.clone())) })
            .collect();

     
        fn reduce_sum<F: PrimeField>(slice: &[AE<F>]) -> AE<F> {
            match slice.len() {
                0 => AE::Constant(F::zero()),
                1 => slice[0].clone(),
                2 => AE::add(slice[0].clone(), slice[1].clone()),
                _ => {
                    let mid = slice.len()/2;
                    let (left, right) = rayon::join(|| reduce_sum(&slice[..mid]), || reduce_sum(&slice[mid..]));
                    AE::add(left, right)
                }
            }
        }
        let sum_expr = reduce_sum(&terms);
        builder.add_constraint(AE::sub(sum_expr, AE::x(0))); // Σ - eval = 0
        builder
    }

    pub fn commit<E, R>(
        a: &Vec<Vec<E::ScalarField>>,
        rng: &mut R
    ) -> Result<(MLPCSProverKey<E>, MLPCSCommitment<E>), Box<dyn std::error::Error>>
    where
        E: Pairing,
        R: RngCore + CryptoRng,
        ConstraintSystemBuilder<E::ScalarField>: ConstraintSynthesizer<E::ScalarField>,
    {
        let cols = a.len();
        let rows = if cols>0 { a[0].len() } else { 0 };
        for (ci,col) in a.iter().enumerate() { if col.len()!=rows { return Err(format!("column {} length {} != {}", ci, col.len(), rows).into()); } }
        let l_bits = Self::assert_pow2(rows, "rows")?;
        let r_bits = Self::assert_pow2(cols, "cols")?;
        let a_flat = Self::flatten_col_first(a)?; // column-major flatten (columns concatenated)
     
        // Dummy circuit: use random non-zero challenges and compute matching eval so constraint holds
        let mut xl_dummy = Vec::with_capacity(l_bits);
        for _ in 0..l_bits { let mut v = E::ScalarField::rand(rng); if v.is_zero(){ v = E::ScalarField::from(1u64);} xl_dummy.push(v); }
        let mut xr_dummy = Vec::with_capacity(r_bits);
        for _ in 0..r_bits { let mut v = E::ScalarField::rand(rng); if v.is_zero(){ v = E::ScalarField::from(1u64);} xr_dummy.push(v); }
        let challenges_combined: Vec<E::ScalarField> = [xr_dummy.as_slice(), xl_dummy.as_slice()].concat();
        let xi_dummy = xi::xi_from_challenges(&challenges_combined);
        assert_eq!(xi_dummy.len(), a_flat.len(), "dummy xi length mismatch");
        let mut eval_dummy = E::ScalarField::zero();
        for (c,x) in a_flat.iter().zip(xi_dummy.iter()) { eval_dummy += *c * *x; }

        let circuit = Self::build_circuit(&a_flat, eval_dummy, &xl_dummy, &xr_dummy);
        let (pk, vk) = Groth16::<E>::setup(circuit, rng)?;

        Ok((MLPCSProverKey { pk, dims: (rows, cols), l_bits, r_bits }, MLPCSCommitment { vk, dims: (rows, cols), l_bits, r_bits }))
    }

    pub fn open<E, R>(
        pk: &MLPCSProverKey<E>,
        a: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        rng: &mut R
    ) -> Result<MLPCSProof<E>, Box<dyn std::error::Error>>
    where
        E: Pairing,
        R: RngCore + CryptoRng,
        ConstraintSystemBuilder<E::ScalarField>: ConstraintSynthesizer<E::ScalarField>,
    {
        // Accept &Vec per new signature; treat as slices internally
        let xl_slice: &[E::ScalarField] = xl.as_slice();
        let xr_slice: &[E::ScalarField] = xr.as_slice();
        assert_eq!(xl_slice.len(), pk.l_bits, "xl length mismatch");
        assert_eq!(xr_slice.len(), pk.r_bits, "xr length mismatch");
        assert_eq!(a.len(), pk.dims.1, "matrix column count mismatch");
        if pk.dims.1>0 { assert_eq!(a[0].len(), pk.dims.0, "matrix row count mismatch"); }
        let a_flat = Self::flatten_col_first(a)?; // column-major flatten
        let challenges_combined: Vec<E::ScalarField> = [xr_slice, xl_slice].concat();
        let xi = xi::xi_from_challenges(&challenges_combined);

        let mut eval = E::ScalarField::zero();
        for (c,x) in a_flat.iter().zip(xi.iter()) { eval += *c * *x; }

        let circuit = Self::build_circuit(&a_flat, eval, xl_slice, xr_slice);
        let proof = Groth16::<E>::create_random_proof_with_reduction(circuit, &pk.pk, rng)?;
        Ok(MLPCSProof { proof, eval })
    }

    pub fn verify<E>(
        comm: &MLPCSCommitment<E>,
        eval: E::ScalarField,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &MLPCSProof<E>
    ) -> Result<bool, Box<dyn std::error::Error>>
    where
        E: Pairing,
    {
        assert_eq!(proof.eval, eval, "eval mismatch");
        let xl_slice: &[E::ScalarField] = xl.as_slice();
        let xr_slice: &[E::ScalarField] = xr.as_slice();
        assert_eq!(xl_slice.len(), comm.l_bits, "xl length mismatch");
        assert_eq!(xr_slice.len(), comm.r_bits, "xr length mismatch");

        let prepared = PreparedVerifyingKey::from(comm.vk.clone());
        let mut public_inputs = Vec::with_capacity(1 + comm.l_bits + comm.r_bits);
       
        public_inputs.push(eval);
        public_inputs.extend_from_slice(xl_slice);
        public_inputs.extend_from_slice(xr_slice);
        
        let ok = Groth16::<E>::verify_proof(&prepared, &proof.proof, &public_inputs)?;
       
        Ok(ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*; use ark_bls12_381::{Bls12_381, Fr}; use ark_std::rand::{SeedableRng, rngs::StdRng, Rng};
    fn rng() -> StdRng { StdRng::seed_from_u64(99) }

    #[test]
    fn test_mlpcs_basic() {
        let mut rng = rng();
        let l_bits=2usize; let r_bits=1usize; let rows=1<<l_bits; let cols=1<<r_bits; // 4x2
        // column-major: a.len() = cols, a[c].len() = rows
        let mut a = vec![vec![Fr::zero(); rows]; cols];
        for c in 0..cols { for r in 0..rows { a[c][r] = Fr::from((r*10 + c + 7) as u64); }}
        
        let (pk, comm) = MLPCS::commit::<Bls12_381,_>(&a, &mut rng).expect("commit failed");
        
        let xl: Vec<Fr> = (0..l_bits).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let xr: Vec<Fr> = (0..r_bits).map(|_| Fr::from(rng.gen::<u64>())).collect();
    
        let proof = MLPCS::open::<Bls12_381,_>(&pk, &a, &xl, &xr, &mut rng).expect("open failed");
        let ok = MLPCS::verify::<Bls12_381>(&comm, proof.eval, &xl, &xr, &proof).expect("verify failed");
    
        assert!(ok, "verification failed");
    }

    #[test]
    fn test_mlpcs_1024_perf() {
        let mut rng = rng();
        let l_bits = 10usize; // rows = 1024
        let r_bits = 2usize; // cols = 1024
        let rows = 1<<l_bits; let cols = 1<<r_bits;
        
        println!("[mlpcs perf] generating matrix: {} x {} (total elements = {})", rows, cols, rows*cols);
        let t_gen = std::time::Instant::now();
        
        // column-major matrix a[cols][rows]
        let mut a = vec![vec![Fr::zero(); rows]; cols];
        for c in 0..cols { for r in 0..rows { let mut v = Fr::from(rng.gen::<u64>()); if v.is_zero() { v = Fr::from(1u64); } a[c][r] = v; }}
        println!("[mlpcs perf] fill time {:.3}s", t_gen.elapsed().as_secs_f64());

        let t_commit = std::time::Instant::now();
        let (pk, comm) = MLPCS::commit::<Bls12_381,_>(&a, &mut rng).expect("commit failed");
        println!("[mlpcs perf] commit(setup) time {:.3}s", t_commit.elapsed().as_secs_f64());
        println!("[mlpcs perf] l_bits={}, r_bits={}", pk.l_bits, pk.r_bits);
        println!("[mlpcs perf] commitment size = {} bytes", comm.get_size());

        let xl: Vec<Fr> = (0..l_bits).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let xr: Vec<Fr> = (0..r_bits).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let t_open = std::time::Instant::now();
        let proof = MLPCS::open::<Bls12_381,_>(&pk, &a, &xl, &xr, &mut rng).expect("open failed");
        println!("[mlpcs perf] open proof time {:.3}s", t_open.elapsed().as_secs_f64());
        println!("[mlpcs perf] proof size = {} bytes", proof.get_size());

        let t_verify = std::time::Instant::now();
        let ok = MLPCS::verify::<Bls12_381>(&comm, proof.eval, &xl, &xr, &proof).expect("verify failed");
        println!("[mlpcs perf] verify time {:.3}s result={}", t_verify.elapsed().as_secs_f64(), ok);
        assert!(ok);
    }
}