//! Groth16 proof system implementation for arithmetic expressions

use ark_ff::PrimeField;
use ark_groth16::{Groth16, Proof, ProvingKey, VerifyingKey, PreparedVerifyingKey};
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_std::rand::{RngCore, CryptoRng};
use ark_snark::CircuitSpecificSetupSNARK;
use ark_ec::pairing::Pairing;

use super::arithmetic_expression::{ConstraintSystemBuilder};

// Groth16 proof generator
pub struct Groth16Prover;

impl Groth16Prover {
    // Setup Groth16 proof system
    pub fn setup<F, E, R>(
        constraint_system: &ConstraintSystemBuilder<F>,
        rng: &mut R,
    ) -> Result<(ProvingKey<E>, VerifyingKey<E>), Box<dyn std::error::Error>>
    where
        F: PrimeField + Copy,
        E: Pairing<ScalarField = F>,
        R: RngCore + CryptoRng,
        ConstraintSystemBuilder<F>: ConstraintSynthesizer<F>,
    {
        let circuit = constraint_system.clone();
        
        let (pk, vk) = Groth16::<E>::setup(circuit, rng)?;
        
        Ok((pk, vk))
    }

    // Produce the proof
    pub fn prove<F, E, R>(
        proving_key: &ProvingKey<E>,
        constraint_system: ConstraintSystemBuilder<F>,
        rng: &mut R,
    ) -> Result<Proof<E>, Box<dyn std::error::Error>>
    where
        F: PrimeField + Copy,
        E: Pairing<ScalarField = F>,
        R: RngCore + CryptoRng,
        ConstraintSystemBuilder<F>: ConstraintSynthesizer<F>,
    {
        let proof = Groth16::<E>::create_random_proof_with_reduction(
            constraint_system, 
            proving_key, 
            rng
        )?;
        
        Ok(proof)
    }

    // Produce a proof supplying (or overriding) public & private inputs just-in-time.
    // This is a convenience wrapper so callers don't have to pre-populate the builder.
    pub fn prove_with_pub_pri<F, E, R>(
        proving_key: &ProvingKey<E>,
        mut builder: ConstraintSystemBuilder<F>,
        pub_inputs: Vec<F>,
        pri_inputs: Vec<F>,
        rng: &mut R,
    ) -> Result<Proof<E>, Box<dyn std::error::Error>>
    where
        F: PrimeField + Copy,
        E: Pairing<ScalarField = F>,
        R: RngCore + CryptoRng,
        ConstraintSystemBuilder<F>: ConstraintSynthesizer<F>,
    {
        builder.set_public_inputs(pub_inputs)
               .set_private_inputs(pri_inputs);
        Self::prove::<F, E, R>(proving_key, builder, rng)
    }

    // Verify the proof with prepared verifying key
    pub fn verify<F, E>(
        prepared_vk: &PreparedVerifyingKey<E>,
        public_inputs: &[F],
        proof: &Proof<E>,
    ) -> Result<bool, Box<dyn std::error::Error>>
    where
        F: PrimeField + Copy,
        E: Pairing<ScalarField = F>,
    {
        let public_inputs_converted: Vec<F> = public_inputs
            .iter()
            .map(|&input| input)
            .collect();

        // Verify with prepared verifying key
        let is_valid = Groth16::<E>::verify_proof(
            prepared_vk,
            proof,
            &public_inputs_converted,
        )?;

        Ok(is_valid)
    }

    // Prepare the verifying key
    pub fn prepare_verifying_key<E>(
        vk: &VerifyingKey<E>
    ) -> PreparedVerifyingKey<E> 
    where
        E: Pairing,
    {
        PreparedVerifyingKey::from(vk.clone())
    }
}

// Alias for Curves BLS12-381
pub mod curves {
    use super::*;

    // BLS12-381 Curve Groth16 proof generator
    pub type Bls12_381Prover = Groth16Prover;
    
    // BLS12-381 alias
    pub mod bls12_381 {
        use ark_bls12_381::{Bls12_381, Fr};
        use super::super::*;
        
        pub type Proof = ark_groth16::Proof<Bls12_381>;
        pub type ProvingKey = ark_groth16::ProvingKey<Bls12_381>;
        pub type VerifyingKey = ark_groth16::VerifyingKey<Bls12_381>;
        pub type PreparedVerifyingKey = ark_groth16::PreparedVerifyingKey<Bls12_381>;
        pub type Field = Fr;
        
        impl Groth16Prover {
            // BLS12-381 setup
            pub fn setup_bls12_381<R>(
                constraint_system: &ConstraintSystemBuilder<Fr>,
                rng: &mut R,
            ) -> Result<(ProvingKey, VerifyingKey), Box<dyn std::error::Error>>
            where
                R: RngCore + CryptoRng,
                ConstraintSystemBuilder<Fr>: ConstraintSynthesizer<Fr>,
            {
                Self::setup::<Fr, Bls12_381, R>(constraint_system, rng)
            }
            
            // BLS12-381 prove
            pub fn prove_bls12_381<R>(
                proving_key: &ProvingKey,
                constraint_system: ConstraintSystemBuilder<Fr>,
                rng: &mut R,
            ) -> Result<Proof, Box<dyn std::error::Error>>
            where
                R: RngCore + CryptoRng,
                ConstraintSystemBuilder<Fr>: ConstraintSynthesizer<Fr>,
            {
                Self::prove::<Fr, Bls12_381, R>(proving_key, constraint_system, rng)
            }

            // BLS12-381 convenience: prove while directly supplying pub & pri inputs
            pub fn prove_with_pub_pri_bls12_381<R>(
                proving_key: &ProvingKey,
                builder: ConstraintSystemBuilder<Fr>,
                pub_inputs: Vec<Fr>,
                pri_inputs: Vec<Fr>,
                rng: &mut R,
            ) -> Result<Proof, Box<dyn std::error::Error>>
            where
                R: RngCore + CryptoRng,
                ConstraintSystemBuilder<Fr>: ConstraintSynthesizer<Fr>,
            {
                Groth16Prover::prove_with_pub_pri::<Fr, Bls12_381, R>(
                    proving_key,
                    builder,
                    pub_inputs,
                    pri_inputs,
                    rng,
                )
            }
            
            // BLS12-381 verify
            pub fn verify_bls12_381(
                prepared_verifying_key: &PreparedVerifyingKey,
                public_inputs: &[Fr],
                proof: &Proof,
            ) -> Result<bool, Box<dyn std::error::Error>> {
                Self::verify::<Fr, Bls12_381>(prepared_verifying_key, public_inputs, proof)
            }

            // Prepare the verifying key
            pub fn prepare_verifying_key_bls12_381(
                vk: &VerifyingKey
            ) -> PreparedVerifyingKey {
                PreparedVerifyingKey::from(vk.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::arithmetic_expression::ArithmeticExpression;
    use ark_bls12_381::{Bls12_381, Fr as BlsFr};
    use ark_std::rand::rngs::StdRng;
    use ark_std::rand::SeedableRng;
    use ark_ff::Field; // Add this import for inverse() method

    // A compatible random generator
    fn create_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_groth16_basic_proof_generic() {
        // test the generic version
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );

        assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        // using the generic version
        let (pk, vk) = Groth16Prover::setup::<BlsFr, Bls12_381, _>(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove::<BlsFr, Bls12_381, _>(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key(&vk);
        let is_valid = Groth16Prover::verify::<BlsFr, Bls12_381>(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Proof should be valid");
        
        println!("Generic Groth16 proof generation and verification successful!");
    }

    #[test]
    fn test_groth16_basic_proof_bls12_381() {
        // 测试 BLS12-381 专用版本
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );

    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Proof should be valid");
        
        println!("BLS12-381 specific Groth16 proof generation and verification successful!");
    }

    #[test]
    fn test_groth16_with_private_inputs() {
        // Relation: (p0 + w0) * w1 = p1
        // Choose p0=2, w0=5, w1=3 => (2+5)*3 = 21 => set p1=21
        use ark_bls12_381::Fr as Fp;
        let pub_inputs = vec![Fp::from(2u64), Fp::from(21u64)]; // p0, p1
        let pri_inputs = vec![Fp::from(5u64), Fp::from(3u64)];  // w0, w1

        // Build constraint: (p0 + w0) * w1 - p1 = 0
        use super::super::arithmetic_expression::ArithmeticExpression as AE;
        let mut builder = ConstraintSystemBuilder::new();

        builder.set_public_inputs(pub_inputs.clone())
            .set_private_inputs(pri_inputs.clone());
        let expr = (AE::pub_input(0) + AE::pri_input(0)) * AE::pri_input(1) - AE::pub_input(1);
        builder.add_constraint(expr);

        // Sanity local evaluation
     let lhs = (pub_inputs[0] + pri_inputs[0]) * pri_inputs[1];
        assert_eq!(lhs, pub_inputs[1]);

     assert!(builder.clone().validate_constraints().is_ok());

        let mut rng = create_rng();
        let (pk, vk) = Groth16Prover::setup::<Fp, Bls12_381, _>(&builder, &mut rng)
            .expect("setup failure");

        let proof = Groth16Prover::prove_with_pub_pri::<Fp, Bls12_381, _>(&pk, builder, pub_inputs.clone(), pri_inputs.clone(), &mut rng)
            .expect("prove_with_pub_pri failed");

        let prepared_vk = Groth16Prover::prepare_verifying_key(&vk);
        let is_valid = Groth16Prover::verify::<Fp, Bls12_381>(&prepared_vk, &pub_inputs, &proof)
            .expect("verify failed");
        assert!(is_valid, "Proof with private inputs should verify");
    }

    #[test]
    fn test_groth16_with_prepared_vk() {
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup::<BlsFr, Bls12_381, _>(&builder, &mut rng)
            .expect("Failed to generate keys");

        let prepared_vk = Groth16Prover::prepare_verifying_key(&vk);

        let proof = Groth16Prover::prove::<BlsFr, Bls12_381, _>(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let is_valid = Groth16Prover::verify::<BlsFr, Bls12_381>(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Proof should be valid with prepared VK");
        
        println!("Groth16 proof with prepared VK successful!");
    }

    #[test]
    fn test_groth16_complex_proof() {
        let inputs = vec![
            BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(4u64)
        ];
        
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(5u64)
        );
        
        builder.add_constraint(
            ArithmeticExpression::x(0) * ArithmeticExpression::x(1) - BlsFr::from(6u64)
        );

        builder.set_public_inputs(inputs.clone());
        assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup::<BlsFr, Bls12_381, _>(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove::<BlsFr, Bls12_381, _>(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key(&vk);
        let is_valid = Groth16Prover::verify::<BlsFr, Bls12_381>(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Complex proof should be valid");
        
        println!("Complex Groth16 proof generation and verification successful!");
    }

    #[test]
    fn test_groth16_invalid_proof() {
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup::<BlsFr, Bls12_381, _>(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove::<BlsFr, Bls12_381, _>(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let wrong_inputs = vec![BlsFr::from(5u64), BlsFr::from(6u64)];
        let prepared_vk = Groth16Prover::prepare_verifying_key(&vk);
        let is_valid = Groth16Prover::verify::<BlsFr, Bls12_381>(&prepared_vk, &wrong_inputs, &proof)
            .expect("Failed to verify proof");

        assert!(!is_valid, "Verification with wrong inputs should fail");
        
        println!("Invalid proof test passed - verification correctly failed with wrong inputs");
    }

    #[test]
    fn test_groth16_bls12_381_simple_addition() {
        let inputs = vec![BlsFr::from(5u64), BlsFr::from(7u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // x0 + x1 = 12
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(12u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof using prove_bls12_381");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Simple addition proof should be valid");
        println!("prove_bls12_381 simple addition test passed!");
    }

    #[test]
    fn test_groth16_bls12_381_multiplication() {
        let inputs = vec![BlsFr::from(6u64), BlsFr::from(8u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // x0 * x1 = 48
        builder.add_constraint(
            ArithmeticExpression::x(0) * ArithmeticExpression::x(1) - BlsFr::from(48u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof using prove_bls12_381");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Multiplication proof should be valid");
        println!("prove_bls12_381 multiplication test passed!");
    }

    #[test]
    fn test_groth16_bls12_381_multiple_constraints() {
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64), BlsFr::from(5u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // x0 + x1 = 7
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );
        
        // x0 * x2 = 15
        builder.add_constraint(
            ArithmeticExpression::x(0) * ArithmeticExpression::x(2) - BlsFr::from(15u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Multiple constraints proof should be valid");
        println!("prove_bls12_381 multiple constraints test passed!");
    }

    #[test]
    fn test_groth16_bls12_381_quadratic_constraint() {
        let inputs = vec![BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // x0^2 = 16 (x0 * x0 = 16)
        builder.add_constraint(
            ArithmeticExpression::x(0) * ArithmeticExpression::x(0) - BlsFr::from(16u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof using prove_bls12_381");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Quadratic constraint proof should be valid");
        println!("prove_bls12_381 quadratic constraint test passed!");
    }

    #[test]
    fn test_groth16_bls12_381_with_constants() {
        let inputs = vec![BlsFr::from(10u64), BlsFr::from(20u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // 2 * x0 + 3 * x1 = 80
        builder.add_constraint(
            ArithmeticExpression::x(0) * BlsFr::from(2u64) + 
            ArithmeticExpression::x(1) * BlsFr::from(3u64) - 
            BlsFr::from(80u64)
        );

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof using prove_bls12_381");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Constants constraint proof should be valid");
        println!("prove_bls12_381 with constants test passed!");
    }

    #[test]
    fn test_groth16_bls12_381_different_seeds() {
        let inputs = vec![BlsFr::from(1u64), BlsFr::from(2u64)];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(3u64)
        );

        // Test with different random seeds
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(67890);

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng1)
            .expect("Failed to generate keys");

        let proof1 = Groth16Prover::prove_bls12_381(&pk, builder.clone(), &mut rng1)
            .expect("Failed to generate first proof");

        let proof2 = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng2)
            .expect("Failed to generate second proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);

        let is_valid1 = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof1)
            .expect("Failed to verify first proof");
        let is_valid2 = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof2)
            .expect("Failed to verify second proof");

        assert!(is_valid1, "First proof should be valid");
        assert!(is_valid2, "Second proof should be valid");
        
        // Proofs should be different (due to randomness)
        assert_ne!(proof1.a, proof2.a, "Proofs should be different due to randomness");
        
        println!("prove_bls12_381 different seeds test passed!");
    }

    #[test]
    fn test_groth16_equal_vec_constraint() {
        // test equal_vec: [a, b] == [c, d] => a == c && b == d
        let inputs = vec![
            BlsFr::from(5u64),  // x0 = 5
            BlsFr::from(7u64),  // x1 = 7
            BlsFr::from(5u64),  // x2 = 5 (should equal x0)
            BlsFr::from(7u64),  // x3 = 7 (should equal x1)
        ];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // Create two vectors: [x0, x1] and [x2, x3]
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        
        // Add equal_vec constraint using the correct method name
        let result = builder.add_equal_vec_constraints(vec1, vec2);
        assert!(result.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "equal_vec constraint proof should be valid");
        println!("equal_vec constraint test passed!");
    }

    #[test]
    fn test_groth16_equal_vec_constraint_invalid() {
        // test equal_vec failure scenario: [a, b] != [c, d]
        let inputs = vec![
            BlsFr::from(5u64),  // x0 = 5
            BlsFr::from(7u64),  // x1 = 7
            BlsFr::from(5u64),  // x2 = 5 (equals x0)
            BlsFr::from(8u64),  // x3 = 8 (NOT equal to x1)
        ];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        
        // Add equal_vec constraint using the correct method name
        let result = builder.add_equal_vec_constraints(vec1, vec2);
        assert!(result.is_ok());

        // This should fail validation because constraints are not satisfied
    builder.set_public_inputs(inputs.clone());
    let validation_result = builder.validate_constraints();
        assert!(validation_result.is_err(), "Validation should fail with invalid equal_vec constraint");
        
        println!("equal_vec invalid constraint test passed - correctly failed validation!");
    }

    #[test]
    fn test_groth16_mul_vec_constraint() {
        // test mul_vec: [a, b] * [c, d] = [e, f] => a*c = e && b*d = f
        let inputs = vec![
            BlsFr::from(3u64),  // x0 = 3
            BlsFr::from(4u64),  // x1 = 4
            BlsFr::from(5u64),  // x2 = 5
            BlsFr::from(6u64),  // x3 = 6
            BlsFr::from(15u64), // x4 = 15 (3 * 5)
            BlsFr::from(24u64), // x5 = 24 (4 * 6)
        ];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // Create three vectors: [x0, x1], [x2, x3], [x4, x5]
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        let result_vec = vec![ArithmeticExpression::x(4), ArithmeticExpression::x(5)];
        
        // Add mul_vec constraint using the correct method name
        let result = builder.add_mul_vec_constraint(vec1, vec2, result_vec);
        assert!(result.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "mul_vec constraint proof should be valid");
        println!("mul_vec constraint test passed!");
    }

    #[test]
    fn test_groth16_mul_vec_constraint_with_constants() {
        // test mul_vec contains constants
        let inputs = vec![
            BlsFr::from(2u64),  // x0 = 2
            BlsFr::from(3u64),  // x1 = 3
            BlsFr::from(6u64),  // x2 = 6 (2 * 3)
            BlsFr::from(12u64), // x3 = 12 (4 * 3)
        ];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // [x0, 4] * [x1, x1] = [x2, x3]
        let vec1 = vec![
            ArithmeticExpression::x(0), 
            ArithmeticExpression::Constant(BlsFr::from(4u64))
        ];
        let vec2 = vec![ArithmeticExpression::x(1), ArithmeticExpression::x(1)];
        let result_vec = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        
        let result = builder.add_mul_vec_constraint(vec1, vec2, result_vec);
        assert!(result.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "mul_vec with constants constraint proof should be valid");
        println!("mul_vec with constants constraint test passed!");
    }

    #[test]
    fn test_groth16_inv_vec_constraint() {
        // test inv_vec: for [a, b]，with inverse [1/a, 1/b]
        let a = BlsFr::from(4u64);
        let b = BlsFr::from(5u64);
        let inv_a = a.inverse().unwrap(); // 1/4
        let inv_b = b.inverse().unwrap(); // 1/5
        
        let inputs = vec![a, b, inv_a, inv_b];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // Create vectors: [x0, x1] and [x2, x3] where x2 = 1/x0, x3 = 1/x1
        let vec = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let inv_vec = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        
        // Add inv_vec constraint using the correct method name and remove references
        let result = builder.add_inv_vec_constraint(vec, inv_vec);
        assert!(result.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "inv_vec constraint proof should be valid");
        println!("inv_vec constraint test passed!");
    }

    #[test]
    fn test_groth16_inv_vec_constraint_single_element() {
        // test single element inv_vec
        let a = BlsFr::from(7u64);
        let inv_a = a.inverse().unwrap(); // 1/7
        
        let inputs = vec![a, inv_a];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        let vec = vec![ArithmeticExpression::x(0)];
        let inv_vec = vec![ArithmeticExpression::x(1)];
        
        // Use the correct method name
        let result = builder.add_inv_vec_constraint(vec, inv_vec);
        assert!(result.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Single element inv_vec constraint proof should be valid");
        println!("Single element inv_vec constraint test passed!");
    }

    #[test]
    fn test_groth16_combined_vec_constraints() {
        // test conbined vector constraints
        let inputs = vec![
            BlsFr::from(2u64),  // x0 = 2
            BlsFr::from(3u64),  // x1 = 3
            BlsFr::from(2u64),  // x2 = 2 (equals x0)
            BlsFr::from(3u64),  // x3 = 3 (equals x1)
            BlsFr::from(4u64),  // x4 = 4 (2 * 2)
            BlsFr::from(9u64),  // x5 = 9 (3 * 3)
        ];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        // Vector definitions
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(2), ArithmeticExpression::x(3)];
        let mul_result = vec![ArithmeticExpression::x(4), ArithmeticExpression::x(5)];
        
        // Add constraints using correct method names:
        // 1. vec1 == vec2 (equal_vec)
        let result1 = builder.add_equal_vec_constraints(vec1.clone(), vec2.clone());
        assert!(result1.is_ok());
        
        // 2. vec1 * vec2 = mul_result (mul_vec)
        let result2 = builder.add_mul_vec_constraint(vec1, vec2, mul_result);
        assert!(result2.is_ok());

    builder.set_public_inputs(inputs.clone());
    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Combined vec constraints proof should be valid");
        println!("Combined vec constraints test passed!");
    }

    #[test]
    fn test_groth16_empty_vec_constraints() {
        // test empty vector constraints（border scenario）
        let inputs = vec![BlsFr::from(1u64)]; // Just a dummy input
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        let empty_vec1: Vec<ArithmeticExpression<BlsFr>> = vec![];
        let empty_vec2: Vec<ArithmeticExpression<BlsFr>> = vec![];
        let empty_vec3: Vec<ArithmeticExpression<BlsFr>> = vec![];
        
        // These should not add any constraints - use correct method names
        let result1 = builder.add_equal_vec_constraints(empty_vec1.clone(), empty_vec2.clone());
        assert!(result1.is_ok());
        
        let result2 = builder.add_mul_vec_constraint(empty_vec1.clone(), empty_vec2.clone(), empty_vec3.clone());
        assert!(result2.is_ok());
        
        let result3 = builder.add_inv_vec_constraint(empty_vec1, empty_vec2);
        assert!(result3.is_ok());

        // Add a simple constraint to make the system non-trivial
        builder.add_constraint(ArithmeticExpression::x(0) - BlsFr::from(1u64));

    assert!(builder.validate_constraints().is_ok());

        let mut rng = create_rng();

        let (pk, vk) = Groth16Prover::setup_bls12_381(&builder, &mut rng)
            .expect("Failed to generate keys");

        let proof = Groth16Prover::prove_bls12_381(&pk, builder, &mut rng)
            .expect("Failed to generate proof");

        let prepared_vk = Groth16Prover::prepare_verifying_key_bls12_381(&vk);
        let is_valid = Groth16Prover::verify_bls12_381(&prepared_vk, &inputs, &proof)
            .expect("Failed to verify proof");

        assert!(is_valid, "Empty vec constraints proof should be valid");
        println!("Empty vec constraints test passed!");
    }

}