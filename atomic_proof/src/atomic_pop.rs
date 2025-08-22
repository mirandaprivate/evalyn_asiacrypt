//! Implement the matmul protocol
//!
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::UniformRand;

use fsproof::helper_trans::Transcript;

use crate::pop::arithmetic_expression::{
    ArithmeticExpression, ConstraintSystemBuilder,
};

// Atomic Proof for a matrix operation
#[derive(Debug, Clone)]
pub struct AtomicPoP<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub hat_c: F,
    pub hat_a: F,
    pub hat_b: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub point_a: (Vec<F>, Vec<F>),
    pub point_b: (Vec<F>, Vec<F>),
    pub challenges: Vec<F>,
    pub responses: Vec<F>,
    pub mapping: AtomicPoPMapping,
    pub check: ArithmeticExpression<F>,
    pub link_xa: (Vec<ArithmeticExpression<F>>, Vec<ArithmeticExpression<F>>),
    pub link_xb: (Vec<ArithmeticExpression<F>>, Vec<ArithmeticExpression<F>>),
    // ready flags (minimal):
    // 0: pop transcript (set_pop_trans) done
    // 1: check expression set
    // 2: link_xa set
    // 3: link_xb set
    pub ready: (bool, bool, bool, bool),
}

// Atomic Proof for a matrix operation
#[derive(Debug, Clone)]
pub struct AtomicMultiPoP<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub hat_c: F,
    pub point_c: (Vec<F>, Vec<F>),
    pub hat_inputs: Vec<F>,
    pub point_inputs: Vec<(Vec<F>, Vec<F>)>, 
    pub challenges: Vec<F>,
    pub responses: Vec<F>,
    pub mapping: AtomicMultiPoPMapping,
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    // 0: pop transcript set
    // 1: check set
    // 2: links set
    pub ready: (bool, bool, bool),
}

// Mapping between the atomic proof elements and the transcript index
#[derive(Debug, Clone)]
pub struct AtomicPoPMapping {
    pub hat_c_index: usize,
    pub hat_a_index: usize,
    pub hat_b_index: usize,
    pub point_c_index: (Vec<usize>, Vec<usize>),
    pub point_a_index: (Vec<usize>, Vec<usize>),
    pub point_b_index: (Vec<usize>, Vec<usize>),
    pub challenges_index: Vec<usize>,
    pub responses_index: Vec<usize>,
}

// Mapping between the atomic proof elements and the transcript index
#[derive(Debug, Clone)]
pub struct AtomicMultiPoPMapping {
    pub hat_c_index: usize,
    pub point_c_index: (Vec<usize>, Vec<usize>),
    pub hat_inputs_index: Vec<usize>,
    pub point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
    pub challenges_index: Vec<usize>,
    pub responses_index: Vec<usize>,
}

impl AtomicPoPMapping {
    // Create a new mapping from the atomic proof elements to the transcript index
    pub fn new() -> Self {
        Self {
            hat_c_index: 0,
            hat_a_index: 0,
            hat_b_index: 0,
            point_c_index: (Vec::new(), Vec::new()),
            point_a_index: (Vec::new(), Vec::new()),
            point_b_index: (Vec::new(), Vec::new()),
            challenges_index: Vec::new(),
            responses_index: Vec::new(),
        }
    }

    // Set mapping
    pub fn set_message_mapping(
        &mut self,
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
    ) {
        self.hat_c_index = hat_c_index;
        self.point_c_index = point_c_index;
    }

    // Set trans mapping
    pub fn set_trans_mapping(
        &mut self,
        hat_a_index: usize,
        hat_b_index: usize,
        point_a_index: (Vec<usize>, Vec<usize>),
        point_b_index: (Vec<usize>, Vec<usize>),
        challenges_index: Vec<usize>,
        responses_index: Vec<usize>,
    ) {
        self.hat_a_index = hat_a_index;
        self.hat_b_index = hat_b_index;
        self.point_a_index = point_a_index;
        self.point_b_index = point_b_index;
        self.challenges_index = challenges_index;
        self.responses_index = responses_index;
    } 
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicPoP<F> {
    // Create a new atomic proof
    pub fn new() -> Self {
        Self {
            hat_c: F::zero(),
            hat_a: F::zero(),
            hat_b: F::zero(),
            point_c: (Vec::new(), Vec::new()),
            point_a: (Vec::new(), Vec::new()),
            point_b: (Vec::new(), Vec::new()),
            challenges: Vec::new(),
            responses: Vec::new(),
            mapping: AtomicPoPMapping::new(),
            check: ArithmeticExpression::constant(F::zero()),
            link_xa: (Vec::new(), Vec::new()),
            link_xb: (Vec::new(), Vec::new()),
            ready: (false, false, false, false),
        }
    }

    pub fn set_message(
        &mut self,
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_mapping: usize,
        point_c_mapping: (Vec<usize>, Vec<usize>),
    ) {
        self.hat_c = hat_c;
        self.point_c = point_c;
        self.mapping.set_message_mapping(hat_c_mapping, point_c_mapping);
    }

    pub fn set_pop_trans(&mut self,
        hat_a: F,
        hat_b: F,
        point_a: (Vec<F>, Vec<F>),
        point_b: (Vec<F>, Vec<F>),
        challenges: Vec<F>,
        responses: Vec<F>,
        hat_a_index: usize,
        hat_b_index: usize,
        point_a_index: (Vec<usize>, Vec<usize>),
        point_b_index: (Vec<usize>, Vec<usize>),
        challenges_index: Vec<usize>,
        responses_index: Vec<usize>,
    ) {
        self.mapping.set_trans_mapping(
            hat_a_index,
            hat_b_index,
            point_a_index,
            point_b_index,
            challenges_index,
            responses_index,
        );

        self.hat_a = hat_a;
        self.hat_b = hat_b;
        self.point_a = point_a;
        self.point_b = point_b;
        self.challenges = challenges;
        self.responses = responses;

    self.ready.0 = true; // pop transcript set
    }

    pub fn set_check(&mut self, check: ArithmeticExpression<F>) {
        self.check = check;
    self.ready.1 = true; // check set
    }

    pub fn set_link_xa(&mut self, link_xa: (Vec<ArithmeticExpression<F>>, Vec<ArithmeticExpression<F>>)) {
        self.link_xa = link_xa;
    self.ready.2 = true; // link xa set
    }

    pub fn set_link_xb(&mut self, link_xb: (Vec<ArithmeticExpression<F>>, Vec<ArithmeticExpression<F>>)) {
        self.link_xb = link_xb;
    self.ready.3 = true; // link xb set
    }

    pub fn is_ready(&self) -> bool {
    self.ready.0 && self.ready.1 && self.ready.2 && self.ready.3
    }

    pub fn get_c(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.hat_c, self.point_c.clone())
    }

    pub fn get_c_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.hat_c_index, self.mapping.point_c_index.clone())
    }

    pub fn get_a(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.hat_a, self.point_a.clone())
    }

    pub fn get_a_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.hat_a_index, self.mapping.point_a_index.clone())
    }
    
    pub fn get_b(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.hat_b, self.point_b.clone())
    }

    pub fn get_b_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.hat_b_index, self.mapping.point_b_index.clone())
    }

    pub fn synthesize_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        // 1. Add the main 'check' constraint
        cs_builder.add_constraint(self.check.clone());

        // 2. Add 'link_xa' constraints (left and right)
        for constraint in &self.link_xa.0 {
            cs_builder.add_constraint(constraint.clone());
        }
        for constraint in &self.link_xa.1 {
            cs_builder.add_constraint(constraint.clone());
        }

        // 3. Add 'link_xb' constraints (left and right)
        for constraint in &self.link_xb.0 {
            cs_builder.add_constraint(constraint.clone());
        }
        for constraint in &self.link_xb.1 {
            cs_builder.add_constraint(constraint.clone());
        }
        
        true
    }

    pub fn verify_via_atomic_pop(&self, transcript: &Transcript<F>) -> bool {

        let input = transcript.get_fs_proof_vec();
        // treat transcript values as public inputs; no private inputs at verify stage
        if self.check.evaluate(&input, &[]).unwrap() != F::zero() {
            return false;
        }

        for expr in &self.link_xa.0 {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        for expr in &self.link_xa.1 {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        for expr in &self.link_xb.0 {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        for expr in &self.link_xb.1 {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        for expr in &self.link_xb.1 {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        true
    }
}

impl AtomicMultiPoPMapping {
    // Create a new mapping from the atomic proof elements to the transcript index
    pub fn new() -> Self {
        Self {
            hat_c_index: 0,
            point_c_index: (Vec::new(), Vec::new()),
            hat_inputs_index: Vec::new(),
            point_inputs_index: Vec::new(),
            challenges_index: Vec::new(),
            responses_index: Vec::new(),
        }
    }

    // Set mapping for the output commitment
    pub fn set_message_mapping(
        &mut self,
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
    ) {
        self.hat_c_index = hat_c_index;
        self.point_c_index = point_c_index;
    }

    // Set mapping for the transcript elements
    pub fn set_trans_mapping(
        &mut self,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
        challenges_index: Vec<usize>,
        responses_index: Vec<usize>,
    ) {
        self.hat_inputs_index = hat_inputs_index;
        self.point_inputs_index = point_inputs_index;
        self.challenges_index = challenges_index;
        self.responses_index = responses_index;
    }
}

impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMultiPoP<F> {
    // Create a new atomic proof for multiple inputs
    pub fn new() -> Self {
        Self {
            hat_c: F::zero(),
            point_c: (Vec::new(), Vec::new()),
            hat_inputs: Vec::new(),
            point_inputs: Vec::new(),
            challenges: Vec::new(),
            responses: Vec::new(),
            mapping: AtomicMultiPoPMapping::new(),
            check: ArithmeticExpression::constant(F::zero()),
            link_inputs: Vec::new(),
            ready: (false, false, false),
        }
    }

    // Set the output commitment message
    pub fn set_message(
        &mut self,
        hat_c: F,
        point_c: (Vec<F>, Vec<F>),
        hat_c_index: usize,
        point_c_index: (Vec<usize>, Vec<usize>),
    ) {
        self.hat_c = hat_c;
        self.point_c = point_c;
        self.mapping.set_message_mapping(hat_c_index, point_c_index);
    }

    // Set the proof transcript
    pub fn set_pop_trans(
        &mut self,
        hat_inputs: Vec<F>,
        point_inputs: Vec<(Vec<F>, Vec<F>)>,
        challenges: Vec<F>,
        responses: Vec<F>,
        hat_inputs_index: Vec<usize>,
        point_inputs_index: Vec<(Vec<usize>, Vec<usize>)>,
        challenges_index: Vec<usize>,
        responses_index: Vec<usize>,
    ) {
        self.mapping.set_trans_mapping(
            hat_inputs_index,
            point_inputs_index,
            challenges_index,
            responses_index,
        );

        self.hat_inputs = hat_inputs;
        self.point_inputs = point_inputs;
        self.challenges = challenges;
        self.responses = responses;

    self.ready.0= true; // pop transcript set
    }

    // Set the verification check expression
    pub fn set_check(&mut self, check: ArithmeticExpression<F>) {
        self.check = check;
    self.ready.1 = true; // check set
    }

    // Set the linking expressions for inputs
    pub fn set_link_inputs(&mut self, link_inputs: Vec<ArithmeticExpression<F>>) {
        self.link_inputs = link_inputs;
    self.ready.2 = true; // links set
    }

    // Check if all components of the proof are set
    pub fn is_ready(&self) -> bool {
        self.ready.0 && self.ready.1 && self.ready.2
    }

    // Get the output commitment
    pub fn get_c(&self) -> (F, (Vec<F>, Vec<F>)) {
        (self.hat_c, self.point_c.clone())
    }

    // Get the mapping indices for the output commitment
    pub fn get_c_index(&self) -> (usize, (Vec<usize>, Vec<usize>)) {
        (self.mapping.hat_c_index, self.mapping.point_c_index.clone())
    }

    // Get the input commitments
    pub fn get_inputs(&self) -> (Vec<F>, Vec<(Vec<F>, Vec<F>)>) {
        (self.hat_inputs.clone(), self.point_inputs.clone())
    }

    // Get the mapping indices for the input commitments
    pub fn get_inputs_index(&self) -> (Vec<usize>, Vec<(Vec<usize>, Vec<usize>)>) {
        (self.mapping.hat_inputs_index.clone(), self.mapping.point_inputs_index.clone())
    }

    // Synthesize constraints for the constraint system builder
    pub fn synthesize_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        // 1. Add the main 'check' constraint
        cs_builder.add_constraint(self.check.clone());

        // 2. Add 'link_inputs' constraints
        for constraint in &self.link_inputs {
            cs_builder.add_constraint(constraint.clone());
        }
        
        true
    }

    // Verify the proof using the transcript
    pub fn verify_via_atomic_pop(&self, transcript: &Transcript<F>) -> bool {
        let input = transcript.get_fs_proof_vec();
        if self.check.evaluate(&input, &[]).unwrap() != F::zero() {
            return false;
        }

        for expr in &self.link_inputs {
            if expr.evaluate(&input, &[]).unwrap() != F::zero() {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;


    #[test]
    fn test_atomic_pop_mapping_creation() {
        println!("=== Testing AtomicPoPMapping Creation ===");
        
        let mapping = AtomicPoPMapping::new();
        
        // Check initial values
        assert_eq!(mapping.hat_c_index, 0);
        assert_eq!(mapping.hat_a_index, 0);
        assert_eq!(mapping.hat_b_index, 0);
        assert_eq!(mapping.point_c_index.0.len(), 0);
        assert_eq!(mapping.point_c_index.1.len(), 0);
        assert_eq!(mapping.point_a_index.0.len(), 0);
        assert_eq!(mapping.point_a_index.1.len(), 0);
        assert_eq!(mapping.point_b_index.0.len(), 0);
        assert_eq!(mapping.point_b_index.1.len(), 0);
        assert_eq!(mapping.challenges_index.len(), 0);
        assert_eq!(mapping.responses_index.len(), 0);
        
        println!("✅ AtomicPoPMapping creation test passed!");
    }

    #[test]
    fn test_atomic_pop_mapping_set_message_mapping() {
        println!("=== Testing AtomicPoPMapping Message Mapping ===");
        
        let mut mapping = AtomicPoPMapping::new();
        
        let hat_c_index = 42;
        let point_c_index = (vec![1, 2, 3], vec![4, 5, 6]);
        
        mapping.set_message_mapping(hat_c_index, point_c_index.clone());
        
        assert_eq!(mapping.hat_c_index, hat_c_index);
        assert_eq!(mapping.point_c_index.0, point_c_index.0);
        assert_eq!(mapping.point_c_index.1, point_c_index.1);
        
        println!("✅ AtomicPoPMapping message mapping test passed!");
    }

    #[test]
    fn test_atomic_pop_mapping_set_trans_mapping() {
        println!("=== Testing AtomicPoPMapping Trans Mapping ===");
        
        let mut mapping = AtomicPoPMapping::new();
        
        let hat_a_index = 10;
        let hat_b_index = 20;
        let point_a_index = (vec![11, 12], vec![13, 14]);
        let point_b_index = (vec![21, 22], vec![23, 24]);
        let challenges_index = vec![100, 101, 102];
        let responses_index = vec![200, 201, 202];
        
        mapping.set_trans_mapping(
            hat_a_index,
            hat_b_index,
            point_a_index.clone(),
            point_b_index.clone(),
            challenges_index.clone(),
            responses_index.clone(),
        );
        
        assert_eq!(mapping.hat_a_index, hat_a_index);
        assert_eq!(mapping.hat_b_index, hat_b_index);
        assert_eq!(mapping.point_a_index.0, point_a_index.0);
        assert_eq!(mapping.point_a_index.1, point_a_index.1);
        assert_eq!(mapping.point_b_index.0, point_b_index.0);
        assert_eq!(mapping.point_b_index.1, point_b_index.1);
        assert_eq!(mapping.challenges_index, challenges_index);
        assert_eq!(mapping.responses_index, responses_index);
        
        println!("✅ AtomicPoPMapping trans mapping test passed!");
    }

    #[test]
    fn test_atomic_pop_creation() {
        println!("=== Testing AtomicPoP Creation ===");
        
        let hat_c = BlsFr::from(100u64);
        let point_c = (vec![BlsFr::from(1u64), BlsFr::from(2u64)], 
                       vec![BlsFr::from(3u64), BlsFr::from(4u64)]);
        let hat_c_mapping = 5;
        let point_c_mapping = (vec![0, 1], vec![2, 3]);
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            hat_c,
            point_c.clone(),
            hat_c_mapping,
            point_c_mapping.clone(),
        );
        
        // Check initial values
        assert_eq!(atomic_pop.hat_c, hat_c);
        assert_eq!(atomic_pop.hat_a, BlsFr::zero());
        assert_eq!(atomic_pop.hat_b, BlsFr::zero());
        assert_eq!(atomic_pop.point_c.0, point_c.0);
        assert_eq!(atomic_pop.point_c.1, point_c.1);
        assert_eq!(atomic_pop.point_a.0.len(), 0);
        assert_eq!(atomic_pop.point_a.1.len(), 0);
        assert_eq!(atomic_pop.point_b.0.len(), 0);
        assert_eq!(atomic_pop.point_b.1.len(), 0);
        assert_eq!(atomic_pop.challenges.len(), 0);
        assert_eq!(atomic_pop.responses.len(), 0);
        
        // Check mapping
        assert_eq!(atomic_pop.mapping.hat_c_index, hat_c_mapping);
        assert_eq!(atomic_pop.mapping.point_c_index.0, point_c_mapping.0);
        assert_eq!(atomic_pop.mapping.point_c_index.1, point_c_mapping.1);
        
    // Check ready state (after set_message only, pop transcript NOT set yet)
    assert_eq!(atomic_pop.ready.0, false);  // Pop transcript not set
    assert_eq!(atomic_pop.ready.1, false); // Check not set
    assert_eq!(atomic_pop.ready.2, false); // Link xa not set
    assert_eq!(atomic_pop.ready.3, false); // Link xb not set
        assert!(!atomic_pop.is_ready());
        
        println!("✅ AtomicPoP creation test passed!");
    }

    #[test]
    fn test_atomic_pop_set_pop_trans() {
        println!("=== Testing AtomicPoP Set Pop Trans ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
        let hat_a = BlsFr::from(200u64);
        let hat_b = BlsFr::from(300u64);
        let point_a = (vec![BlsFr::from(10u64), BlsFr::from(11u64)], 
                       vec![BlsFr::from(12u64), BlsFr::from(13u64)]);
        let point_b = (vec![BlsFr::from(20u64), BlsFr::from(21u64)], 
                       vec![BlsFr::from(22u64), BlsFr::from(23u64)]);
        let challenges = vec![BlsFr::from(500u64), BlsFr::from(501u64)];
        let responses = vec![BlsFr::from(600u64), BlsFr::from(601u64)];
        let hat_a_index = 10;
        let hat_b_index = 11;
        let point_a_index = (vec![12, 13], vec![14, 15]);
        let point_b_index = (vec![16, 17], vec![18, 19]);
        let challenges_index = vec![20, 21];
        let responses_index = vec![30, 31];
        
        atomic_pop.set_pop_trans(
            hat_a,
            hat_b,
            point_a.clone(),
            point_b.clone(),
            challenges.clone(),
            responses.clone(),
            hat_a_index,
            hat_b_index,
            point_a_index.clone(),
            point_b_index.clone(),
            challenges_index.clone(),
            responses_index.clone(),
        );
        
        // Check values were set correctly
        assert_eq!(atomic_pop.hat_a, hat_a);
        assert_eq!(atomic_pop.hat_b, hat_b);
        assert_eq!(atomic_pop.point_a.0, point_a.0);
        assert_eq!(atomic_pop.point_a.1, point_a.1);
        assert_eq!(atomic_pop.point_b.0, point_b.0);
        assert_eq!(atomic_pop.point_b.1, point_b.1);
        assert_eq!(atomic_pop.challenges, challenges);
        assert_eq!(atomic_pop.responses, responses);
        
        // Check mapping
        assert_eq!(atomic_pop.mapping.hat_a_index, hat_a_index);
        assert_eq!(atomic_pop.mapping.hat_b_index, hat_b_index);
        assert_eq!(atomic_pop.mapping.point_a_index.0, point_a_index.0);
        assert_eq!(atomic_pop.mapping.point_a_index.1, point_a_index.1);
        assert_eq!(atomic_pop.mapping.point_b_index.0, point_b_index.0);
        assert_eq!(atomic_pop.mapping.point_b_index.1, point_b_index.1);
        assert_eq!(atomic_pop.mapping.challenges_index, challenges_index);
        assert_eq!(atomic_pop.mapping.responses_index, responses_index);
        
    // Check ready state (only transcript set)
    assert_eq!(atomic_pop.ready.0, true); // Pop transcript now set
    assert_eq!(atomic_pop.ready.1, false);
    assert_eq!(atomic_pop.ready.2, false);
    assert_eq!(atomic_pop.ready.3, false);
    assert!(!atomic_pop.is_ready()); // Not fully ready
        
        println!("✅ AtomicPoP set pop trans test passed!");
    }

    #[test]
    fn test_atomic_pop_set_check() {
        println!("=== Testing AtomicPoP Set Check ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
        let check = ArithmeticExpression::constant(BlsFr::from(42u64));
        atomic_pop.set_check(check);
        
    // Check ready state (only check set, transcript missing so still not ready)
    assert_eq!(atomic_pop.ready.0, false);
    assert_eq!(atomic_pop.ready.1, true); // Check now set
    assert_eq!(atomic_pop.ready.2, false);
    assert_eq!(atomic_pop.ready.3, false);
    assert!(!atomic_pop.is_ready());
        
        println!("✅ AtomicPoP set check test passed!");
    }

    #[test]
    fn test_atomic_pop_set_link_xa() {
        println!("=== Testing AtomicPoP Set Link XA ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
        let link_xa = (
            vec![ArithmeticExpression::constant(BlsFr::from(1u64)), ArithmeticExpression::constant(BlsFr::from(2u64))],
            vec![ArithmeticExpression::constant(BlsFr::from(3u64))]
        );
        atomic_pop.set_link_xa(link_xa.clone());
        
        // Check values were set
        assert_eq!(atomic_pop.link_xa.0.len(), link_xa.0.len());
        assert_eq!(atomic_pop.link_xa.1.len(), link_xa.1.len());
        
    // Check ready state (only link_xa set, transcript missing)
    assert_eq!(atomic_pop.ready.0, false);
    assert_eq!(atomic_pop.ready.1, false);
    assert_eq!(atomic_pop.ready.2, true); // link_xa set
    assert_eq!(atomic_pop.ready.3, false);
    assert!(!atomic_pop.is_ready());
        
        println!("✅ AtomicPoP set link xa test passed!");
    }

    #[test]
    fn test_atomic_pop_set_link_xb() {
        println!("=== Testing AtomicPoP Set Link XB ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
        let link_xb = (
            vec![ArithmeticExpression::constant(BlsFr::from(4u64))],
            vec![ArithmeticExpression::constant(BlsFr::from(5u64)), ArithmeticExpression::constant(BlsFr::from(6u64))]
        );
        atomic_pop.set_link_xb(link_xb.clone());
        
        // Check values were set
        assert_eq!(atomic_pop.link_xb.0.len(), link_xb.0.len());
        assert_eq!(atomic_pop.link_xb.1.len(), link_xb.1.len());
        
        // Check ready state
    // Check ready state (only link_xb set, transcript missing)
    assert_eq!(atomic_pop.ready.0, false);
    assert_eq!(atomic_pop.ready.1, false);
    assert_eq!(atomic_pop.ready.2, false);
    assert_eq!(atomic_pop.ready.3, true); // link_xb set
    assert!(!atomic_pop.is_ready());
        
        println!("✅ AtomicPoP set link xb test passed!");
    }

    #[test]
    fn test_atomic_pop_is_ready_complete() {
        println!("=== Testing AtomicPoP Complete Ready State ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
        // Initially not ready
        assert!(!atomic_pop.is_ready());
        
        // Set all components
        atomic_pop.set_pop_trans(
            BlsFr::from(200u64),
            BlsFr::from(300u64),
            (vec![BlsFr::from(10u64)], vec![BlsFr::from(11u64)]),
            (vec![BlsFr::from(20u64)], vec![BlsFr::from(21u64)]),
            vec![BlsFr::from(500u64)],
            vec![BlsFr::from(600u64)],
            10, 11,
            (vec![12], vec![13]),
            (vec![14], vec![15]),
            vec![20],
            vec![30],
        );
        
        atomic_pop.set_check(ArithmeticExpression::constant(BlsFr::from(42u64)));
        atomic_pop.set_link_xa((vec![ArithmeticExpression::constant(BlsFr::from(1u64))], vec![ArithmeticExpression::constant(BlsFr::from(2u64))]));
        atomic_pop.set_link_xb((vec![ArithmeticExpression::constant(BlsFr::from(3u64))], vec![ArithmeticExpression::constant(BlsFr::from(4u64))]));
        
        // Now should be ready
        assert!(atomic_pop.is_ready());
        
        // Check all ready flags
    assert_eq!(atomic_pop.ready.0, true); // transcript
    assert_eq!(atomic_pop.ready.1, true); // check
    assert_eq!(atomic_pop.ready.2, true); // link_xa
    assert_eq!(atomic_pop.ready.3, true); // link_xb
        
        println!("✅ AtomicPoP complete ready state test passed!");
    }

    #[test]
    fn test_atomic_pop_partial_ready_states() {
        println!("=== Testing AtomicPoP Partial Ready States ===");
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            BlsFr::from(100u64),
            (vec![BlsFr::from(1u64)], vec![BlsFr::from(2u64)]),
            0,
            (vec![0], vec![1]),
        );
        
    // Test each component individually (after set_message only nothing set)
    assert_eq!(atomic_pop.ready.0, false);
    assert_eq!(atomic_pop.ready.1, false);
    assert_eq!(atomic_pop.ready.2, false);
    assert_eq!(atomic_pop.ready.3, false);
        assert!(!atomic_pop.is_ready());
        
        // Set trans
        atomic_pop.set_pop_trans(
            BlsFr::from(200u64),
            BlsFr::from(300u64),
            (vec![BlsFr::from(10u64)], vec![BlsFr::from(11u64)]),
            (vec![BlsFr::from(20u64)], vec![BlsFr::from(21u64)]),
            vec![BlsFr::from(500u64)],
            vec![BlsFr::from(600u64)],
            10, 11,
            (vec![12], vec![13]),
            (vec![14], vec![15]),
            vec![20],
            vec![30],
        );
        
    assert_eq!(atomic_pop.ready.0, true); // transcript set
        assert!(!atomic_pop.is_ready());
        
        // Set check
        atomic_pop.set_check(ArithmeticExpression::constant(BlsFr::from(42u64)));
    assert_eq!(atomic_pop.ready.1, true); // check set
        assert!(!atomic_pop.is_ready());
        
        // Set link xa
        atomic_pop.set_link_xa((vec![ArithmeticExpression::constant(BlsFr::from(1u64))], vec![ArithmeticExpression::constant(BlsFr::from(2u64))]));
    assert_eq!(atomic_pop.ready.2, true); // link_xa set
        assert!(!atomic_pop.is_ready());
        
        // Set link xb - now should be ready
        atomic_pop.set_link_xb((vec![ArithmeticExpression::constant(BlsFr::from(3u64))], vec![ArithmeticExpression::constant(BlsFr::from(4u64))]));
    assert_eq!(atomic_pop.ready.3, true); // link_xb set
        assert!(atomic_pop.is_ready());
        
        println!("✅ AtomicPoP partial ready states test passed!");
    }

    #[test]
    fn test_atomic_pop_field_values_integrity() {
        println!("=== Testing AtomicPoP Field Values Integrity ===");
        
        // Create with specific values
        let hat_c = BlsFr::from(12345u64);
        let point_c = (
            vec![BlsFr::from(111u64), BlsFr::from(222u64), BlsFr::from(333u64)],
            vec![BlsFr::from(444u64), BlsFr::from(555u64)]
        );
        
        let mut atomic_pop = AtomicPoP::new();
        atomic_pop.set_message(
            hat_c,
            point_c.clone(),
            99,
            (vec![0, 1, 2], vec![3, 4]),
        );
        
        // Verify initial values
        assert_eq!(atomic_pop.hat_c, hat_c);
        assert_eq!(atomic_pop.point_c.0.len(), 3);
        assert_eq!(atomic_pop.point_c.1.len(), 2);
        assert_eq!(atomic_pop.point_c.0[0], BlsFr::from(111u64));
        assert_eq!(atomic_pop.point_c.0[1], BlsFr::from(222u64));
        assert_eq!(atomic_pop.point_c.0[2], BlsFr::from(333u64));
        assert_eq!(atomic_pop.point_c.1[0], BlsFr::from(444u64));
        assert_eq!(atomic_pop.point_c.1[1], BlsFr::from(555u64));
        
        // Set more values and verify
        let hat_a = BlsFr::from(11111u64);
        let hat_b = BlsFr::from(22222u64);
        let point_a = (
            vec![BlsFr::from(1000u64)],
            vec![BlsFr::from(2000u64), BlsFr::from(3000u64)]
        );
        let point_b = (
            vec![BlsFr::from(4000u64), BlsFr::from(5000u64)],
            vec![BlsFr::from(6000u64)]
        );
        let challenges = vec![
            BlsFr::from(7000u64),
            BlsFr::from(8000u64),
            BlsFr::from(9000u64)
        ];
        let responses = vec![
            BlsFr::from(17000u64),
            BlsFr::from(18000u64),
            BlsFr::from(19000u64)
        ];
        
        atomic_pop.set_pop_trans(
            hat_a,
            hat_b,
            point_a.clone(),
            point_b.clone(),
            challenges.clone(),
            responses.clone(),
            50, 51,
            (vec![52], vec![53, 54]),
            (vec![55, 56], vec![57]),
            vec![58, 59, 60],
            vec![68, 69, 70],
        );
        
        // Verify all values were set correctly
        assert_eq!(atomic_pop.hat_a, hat_a);
        assert_eq!(atomic_pop.hat_b, hat_b);
        assert_eq!(atomic_pop.point_a.0, point_a.0);
        assert_eq!(atomic_pop.point_a.1, point_a.1);
        assert_eq!(atomic_pop.point_b.0, point_b.0);
        assert_eq!(atomic_pop.point_b.1, point_b.1);
        assert_eq!(atomic_pop.challenges, challenges);
        assert_eq!(atomic_pop.responses, responses);
        
        // Verify mapping values
        assert_eq!(atomic_pop.mapping.hat_c_index, 99);
        assert_eq!(atomic_pop.mapping.hat_a_index, 50);
        assert_eq!(atomic_pop.mapping.hat_b_index, 51);
        assert_eq!(atomic_pop.mapping.point_c_index.0, vec![0, 1, 2]);
        assert_eq!(atomic_pop.mapping.point_c_index.1, vec![3, 4]);
        assert_eq!(atomic_pop.mapping.point_a_index.0, vec![52]);
        assert_eq!(atomic_pop.mapping.point_a_index.1, vec![53, 54]);
        assert_eq!(atomic_pop.mapping.point_b_index.0, vec![55, 56]);
        assert_eq!(atomic_pop.mapping.point_b_index.1, vec![57]);
        assert_eq!(atomic_pop.mapping.challenges_index, vec![58, 59, 60]);
        assert_eq!(atomic_pop.mapping.responses_index, vec![68, 69, 70]);
        
        println!("✅ AtomicPoP field values integrity test passed!");
    }
}
