//! ```text
//! Prove all entries in a vector are within the range [0, 1, ..., 2^k - 1]
//! 
//! Projections of this structured table can be computed recursively in logarithmic time
//! 
//! Equivalently, each non-negative entry is at most k bits
//! 
//! Achieved via a 2^k sized lookup proof
//! ```
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{UniformRand};

use fsproof::helper_trans::Transcript;

use atomic_proof::pop::arithmetic_expression::{ArithmeticExpression, ConstraintSystemBuilder};
use atomic_proof::AtomicMatProtocol;
use crate::protocols::lookup::LookUp;


// LookUp already imported above


#[derive(Debug, Clone)]
pub struct RangeProofMapping {
    pub target_hat_index: usize,
    pub target_point_index: (Vec<usize>, Vec<usize>),
    pub table_hat_index: usize,
    pub table_point_index: (Vec<usize>, Vec<usize>),
    pub auxiliary_target_hat_index: usize,
    pub auxiliary_target_point_index: (Vec<usize>, Vec<usize>),
    pub auxiliary_table_hat_index: usize,
    pub auxiliary_table_point_index: (Vec<usize>, Vec<usize>),
}

// (Removed placeholder LinearCombination/BatchPoint definitions; using real types from atomic_proof)

#[derive(Debug, Clone)]
pub struct RangeProofAtomicPoP<F: PrimeField> {
    pub target_hat: F,
    pub target_point: (Vec<F>, Vec<F>),
    pub table_hat: F,
    pub table_point: (Vec<F>, Vec<F>),
    pub auxiliary_target_hat: F,
    pub auxiliary_target_point: (Vec<F>, Vec<F>),
    pub auxiliary_table_hat: F,
    pub auxiliary_table_point: (Vec<F>, Vec<F>),
    pub mapping: RangeProofMapping,
    pub check: ArithmeticExpression<F>,
    pub link_inputs: Vec<ArithmeticExpression<F>>,
    pub ready: (bool, bool, bool),
}

#[derive(Debug, Clone)]
pub struct RangeProof<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize>
{
    pub k: usize,
    pub target_len: usize,
    pub atomic_pop: RangeProofAtomicPoP<F>,
    pub lookup: LookUp<F>,
}

impl<F: PrimeField> RangeProofAtomicPoP<F> {
    pub fn new() -> Self {
        Self {
            target_hat: F::zero(),
            target_point: (Vec::new(), Vec::new()),
            table_hat: F::zero(),
            table_point: (Vec::new(), Vec::new()),
            auxiliary_target_hat: F::zero(),
            auxiliary_target_point: (Vec::new(), Vec::new()),
            auxiliary_table_hat: F::zero(),
            auxiliary_table_point: (Vec::new(), Vec::new()),
            mapping: RangeProofMapping {
                target_hat_index: 0,
                target_point_index: (Vec::new(), Vec::new()),
                table_hat_index: 0,
                table_point_index: (Vec::new(), Vec::new()),
                auxiliary_target_hat_index: 0,
                auxiliary_target_point_index: (Vec::new(), Vec::new()),
                auxiliary_table_hat_index: 0,
                auxiliary_table_point_index: (Vec::new(), Vec::new()),
            },
            check: ArithmeticExpression::constant(F::zero()),
            link_inputs: Vec::new(),
            ready: (false, false, false),
        }
    }


    pub fn set_pop_trans(
        &mut self,
        target_hat: F,
        target_point: (Vec<F>, Vec<F>),
        table_hat: F,
        table_point: (Vec<F>, Vec<F>),
        auxiliary_target_hat: F,
        auxiliary_target_point: (Vec<F>, Vec<F>),
        auxiliary_table_hat: F,
        auxiliary_table_point: (Vec<F>, Vec<F>),
        target_hat_index: usize,
        target_point_index: (Vec<usize>, Vec<usize>),
        table_hat_index: usize,
        table_point_index: (Vec<usize>, Vec<usize>),
        auxiliary_target_hat_index: usize,
        auxiliary_target_point_index: (Vec<usize>, Vec<usize>),
        auxiliary_table_hat_index: usize,
        auxiliary_table_point_index: (Vec<usize>, Vec<usize>),
    ) {
        self.target_hat = target_hat;
        self.target_point = target_point;
        self.table_hat = table_hat;
        self.table_point = table_point;
        self.auxiliary_target_hat = auxiliary_target_hat;
        self.auxiliary_target_point = auxiliary_target_point;
        self.auxiliary_table_hat = auxiliary_table_hat;
        self.auxiliary_table_point = auxiliary_table_point;
        self.mapping.target_hat_index = target_hat_index;
        self.mapping.target_point_index = target_point_index;
        self.mapping.table_hat_index = table_hat_index;
        self.mapping.table_point_index = table_point_index;
        self.mapping.auxiliary_target_hat_index = auxiliary_target_hat_index;
        self.mapping.auxiliary_target_point_index = auxiliary_target_point_index;
        self.mapping.auxiliary_table_hat_index = auxiliary_table_hat_index;
        self.mapping.auxiliary_table_point_index = auxiliary_table_point_index;
        self.ready.0 = true;
    }

    pub fn set_check(&mut self, check: ArithmeticExpression<F>) {
        self.check = check;
        self.ready.1 = true;
    }

    pub fn set_links(&mut self, link_inputs: Vec<ArithmeticExpression<F>>) {
        self.link_inputs = link_inputs;
        self.ready.2 = true;
    }

    pub fn is_ready(&self) -> bool {
        self.ready.0 && self.ready.1 && self.ready.2
    }
}



impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> RangeProof<F> {
    pub fn new(k: usize, target_len: usize) -> Self {
        
        Self {
            k,
            target_len,
            atomic_pop: RangeProofAtomicPoP::new(),
            lookup: LookUp::new(1, target_len, 1usize << k)
        }
    }

    pub fn default() -> Self {
        Self::new(0, 0)
    }

    pub fn set_input(
        &mut self,
        target: Vec<F>,
        auxiliary_target: Vec<F>,
        auxiliary_table: Vec<F>,
    ) {
    let table: Vec<F> = (0..(1 << self.k)).map(|i| F::from(i as u64)).collect();
    self.lookup.set_input(vec![target], vec![table], auxiliary_target, auxiliary_table);
    }
}



impl<F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize> AtomicMatProtocol<F> for RangeProof<F> {
    fn clear(&mut self) {
        self.lookup.clear();
    }
    
    fn reduce_prover(&mut self, trans: &mut Transcript<F>) -> bool {
        let flag_lookup = self.lookup.reduce_prover(trans);

        // We only have one column in this specialized Lookup (num_col = 1)
        let target_hat = self.lookup.atomic_pop.target_hats[0];
        let target_point = self.lookup.atomic_pop.target_points[0].clone();
        let table_hat = self.lookup.atomic_pop.table_hats[0];
        let table_point = self.lookup.atomic_pop.table_points[0].clone();
        let auxiliary_target_hat = self.lookup.atomic_pop.auxiliary_target_hat;
        let auxiliary_target_point = self.lookup.atomic_pop.auxiliary_target_points.clone();
        let auxiliary_table_hat = self.lookup.atomic_pop.auxiliary_table_hat;
        let auxiliary_table_point = self.lookup.atomic_pop.auxiliary_table_points.clone();

        // Indices from lookup mapping
        let target_hat_index = self.lookup.atomic_pop.mapping.target_hats_index[0];
        let target_point_index = self.lookup.atomic_pop.mapping.target_points_index[0].clone();
        let table_hat_index = self.lookup.atomic_pop.mapping.table_hats_index[0];
        let table_point_index = self.lookup.atomic_pop.mapping.table_points_index[0].clone();
        let auxiliary_target_hat_index = self.lookup.atomic_pop.mapping.auxiliary_target_hat_index;
        let auxiliary_target_point_index = self.lookup.atomic_pop.mapping.auxiliary_target_points_index.clone();
        let auxiliary_table_hat_index = self.lookup.atomic_pop.mapping.auxiliary_table_hat_index;
        let auxiliary_table_point_index = self.lookup.atomic_pop.mapping.auxiliary_table_points_index.clone();

        let table_point_clone = table_point.clone();
        
        self.atomic_pop.set_pop_trans(
            target_hat,
            target_point,
            table_hat.clone(),
            table_point_clone,
            auxiliary_target_hat,
            auxiliary_target_point,
            auxiliary_table_hat,
            auxiliary_table_point,
            target_hat_index,
            target_point_index,
            table_hat_index,
            table_point_index,
            auxiliary_target_hat_index,
            auxiliary_target_point_index,
            auxiliary_table_hat_index,
            auxiliary_table_point_index,
        );

        // Range table projection identity (simple consistency check)
        let two_powers = (0..self.k).map(|i| F::from(2u64).pow(&[i as u64])).collect::<Vec<F>>();
        let xx = table_point.0.clone().into_iter().rev().collect::<Vec<_>>();

        let mut cur_ip = F::zero();
        let mut cur_mul = F::one();

        for i in 0..self.k {
            cur_ip = cur_ip.clone() * (F::one() + &xx[i]) + cur_mul.clone() * &two_powers[i] * &xx[i];
            cur_mul =  cur_mul * (F::one() + &xx[i]);
        }
        let flag_table = table_hat == cur_ip;
        assert!(flag_table, "table_hat wrong in RangeProof");

        // println!("two_powers: {:?}", two_powers);
       
        // // println!("xi_from_two_powers: {:?}", xi::xi_from_challenges(&two_powers)); 

        // let table_mat = DenseMatFieldCM::<F>::from_data(vec![self.lookup.protocol_input.table[0].clone()]);
        // let hat_table_expected = table_mat.proj_lr_challenges(&table_point.0, &table_point.1);

        // // println!("table_point.0.len: {}, table_point.1.len: {}", table_point.0.len(), table_point.1.len());
        // // println!("k: {}", self.k);
        // // println!("test: {}",   (xi::xi_ip_from_challenges(&vec![F::one(); self.k], &vec![F::from(2u64); self.k]) -
        // //     xi::xi_ip_from_challenges(&vec![F::one(); self.k], &vec![F::one(); self.k]))
        // // ); 
   

        // assert_eq!(table_hat, hat_table_expected, "Table hats not correct");

        // assert_eq!(hat_table_expected, cur_ip, "Xi Computation wrong");
        // assert!(flag_table, "table_hat wrong");

        println!("✅ RangeProof reduce_prover completed successfully");
        flag_lookup && flag_table
    }

    fn verify_as_subprotocol(&mut self, trans: &mut Transcript<F>) -> bool {
        let flag_lookup = self.lookup.verify_as_subprotocol(trans);

        let target_hat = self.lookup.atomic_pop.target_hats[0];
        let target_point = self.lookup.atomic_pop.target_points[0].clone();
        let table_hat = self.lookup.atomic_pop.table_hats[0];
        let table_point = self.lookup.atomic_pop.table_points[0].clone();
        let auxiliary_target_hat = self.lookup.atomic_pop.auxiliary_target_hat;
        let auxiliary_target_point = self.lookup.atomic_pop.auxiliary_target_points.clone();
        let auxiliary_table_hat = self.lookup.atomic_pop.auxiliary_table_hat;
        let auxiliary_table_point = self.lookup.atomic_pop.auxiliary_table_points.clone();

        let target_hat_index = self.lookup.atomic_pop.mapping.target_hats_index[0];
        let target_point_index = self.lookup.atomic_pop.mapping.target_points_index[0].clone();
        let table_hat_index = self.lookup.atomic_pop.mapping.table_hats_index[0];
        let table_point_index = self.lookup.atomic_pop.mapping.table_points_index[0].clone();
        let auxiliary_target_hat_index = self.lookup.atomic_pop.mapping.auxiliary_target_hat_index;
        let auxiliary_target_point_index = self.lookup.atomic_pop.mapping.auxiliary_target_points_index.clone();
        let auxiliary_table_hat_index = self.lookup.atomic_pop.mapping.auxiliary_table_hat_index;
        let auxiliary_table_point_index = self.lookup.atomic_pop.mapping.auxiliary_table_points_index.clone();



        let table_point_clone = table_point.clone();
        self.atomic_pop.set_pop_trans(
            target_hat,
            target_point,
            table_hat,
            table_point_clone,
            auxiliary_target_hat,
            auxiliary_target_point,
            auxiliary_table_hat,
            auxiliary_table_point,
            target_hat_index,
            target_point_index,
            table_hat_index,
            table_point_index,
            auxiliary_target_hat_index,
            auxiliary_target_point_index,
            auxiliary_table_hat_index,
            auxiliary_table_point_index,
        );

        let two_powers = (0..self.k).map(|i| F::from(2u64).pow(&[i as u64])).collect::<Vec<F>>();
        let xx = table_point.0.clone().into_iter().rev().collect::<Vec<_>>();

        let mut cur_ip = F::zero();
        let mut cur_mul = F::one();

        for i in 0..self.k {
            cur_ip = cur_ip.clone() * (F::one() + &xx[i]) + cur_mul.clone() * &two_powers[i] * &xx[i];
            cur_mul =  cur_mul * (F::one() + &xx[i]);
        }
        let flag_table = table_hat == cur_ip;

        assert!(flag_table, "Table Projection Not Satisified");

        println!("✅ RangeProof verify_as_subprotocol completed successfully");
        flag_lookup && flag_table
    }


    fn prepare_atomic_pop(&mut self) -> bool {
        let flag_lookup = self.lookup.prepare_atomic_pop();
        if !flag_lookup { return false; }

        
        let table_hat_expr = ArithmeticExpression::input(self.atomic_pop.mapping.table_hat_index);
        let table_point_expr: Vec<ArithmeticExpression<F>> = self.atomic_pop.mapping.table_point_index.0
        .clone().into_iter().map(|i| ArithmeticExpression::input(i)).collect();

        let two_powers = (0..self.k).rev().map(|i| F::from(2u64).pow(&[i as u64])).collect::<Vec<F>>();
        let two_powers_expr = (two_powers.iter().map(|&x| ArithmeticExpression::constant(x))).collect::<Vec<ArithmeticExpression<F>>>();

        let mut cur_ip_expr = ArithmeticExpression::constant(F::zero());
        let mut cur_mul = ArithmeticExpression::constant(F::one());

        for i in 0..self.k {
            cur_ip_expr = ArithmeticExpression::add(
                ArithmeticExpression::mul(
                    cur_ip_expr.clone(),
                    ArithmeticExpression::add(
                        ArithmeticExpression::constant(F::one()),
                        table_point_expr[i].clone()
                    )
                ),
                ArithmeticExpression::mul(
                    cur_mul.clone(),
                    ArithmeticExpression::mul(two_powers_expr[i].clone(), table_point_expr[i].clone())
                )
            );

            cur_mul = ArithmeticExpression::mul(
                cur_mul,
                ArithmeticExpression::add(
                    ArithmeticExpression::constant(F::one()),
                    table_point_expr[i].clone()
                )
            );
        }

        let check = ArithmeticExpression::sub(
            cur_ip_expr,
            table_hat_expr
        );

        self.atomic_pop.set_check(check);
        self.atomic_pop.set_links(vec![]);
        self.atomic_pop.is_ready()
    }

    fn synthesize_atomic_pop_constraints(&self, cs_builder: &mut ConstraintSystemBuilder<F>) -> bool {
        let flag_lookup = self.lookup.synthesize_atomic_pop_constraints(cs_builder);
        if !flag_lookup { return false; }
        if !self.atomic_pop.is_ready() {
            println!("!!!!!!!!!!!!!!!!!! Atomic pop is not ready in RangeProof when synthesizing constraints");
            return false;
        }
        cs_builder.add_constraint(self.atomic_pop.check.clone());
        for constraint in &self.atomic_pop.link_inputs { cs_builder.add_constraint(constraint.clone()); }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;
    use fsproof::Transcript;
    use mat::MyInt;

    #[test]
    fn test_rangeproof_roundtrip_and_constraints() {
        let k = 4usize; // range 0..15
        let target_len = 8usize; // small vector
        let mut rng = StdRng::seed_from_u64(20250820);
        
        // Sample integer targets in range
        let target_int: Vec<MyInt> = (0..target_len).map(|_| rng.random_range(0..(1<<k) as MyInt)).collect();
        // Build auxiliaries via builder
        let (aux_target_int, aux_table_int) = crate::utils::table_builder::range_auxiliary_builder(&target_int, k);
        // Convert to field
        let target: Vec<Fr> = target_int.iter().map(|&v| Fr::from(v as u64)).collect();
        let auxiliary_target: Vec<Fr> = aux_target_int.iter().map(|&v| Fr::from(v as u64)).collect();
        let auxiliary_table: Vec<Fr> = aux_table_int.iter().map(|&v| Fr::from(v as u64)).collect();

        let mut proto = RangeProof::<Fr>::new(k, target_len);
        proto.set_input(target.clone(), auxiliary_target.clone(), auxiliary_table.clone());

        // Prover side
        let mut prover_trans = Transcript::new(Fr::zero());
        assert!(proto.reduce_prover(&mut prover_trans), "RangeProof reduce_prover failed");

        // Verifier side
        let mut verifier_trans = prover_trans.clone();
        verifier_trans.reset_pointer();
        let mut v_proto = proto.clone();
        assert!(v_proto.verify_as_subprotocol(&mut verifier_trans), "RangeProof verify_as_subprotocol failed");
        assert_eq!(prover_trans.pointer, verifier_trans.pointer, "Transcript pointer mismatch for RangeProof");

        // Additionally: synthesize Range constraints using the new unified indexing scheme
        // All transcript values are treated as private; no public inputs in Range itself.
        use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder as CSB;
        let mut cs_builder = CSB::<Fr>::new();
        assert!(proto.prepare_atomic_pop(), "RangeProof prepare_atomic_pop failed");
        let pri_inputs = prover_trans.get_trans_seq();
        cs_builder.set_private_inputs(pri_inputs);
        cs_builder.set_public_inputs(Vec::new());
        assert!(proto.synthesize_atomic_pop_constraints(&mut cs_builder), "RangeProof synthesize_atomic_pop_constraints failed");
        // Validate constraints under unified inputs [pri..., pub(=empty)...]
        if let Err(e) = cs_builder.validate_constraints() {
            panic!("RangeProof constraints validation failed: {}", e);
        }
    }

    #[test]
    fn test_rangeproof_with_batchproj_auxiliary() {
        use atomic_proof::protocols::batchproj::BatchProj;
        use atomic_proof::AtomicMatProtocol as _; // trait for prepare / synthesize
        use mat::utils::matdef::DenseMatCM;
        use mat::MyInt;
        use atomic_proof::pop::arithmetic_expression::ConstraintSystemBuilder as CSB2;

        let k = 4usize; // 0..15
        let target_len = 8usize;
        let mut rng = StdRng::seed_from_u64(20250820);
        // Integer target then build auxiliaries
        let target_int: Vec<MyInt> = (0..target_len).map(|_| rng.random_range(0..(1<<k) as MyInt)).collect();
        let (aux_target_int, aux_table_int) = crate::utils::table_builder::range_auxiliary_builder(&target_int, k);
        let target: Vec<Fr> = target_int.iter().map(|&v| Fr::from(v as u64)).collect();
        let auxiliary_target: Vec<Fr> = aux_target_int.iter().map(|&v| Fr::from(v as u64)).collect();
        let auxiliary_table: Vec<Fr> = aux_table_int.iter().map(|&v| Fr::from(v as u64)).collect();

        // Build range proof (lookup) first
        let mut rp = RangeProof::<Fr>::new(k, target_len);
        rp.set_input(target.clone(), auxiliary_target.clone(), auxiliary_table.clone());
        let mut trans = Transcript::new(Fr::zero());
        assert!(rp.reduce_prover(&mut trans), "RangeProof reduce_prover failed (combo)");

        // Reuse auxiliary projections (target/table) to build BatchProj (no new transcript pushes)
        let hat_indices = vec![
            rp.lookup.atomic_pop.mapping.auxiliary_target_hat_index,
            rp.lookup.atomic_pop.mapping.auxiliary_table_hat_index,
        ];
        let point_indices = vec![
            rp.lookup.atomic_pop.mapping.auxiliary_target_points_index.clone(),
            rp.lookup.atomic_pop.mapping.auxiliary_table_points_index.clone(),
        ];
        let hats = vec![
            rp.lookup.atomic_pop.auxiliary_target_hat,
            rp.lookup.atomic_pop.auxiliary_table_hat,
        ];
        let points = vec![
            rp.lookup.atomic_pop.auxiliary_target_points.clone(),
            rp.lookup.atomic_pop.auxiliary_table_points.clone(),
        ];

        // Prepare RangeProof atomic PoP (was missing, caused atomic_pop not ready during synth)
        assert!(rp.prepare_atomic_pop(), "RangeProof prepare_atomic_pop failed in combo");


        // Construct matrices consistent with those projections: (len,1) matrices of original integer counts
        let mut mats: Vec<DenseMatCM<MyInt, Fr>> = Vec::new();
        // target auxiliary matrix
        let mut mat_t = DenseMatCM::<MyInt, Fr>::new(target_len, 1);
        let mut col_t: Vec<MyInt> = Vec::with_capacity(target_len);
        for _ in 0..target_len { col_t.push(1 as MyInt); } // all ones like auxiliary_target
        mat_t.set_data(vec![col_t]);
        mats.push(mat_t);
        // table auxiliary matrix
        let table_len = 1<<k;
        let mut mat_tab = DenseMatCM::<MyInt, Fr>::new(table_len, 1);
        let mut col_tab: Vec<MyInt> = Vec::with_capacity(table_len);
        for _ in 0..table_len { col_tab.push(1 as MyInt); }
        mat_tab.set_data(vec![col_tab]);
        mats.push(mat_tab);

        let mut batchproj = BatchProj::new(hats.clone(), points.clone(), hat_indices.clone(), point_indices.clone());
        batchproj.set_input(mats);

        assert!(batchproj.reduce_prover(&mut trans), "BatchProj reduce_prover failed (range combo)");
        assert!(batchproj.prepare_atomic_pop(), "BatchProj prepare_atomic_pop failed (range combo)");

        // Combined constraint system
        let mut cs_builder = CSB2::<Fr>::new();
        let all_inputs = trans.get_trans_seq();
        // public inputs: final exposure of batchproj
        let mapping_bp = &batchproj.atomic_pop.mapping;
        let mut max_pub = mapping_bp.final_c_hat_index;
        for &i in mapping_bp.final_c_point_index.0.iter() { if i > max_pub { max_pub = i; } }
        for &i in mapping_bp.final_c_point_index.1.iter() { if i > max_pub { max_pub = i; } }
        let mut pub_inputs = vec![Fr::zero(); max_pub+1];
        pub_inputs[mapping_bp.final_c_hat_index] = trans.get_at_position(mapping_bp.final_c_hat_index);
        for &i in mapping_bp.final_c_point_index.0.iter() { pub_inputs[i] = trans.get_at_position(i); }
        for &i in mapping_bp.final_c_point_index.1.iter() { pub_inputs[i] = trans.get_at_position(i); }
        cs_builder.set_public_inputs(pub_inputs);
        cs_builder.set_private_inputs(all_inputs);

        assert!(rp.synthesize_atomic_pop_constraints(&mut cs_builder), "RangeProof synthesize failed in combo");
        let before = cs_builder.arithmetic_constraints.len();
        assert!(batchproj.synthesize_atomic_pop_constraints(&mut cs_builder), "BatchProj synthesize failed in combo");
        assert!(cs_builder.arithmetic_constraints.len() > before, "No new constraints added by BatchProj in combo test");
    }
}