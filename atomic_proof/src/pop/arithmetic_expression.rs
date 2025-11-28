use ark_ff::PrimeField;
use ark_r1cs_std::prelude::*;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};

// Node of arithmetic expression
#[derive(Clone, Debug, PartialEq)]
pub enum ArithmeticExpression<F: PrimeField> {
    // Constant node
    Constant(F),
    // Private input (witness) variable, indexed by position in the private input vector
    PriInput(usize),
    // Public input (instance) variable, indexed by position in the public input vector
    PubInput(usize),
    // Addition node, contains two sub-expressions
    Add {
        left: Box<ArithmeticExpression<F>>,
        right: Box<ArithmeticExpression<F>>,
    },
    // Subtraction node, contains two sub-expressions
    Sub {
        left: Box<ArithmeticExpression<F>>,
        right: Box<ArithmeticExpression<F>>,
    },
    // Multiplication node, contains two sub-expressions
    Mul {
        left: Box<ArithmeticExpression<F>>,
        right: Box<ArithmeticExpression<F>>,
    },
    // Inverse node, contains one sub-expression
    // Constraint: inv_var * inner_expr = 1
    Inv {
        inner: Box<ArithmeticExpression<F>>,
    },
}

// Constraint system builder - core component
// Maintains disjoint public (instance) and private (witness) inputs
#[derive(Clone, Debug)]
pub struct ConstraintSystemBuilder<F: PrimeField> {
    pub num_pub_inputs: usize,
    pub num_pri_inputs: usize,
    pub arithmetic_constraints: Vec<ArithmeticExpression<F>>,
    pub pub_inputs: Vec<F>,
    pub pri_inputs: Vec<F>,
}

impl<F: PrimeField> ConstraintSystemBuilder<F> {
    // Create a new constraint system builder
    pub fn new() -> Self {
        Self {
            num_pub_inputs: 0,
            num_pri_inputs: 0,
            arithmetic_constraints: Vec::new(),
            pub_inputs: Vec::new(),
            pri_inputs: Vec::new(),
        }
    }

    // Backwards-compatible helper: treat supplied vector as public inputs
    pub fn set_inputs(&mut self, pub_inputs: Vec<F>) -> &mut Self {
        self.set_public_inputs(pub_inputs)
    }

    pub fn set_public_inputs(&mut self, pub_inputs: Vec<F>) -> &mut Self {
        self.num_pub_inputs = pub_inputs.len();
        self.pub_inputs = pub_inputs;
        self
    }

    pub fn set_private_inputs(&mut self, pri_inputs: Vec<F>) -> &mut Self {
        self.num_pri_inputs = pri_inputs.len();
        self.pri_inputs = pri_inputs;
        self
    }

    // Add an arithmetic constraint: expression = 0
    pub fn add_constraint(&mut self, expr: ArithmeticExpression<F>) -> &mut Self {
        self.arithmetic_constraints.push(expr);
        self
    }

    // Add an equality constraint: left = right, converted to left - right = 0
    pub fn add_equal_constraint(
        &mut self,
        left: ArithmeticExpression<F>,
        right: ArithmeticExpression<F>,
    ) -> &mut Self {
        let zero_expr = ArithmeticExpression::sub(left, right);
        self.add_constraint(zero_expr)
    }

    
    // Add vector equality constraints: vec_a[i] = vec_b[i] for all i
    // Each element constraint is: vec_a[i] - vec_b[i] = 0
    pub fn add_equal_vec_constraints(
        &mut self,
        vec_a: Vec<ArithmeticExpression<F>>,
        vec_b: Vec<ArithmeticExpression<F>>,
    ) -> Result<&mut Self, String> {
        if vec_a.len() != vec_b.len() {
            return Err(format!(
                "Vector length mismatch: vec_a has {} elements, vec_b has {} elements",
                vec_a.len(),
                vec_b.len()
            ));
        }

        for (_, (expr_a, expr_b)) in vec_a.into_iter().zip(vec_b.into_iter()).enumerate() {
            // Add constraint: expr_a - expr_b = 0
            let constraint = expr_a - expr_b;
            self.add_constraint(constraint);
            
            // println!("Added vector equality constraint {}: expr_a[{}] = expr_b[{}]", i, i, i);
        }

        Ok(self)
    }

    // Add vector element-wise multiplication constraint: vec_a[i] * vec_b[i] = vec_c[i] for all i
    // Each element constraint is: vec_a[i] * vec_b[i] - vec_c[i] = 0
    pub fn add_mul_vec_constraint(
        &mut self,
        vec_a: Vec<ArithmeticExpression<F>>,
        vec_b: Vec<ArithmeticExpression<F>>,
        vec_c: Vec<ArithmeticExpression<F>>,
    ) -> Result<&mut Self, String> {
        if vec_a.len() != vec_b.len() || vec_a.len() != vec_c.len() {
            return Err(format!(
                "Vector length mismatch: vec_a has {} elements, vec_b has {} elements, vec_c has {} elements",
                vec_a.len(),
                vec_b.len(),
                vec_c.len()
            ));
        }

    for (_, ((expr_a, expr_b), expr_c)) in vec_a.into_iter()
            .zip(vec_b.into_iter())
            .zip(vec_c.into_iter())
            .enumerate() {
            
            // Add constraint: expr_a * expr_b - expr_c = 0
            let constraint = expr_a * expr_b - expr_c;
            self.add_constraint(constraint);
            
            // println!("Added vector multiplication constraint {}: expr_a[{}] * expr_b[{}] = expr_c[{}]", i, i, i, i);
        }

        Ok(self)
    }

    // Add vector inverse constraint: vec_a[i] * vec_b[i] = 1 for all i (vec_b is inverse of vec_a)
    // Each element constraint is: vec_a[i] * vec_b[i] - 1 = 0
    pub fn add_inv_vec_constraint(
        &mut self,
        vec_a: Vec<ArithmeticExpression<F>>,
        vec_b: Vec<ArithmeticExpression<F>>,
    ) -> Result<&mut Self, String> {
        if vec_a.len() != vec_b.len() {
            return Err(format!(
                "Vector length mismatch: vec_a has {} elements, vec_b has {} elements",
                vec_a.len(),
                vec_b.len()
            ));
        }

    for (_, (expr_a, expr_b)) in vec_a.into_iter().zip(vec_b.into_iter()).enumerate() {
            // Add constraint: expr_a * expr_b - 1 = 0
            let constraint = expr_a * expr_b - ArithmeticExpression::Constant(F::one());
            self.add_constraint(constraint);
            
            // println!("Added vector inverse constraint {}: expr_a[{}] * expr_b[{}] = 1", i, i, i);
        }

        Ok(self)
    }

    // Validate all constraints against the provided inputs
    // If any constraint is violated, an error is returned
    pub fn validate_constraints(&self) -> Result<(), String> {
        let public_input_start_index = self.num_pri_inputs;
        let mut combined_inputs = self.pri_inputs.clone();
        combined_inputs.extend_from_slice(&self.pub_inputs);

        for (i, constraint) in self.arithmetic_constraints.iter().enumerate() {
            let value = constraint
                .evaluate_combined(&combined_inputs, public_input_start_index)
                .map_err(|e| format!("Constraint {}: Failed to evaluate expression: {}", i, e))?;
            if value != F::zero() {
                return Err(format!(
                    "Constraint {} violated: expression evaluates to {:?}, expected 0",
                    i, value
                ));
            }
        }
        Ok(())
    }

    // Print a summary of the constraints
    pub fn print_summary(&self) {
        println!("=== Constraint System Summary ===");
        println!("Public inputs length: {:?}", self.num_pub_inputs);
        println!("Private inputs length: {:?}", self.num_pri_inputs);
        println!("Arithmetic constraints count: {}", self.arithmetic_constraints.len());
        
        for (i, constraint) in self.arithmetic_constraints.iter().enumerate() {
            println!("Constraint {}: {:?} = 0", i, constraint);
        }
    }
}

// Synthesize expression constraints into R1CS
impl<F: PrimeField> ConstraintSynthesizer<F> for ConstraintSystemBuilder<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // Allow circuits that have only private inputs (no public inputs)
        if self.num_pub_inputs == 0 && self.num_pri_inputs == 0 {
            return Err(SynthesisError::Unsatisfiable);
        }

        println!("=== Generating R1CS Constraints ===");

        // Build a single input vector: [private (witness)... | public (instance)...]
        let public_input_start_index = self.num_pri_inputs;
        let mut input_vars: Vec<FpVar<F>> = Vec::with_capacity(self.num_pri_inputs + self.num_pub_inputs);
        // Private inputs as witnesses first
        for &val in self.pri_inputs.iter() {
            input_vars.push(FpVar::new_witness(cs.clone(), || Ok(val))?);
        }
        // Public inputs as instance variables afterwards
        for &val in self.pub_inputs.iter() {
            input_vars.push(FpVar::new_input(cs.clone(), || Ok(val))?);
        }

        // Prepare combined concrete values in the same order [pri..., pub...]
        let combined_inputs: Option<Vec<F>> = if self.num_pri_inputs + self.num_pub_inputs > 0 {
            let mut vec = self.pri_inputs.clone();
            vec.extend_from_slice(&self.pub_inputs);
            Some(vec)
        } else {
            None
        };
        let combined_inputs_ref = combined_inputs.as_deref();

        // Generate R1CS constraints for each arithmetic constraint
        for constraint in self.arithmetic_constraints.iter() {
            let expr_var = synthesize_expression_constraints(
                constraint,
                &input_vars,
                combined_inputs_ref,
                public_input_start_index,
                cs.clone(),
            )?;
            expr_var.enforce_equal(&FpVar::Constant(F::zero()))?;
        }

        println!("R1CS constraints generated successfully!");
        println!(
            "Total constraints: {}",
            cs.num_constraints()
        );
        println!(
            "Total instance variables: {}",
            cs.num_instance_variables()
        );
        println!(
            "Total witness variables: {}",
            cs.num_witness_variables()
        );
        
        Ok(())
    }
}

// ArithmeticExpression methods
impl<F: PrimeField> ArithmeticExpression<F> {
    pub fn constant(value: F) -> Self {
        ArithmeticExpression::Constant(value)
    }
    
    pub fn input(index: usize) -> Self { // legacy alias -> public input
        ArithmeticExpression::PriInput(index)
    }


    pub fn pub_input(index: usize) -> Self { Self::PubInput(index) }
    pub fn pri_input(index: usize) -> Self { Self::PriInput(index) }
    
    pub fn add(left: ArithmeticExpression<F>, right: ArithmeticExpression<F>) -> Self {
        ArithmeticExpression::Add {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
    
    pub fn sub(left: ArithmeticExpression<F>, right: ArithmeticExpression<F>) -> Self {
        ArithmeticExpression::Sub {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
    
    pub fn mul(left: ArithmeticExpression<F>, right: ArithmeticExpression<F>) -> Self {
        ArithmeticExpression::Mul {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    // Create inverse expression: 1/expr
    // Note: This will create a constraint that inv_var * expr = 1
    pub fn inv(inner: ArithmeticExpression<F>) -> Self {
        ArithmeticExpression::Inv {
            inner: Box::new(inner),
        }
    }

    // Create an inner product expression between two vectors
    // This computes <left_vec, right_vec> = sum(left_vec[i] * right_vec[i])
    pub fn inner_product(left_vec: &[ArithmeticExpression<F>], right_vec: &[ArithmeticExpression<F>]) -> Self {
        if left_vec.is_empty() || right_vec.is_empty() {
            return ArithmeticExpression::Constant(F::zero());
        }

        if left_vec.len() != right_vec.len() {
            panic!(
                "Vector length mismatch in inner_product: {} vs {}",
                left_vec.len(),
                right_vec.len()
            );
        }

        // Use a balanced recursion so stack depth grows O(log n) instead of O(n).
        fn balanced_inner_product<F: PrimeField>(
            left: &[ArithmeticExpression<F>],
            right: &[ArithmeticExpression<F>],
        ) -> ArithmeticExpression<F> {
            match left.len() {
                0 => ArithmeticExpression::Constant(F::zero()),
                1 => ArithmeticExpression::mul(left[0].clone(), right[0].clone()),
                len => {
                    let mid = len / 2;
                    let left_expr = balanced_inner_product(&left[..mid], &right[..mid]);
                    let right_expr = balanced_inner_product(&left[mid..], &right[mid..]);
                    ArithmeticExpression::add(left_expr, right_expr)
                }
            }
        }

        balanced_inner_product(left_vec, right_vec)
    }

    // Create an expression representing a hat variable (transcript variable)
    // This is used for variables that come from the transcript in atomic proofs
    pub fn x_hat(index: usize) -> Self { // treat transcript hats as public for now
        ArithmeticExpression::PubInput(index)
    }

    pub fn evaluate(&self, pub_inputs: &[F], pri_inputs: &[F]) -> Result<F, String> {
        match self {
            ArithmeticExpression::Constant(value) => Ok(*value),
            ArithmeticExpression::PubInput(index) => {
                if let Some(v) = pub_inputs.get(*index) { return Ok(*v); }
                // Fallback: 兼容旧路径——如果 public 不足但 private 中有同索引，则取 private。
                // 这样其它子协议仍然可以把 transcript 整体放入 pri_inputs，而表达式用 PubInput 索引不需要重写。
                if let Some(v) = pri_inputs.get(*index) { return Ok(*v); }
                Err(format!("Public input index {} out of bounds", index))
            }
            ArithmeticExpression::PriInput(index) => pri_inputs
                .get(*index)
                .copied()
                .ok_or_else(|| format!("Private input index {} out of bounds", index)),
            ArithmeticExpression::Add { left, right } => {
                let left_val = left.evaluate(pub_inputs, pri_inputs)?;
                let right_val = right.evaluate(pub_inputs, pri_inputs)?;
                Ok(left_val + right_val)
            }
            ArithmeticExpression::Sub { left, right } => {
                let left_val = left.evaluate(pub_inputs, pri_inputs)?;
                let right_val = right.evaluate(pub_inputs, pri_inputs)?;
                Ok(left_val - right_val)
            }
            ArithmeticExpression::Mul { left, right } => {
                let left_val = left.evaluate(pub_inputs, pri_inputs)?;
                let right_val = right.evaluate(pub_inputs, pri_inputs)?;
                Ok(left_val * right_val)
            }
            ArithmeticExpression::Inv { inner } => {
                let inner_val = inner.evaluate(pub_inputs, pri_inputs)?;
                if inner_val == F::zero() {
                    return Err("Cannot compute inverse of zero".to_string());
                }
                
                let inverse = inner_val.inverse()
                    .ok_or_else(|| "Failed to compute inverse".to_string())?;
                Ok(inverse)
            }
        }
    }
    
    // Collect all unique input variable indices used in the expression
    pub fn input_indices(&self) -> (Vec<usize>, Vec<usize>) {
        let mut pub_indices = Vec::new();
        let mut pri_indices = Vec::new();
        self.collect_input_indices(&mut pub_indices, &mut pri_indices);
        pub_indices.sort_unstable(); pub_indices.dedup();
        pri_indices.sort_unstable(); pri_indices.dedup();
        (pub_indices, pri_indices)
    }

    fn collect_input_indices(&self, pub_indices: &mut Vec<usize>, pri_indices: &mut Vec<usize>) {
        match self {
            ArithmeticExpression::PubInput(i) => pub_indices.push(*i),
            ArithmeticExpression::PriInput(i) => pri_indices.push(*i),
            ArithmeticExpression::Add { left, right }
            | ArithmeticExpression::Sub { left, right }
            | ArithmeticExpression::Mul { left, right } => {
                left.collect_input_indices(pub_indices, pri_indices);
                right.collect_input_indices(pub_indices, pri_indices);
            }
            ArithmeticExpression::Inv { inner } => inner.collect_input_indices(pub_indices, pri_indices),
            ArithmeticExpression::Constant(_) => {}
        }
    }

    // Convenient method - input variable
    // Creates a new input variable
    pub fn x(index: usize) -> Self { // legacy alias -> public input
        ArithmeticExpression::PubInput(index)
    }
    pub fn x_pri(index: usize) -> Self { ArithmeticExpression::PriInput(index) }
    pub fn x_pub(index: usize) -> Self { ArithmeticExpression::PubInput(index) }

    // Convenient method - constant
    // Creates a new constant
    pub fn c(value: F) -> Self {
        ArithmeticExpression::Constant(value)
    }

    // Creates a linear combination expression: c0*x0 + c1*x1 + ... + cn*xn
    pub fn linear_combination(coefficients: Vec<(usize, F)>) -> Self {
        if coefficients.is_empty() {
            return ArithmeticExpression::constant(F::zero());
        }

        let mut expr = ArithmeticExpression::constant(F::zero());
        
        for (input_index, coeff) in coefficients {
            let term = if coeff == F::one() {
                ArithmeticExpression::pub_input(input_index)
            } else {
                ArithmeticExpression::mul(
                    ArithmeticExpression::constant(coeff),
                    ArithmeticExpression::pub_input(input_index),
                )
            };
            
            expr = if expr == ArithmeticExpression::constant(F::zero()) {
                term
            } else {
                ArithmeticExpression::add(expr, term)
            };
        }

        expr
    }
}

// Operator overloading
use std::ops::{Add, Sub, Mul};

impl<F: PrimeField> Add for ArithmeticExpression<F> {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Add {
            left: Box::new(self),
            right: Box::new(rhs),
        }
    }
}

impl<F: PrimeField> Sub for ArithmeticExpression<F> {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Sub {
            left: Box::new(self),
            right: Box::new(rhs),
        }
    }
}

impl<F: PrimeField> Mul for ArithmeticExpression<F> {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Mul {
            left: Box::new(self),
            right: Box::new(rhs),
        }
    }
}

// Support operations with constants
impl<F: PrimeField> Add<F> for ArithmeticExpression<F> {
    type Output = Self;
    
    fn add(self, rhs: F) -> Self::Output {
        ArithmeticExpression::Add {
            left: Box::new(self),
            right: Box::new(ArithmeticExpression::Constant(rhs)),
        }
    }
}

impl<F: PrimeField> Sub<F> for ArithmeticExpression<F> {
    type Output = Self;
    
    fn sub(self, rhs: F) -> Self::Output {
        ArithmeticExpression::Sub {
            left: Box::new(self),
            right: Box::new(ArithmeticExpression::Constant(rhs)),
        }
    }
}

impl<F: PrimeField> Mul<F> for ArithmeticExpression<F> {
    type Output = Self;
    
    fn mul(self, rhs: F) -> Self::Output {
        ArithmeticExpression::Mul {
            left: Box::new(self),
            right: Box::new(ArithmeticExpression::Constant(rhs)),
        }
    }
}

// Helpers for synthesizing expression constraints into R1CS
// Public inputs are used as input variables, and intermediate computation results are treated as witnesses.
fn synthesize_expression_constraints<F: PrimeField>(
    expr: &ArithmeticExpression<F>,
    input_vars: &[FpVar<F>],                 // unified input vars: [pri..., pub...]
    inputs: Option<&[F]>,                    // unified concrete values: [pri..., pub...]
    public_input_start_index: usize,          // starting index of public segment in unified vectors
    cs: ConstraintSystemRef<F>,
) -> Result<FpVar<F>, SynthesisError> {
    enum StackEntry<'a, F: PrimeField> {
        Enter(&'a ArithmeticExpression<F>),
        Exit(&'a ArithmeticExpression<F>),
    }

    let mut stack = vec![StackEntry::Enter(expr)];
    let mut results: Vec<(FpVar<F>, Option<F>)> = Vec::new();

    while let Some(entry) = stack.pop() {
        match entry {
            StackEntry::Enter(node) => match node {
                ArithmeticExpression::Constant(value) => {
                    results.push((FpVar::Constant(*value), Some(*value)));
                }
                ArithmeticExpression::PubInput(index) => {
                    let var = input_vars
                        .get(public_input_start_index + *index)
                        .cloned()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let value = if let Some(values) = inputs {
                        Some(
                            values
                                .get(public_input_start_index + *index)
                                .copied()
                                .ok_or(SynthesisError::AssignmentMissing)?,
                        )
                    } else {
                        None
                    };
                    results.push((var, value));
                }
                ArithmeticExpression::PriInput(index) => {
                    let var = input_vars
                        .get(*index)
                        .cloned()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let value = if let Some(values) = inputs {
                        Some(
                            values
                                .get(*index)
                                .copied()
                                .ok_or(SynthesisError::AssignmentMissing)?,
                        )
                    } else {
                        None
                    };
                    results.push((var, value));
                }
                ArithmeticExpression::Add { left, right }
                | ArithmeticExpression::Sub { left, right }
                | ArithmeticExpression::Mul { left, right } => {
                    stack.push(StackEntry::Exit(node));
                    stack.push(StackEntry::Enter(right));
                    stack.push(StackEntry::Enter(left));
                }
                ArithmeticExpression::Inv { inner } => {
                    stack.push(StackEntry::Exit(node));
                    stack.push(StackEntry::Enter(inner));
                }
            },
            StackEntry::Exit(node) => match node {
                ArithmeticExpression::Add { .. } => {
                    let (right_var, right_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let (left_var, left_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let expected = match (left_val, right_val) {
                        (Some(l), Some(r)) => Some(l + r),
                        _ => None,
                    };
                    let assignment = expected.unwrap_or_else(|| F::zero());
                    let sum_var = FpVar::new_witness(cs.clone(), || Ok(assignment))?;
                    let computed_sum = &left_var + &right_var;
                    sum_var.enforce_equal(&computed_sum)?;
                    results.push((sum_var, expected));
                }
                ArithmeticExpression::Sub { .. } => {
                    let (right_var, right_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let (left_var, left_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let expected = match (left_val, right_val) {
                        (Some(l), Some(r)) => Some(l - r),
                        _ => None,
                    };
                    let assignment = expected.unwrap_or_else(|| F::zero());
                    let diff_var = FpVar::new_witness(cs.clone(), || Ok(assignment))?;
                    let computed_diff = &left_var - &right_var;
                    diff_var.enforce_equal(&computed_diff)?;
                    results.push((diff_var, expected));
                }
                ArithmeticExpression::Mul { .. } => {
                    let (right_var, right_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let (left_var, left_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let expected = match (left_val, right_val) {
                        (Some(l), Some(r)) => Some(l * r),
                        _ => None,
                    };
                    let assignment = expected.unwrap_or_else(|| F::zero());
                    let product_var = FpVar::new_witness(cs.clone(), || Ok(assignment))?;
                    let computed_product = &left_var * &right_var;
                    product_var.enforce_equal(&computed_product)?;
                    results.push((product_var, expected));
                }
                ArithmeticExpression::Inv { .. } => {
                    let (inner_var, inner_val) = results
                        .pop()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    let expected = match inner_val {
                        Some(value) => {
                            if value == F::zero() {
                                return Err(SynthesisError::DivisionByZero);
                            }
                            Some(value.inverse().ok_or(SynthesisError::DivisionByZero)?)
                        }
                        None => None,
                    };
                    let assignment = expected.unwrap_or_else(|| F::one());
                    let inverse_var = FpVar::new_witness(cs.clone(), || Ok(assignment))?;
                    let product = &inverse_var * &inner_var;
                    let one_var = FpVar::Constant(F::one());
                    product.enforce_equal(&one_var)?;
                    results.push((inverse_var, expected));
                }
                _ => unreachable!("Exit entries are only pushed for composite expressions"),
            },
        }
    }

    if results.len() != 1 {
        return Err(SynthesisError::Unsatisfiable);
    }

    Ok(results.pop().unwrap().0)
}

impl<F: PrimeField> ArithmeticExpression<F> {
    // Evaluate using unified inputs: inputs = [pri..., pub...]
    pub fn evaluate_combined(&self, inputs: &[F], public_input_start_index: usize) -> Result<F, String> {
        match self {
            ArithmeticExpression::Constant(value) => Ok(*value),
            ArithmeticExpression::PubInput(index) => inputs
                .get(public_input_start_index + *index)
                .copied()
                .ok_or_else(|| format!("Public input index {} (global {}) out of bounds", index, public_input_start_index + *index)),
            ArithmeticExpression::PriInput(index) => inputs
                .get(*index)
                .copied()
                .ok_or_else(|| format!("Private input index {} out of bounds", index)),
            ArithmeticExpression::Add { left, right } => Ok(left.evaluate_combined(inputs, public_input_start_index)? + right.evaluate_combined(inputs, public_input_start_index)?),
            ArithmeticExpression::Sub { left, right } => Ok(left.evaluate_combined(inputs, public_input_start_index)? - right.evaluate_combined(inputs, public_input_start_index)?),
            ArithmeticExpression::Mul { left, right } => Ok(left.evaluate_combined(inputs, public_input_start_index)? * right.evaluate_combined(inputs, public_input_start_index)?),
            ArithmeticExpression::Inv { inner } => {
                let v = inner.evaluate_combined(inputs, public_input_start_index)?;
                if v == F::zero() { return Err("Cannot compute inverse of zero".to_string()); }
                Ok(v.inverse().ok_or_else(|| "Failed to compute inverse".to_string())?)
            }
        }
    }
}

pub fn xi_from_challenges_exprs<F: PrimeField>(
    challenges: &Vec<ArithmeticExpression<F>>
) -> Vec<ArithmeticExpression<F>> {
    
    let mut xi_exprs = vec![ArithmeticExpression::constant(F::one())];

    // Important: xi_from_challenges applies challenges from last to first.
    // To match that order, iterate challenges in reverse when expanding the tensor.
    for challenge in challenges.iter().rev() {
        let vec_l_expr = xi_exprs.clone();
        let vec_r_expr: Vec<ArithmeticExpression<F>> = xi_exprs
            .iter()
            .map(|e| ArithmeticExpression::mul(e.clone(), challenge.clone()))
            .collect();
        xi_exprs = vec_l_expr.into_iter().chain(vec_r_expr).collect();
    }
    xi_exprs
}

pub fn xi_ip_from_challenges_exprs<F: PrimeField>(
    challenges_a: &Vec<ArithmeticExpression<F>>,
    challenges_b: &Vec<ArithmeticExpression<F>>,
) -> ArithmeticExpression<F> {

    let mut ip_expr = ArithmeticExpression::constant(F::one());

    for (a, b) in challenges_a.iter().zip(challenges_b.iter()) {
        ip_expr = ArithmeticExpression::mul(ip_expr,
            ArithmeticExpression::add(
                ArithmeticExpression::constant(F::one()),
                ArithmeticExpression::mul(a.clone(), b.clone())
            )
        );
    }

    ip_expr
}

pub fn compute_xi_at_position_expr<F: PrimeField>(position: usize, cur_log_len: usize, xxxx: &Vec<ArithmeticExpression<F>>) -> ArithmeticExpression<F> {
    // Compute the xi value at the given position using the xxxx vector
    let cur_len = (1 << cur_log_len) as usize;

    if position % cur_len != 0 {
        println!("Position {} is not aligned with current length {}", position, cur_len);
        return ArithmeticExpression::constant(F::zero());
    }

    let div = position / cur_len;

    let div_ceil = (div+1).next_power_of_two();
    let log_div_ceil = div_ceil.ilog2() as usize;

    let log_cur_len = cur_len.ilog2() as usize;

    
    let xx_div = &xxxx[xxxx.len() - log_div_ceil - log_cur_len..xxxx.len() - log_cur_len];

    let mut cur_div = div;
    let mut cur_mul = ArithmeticExpression::constant(F::one());

    for i in 0..xx_div.len() {

        if cur_div % 2 == 1 {
            cur_mul = ArithmeticExpression::mul(cur_mul, xx_div[xx_div.len()-i-1].clone());
        }

        cur_div /= 2;
    }

    cur_mul
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;
    use ark_bls12_381::Fr as BlsFr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_arithmetic_public_input_constraint_system() {
        // Test input: [3, 4, 5] as public inputs
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(4u64), BlsFr::from(5u64)];
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());

        // Add constraint: x_0 + x_1 - 7 = 0 (i.e. 3 + 4 = 7)
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(7u64)
        );

        // Validate constraints
        let result = builder.validate_constraints();
        assert!(result.is_ok());

        // Print summary (before generate_constraints)
        builder.print_summary();

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        // Check variable type assignments
        println!("Instance variables (public inputs): {}", cs.num_instance_variables());
        println!("Witness variables (intermediate results): {}", cs.num_witness_variables());

        // Should have 3 public input variables + 1 instance variable for constant 1
        assert!(cs.num_instance_variables() >= 3, "Should have at least 3 instance variables for inputs");

        // Should have witness variables for intermediate calculations
        assert!(cs.num_witness_variables() > 0, "Should have witness variables for intermediate calculations");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Constraints should be satisfied");
    }

    #[test]
    fn test_arithmetic_inverse_basic() {
        // Test basic inverse operation: 1/x_0 = inv_x_0
        // Using x_0 = 2, so 1/2 = 0.5 in field arithmetic
        let inputs = vec![BlsFr::from(2u64)];
    let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());

        // Create inverse expression: 1/x_0
        let inv_expr = ArithmeticExpression::inv(ArithmeticExpression::x(0));
        
        // Test evaluation
    let inv_result = inv_expr.evaluate(&inputs, &[]).expect("Should compute inverse");
        let expected_inverse = BlsFr::from(2u64).inverse().unwrap();
        assert_eq!(inv_result, expected_inverse);

        // Add constraint: inv(x_0) * x_0 - 1 = 0
        builder.add_constraint(
            inv_expr * ArithmeticExpression::x(0) - BlsFr::from(1u64)
        );

        // Validate constraints
    let result = builder.validate_constraints();
        assert!(result.is_ok(), "Inverse constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        println!("=== Inverse Basic Test ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // Should have witness variables for inverse result
        assert!(cs.num_witness_variables() >= 1, "Should have witness variables for inverse");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Inverse constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_inverse_complex() {
        // Test complex inverse: 1/(x_0 + x_1) where x_0 = 3, x_1 = 2
        let inputs = vec![BlsFr::from(3u64), BlsFr::from(2u64)];
    let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());

        // Create complex inverse expression: 1/(x_0 + x_1)
        let sum_expr = ArithmeticExpression::x(0) + ArithmeticExpression::x(1);
        let inv_sum_expr = ArithmeticExpression::inv(sum_expr);
        
        // Test evaluation: 1/(3 + 2) = 1/5
    let inv_result = inv_sum_expr.evaluate(&inputs, &[]).expect("Should compute inverse of sum");
        let expected_inverse = BlsFr::from(5u64).inverse().unwrap();
        assert_eq!(inv_result, expected_inverse);

        // Add constraint: inv(x_0 + x_1) * (x_0 + x_1) - 1 = 0
        let sum_expr_2 = ArithmeticExpression::x(0) + ArithmeticExpression::x(1);
        builder.add_constraint(
            inv_sum_expr * sum_expr_2 - BlsFr::from(1u64)
        );

        // Validate constraints
    let result = builder.validate_constraints();
        assert!(result.is_ok(), "Complex inverse constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        println!("=== Inverse Complex Test ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // Should have multiple witness variables: sum result, inverse result, etc.
        assert!(cs.num_witness_variables() >= 2, "Should have multiple witness variables");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Complex inverse constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_inverse_division() {
        // Test division using inverse: x_0 / x_1 = x_0 * (1/x_1)
        // Using x_0 = 6, x_1 = 2, so 6/2 = 3
        let inputs = vec![BlsFr::from(6u64), BlsFr::from(2u64)];
    let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());

        // Create division expression: x_0 * (1/x_1)
        let division_expr = ArithmeticExpression::x(0) * ArithmeticExpression::inv(ArithmeticExpression::x(1));
        
        // Test evaluation: 6 * (1/2) = 3
    let div_result = division_expr.evaluate(&inputs, &[]).expect("Should compute division");
        assert_eq!(div_result, BlsFr::from(3u64));

        // Add constraint: x_0 * inv(x_1) - 3 = 0
        builder.add_constraint(division_expr - BlsFr::from(3u64));

        // Validate constraints
    let result = builder.validate_constraints();
        assert!(result.is_ok(), "Division constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        println!("=== Inverse Division Test ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Division constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_inverse_zero_error() {
        // Test inverse of zero should fail
        let inputs = vec![BlsFr::from(0u64)];
        
        let inv_expr = ArithmeticExpression::inv(ArithmeticExpression::x(0));
        
        // Should fail to evaluate inverse of zero
    let result = inv_expr.evaluate(&inputs, &[]);
        assert!(result.is_err(), "Inverse of zero should fail");
        assert!(result.unwrap_err().contains("Cannot compute inverse of zero"));
    }

    #[test]
    fn test_arithmetic_inverse_nested() {
        // Test nested inverse: 1/(1/x_0) = x_0
        // Using x_0 = 4
        let inputs = vec![BlsFr::from(4u64)];
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());

        // Create nested inverse expression: 1/(1/x_0)
        let nested_inv_expr = ArithmeticExpression::inv(
            ArithmeticExpression::inv(ArithmeticExpression::x(0))
        );
        
        // Test evaluation: 1/(1/4) = 4
    let result = nested_inv_expr.evaluate(&inputs, &[]).expect("Should compute nested inverse");
        assert_eq!(result, BlsFr::from(4u64));

        // Add constraint: inv(inv(x_0)) - x_0 = 0
        builder.add_constraint(nested_inv_expr - ArithmeticExpression::x(0));

        // Validate constraints
    let validation_result = builder.validate_constraints();
        assert!(validation_result.is_ok(), "Nested inverse constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        println!("=== Inverse Nested Test ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // Should have multiple witness variables for nested calculations
        assert!(cs.num_witness_variables() >= 2, "Should have multiple witness variables for nested inverse");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Nested inverse constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_inverse_rational_function() {
        // Test rational function: (x_0 + x_1) / (x_2 + x_3) = (x_0 + x_1) * (1/(x_2 + x_3))
        // Using x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 1 
        // So (1 + 2) / (3 + 1) = 3 / 4
        let inputs = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(1u64)];
    let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());

        // Create rational function: (x_0 + x_1) * inv(x_2 + x_3)
        let numerator = ArithmeticExpression::x(0) + ArithmeticExpression::x(1);
        let denominator = ArithmeticExpression::x(2) + ArithmeticExpression::x(3);
        let rational_expr = numerator * ArithmeticExpression::inv(denominator);
        
        // Test evaluation: (1 + 2) * (1/(3 + 1)) = 3 * (1/4) = 3/4
    let result = rational_expr.evaluate(&inputs, &[]).expect("Should compute rational function");
        let expected = BlsFr::from(3u64) * BlsFr::from(4u64).inverse().unwrap();
        assert_eq!(result, expected);

        // Add constraint: rational_expr = 3/4
        let expected_result = BlsFr::from(3u64) * BlsFr::from(4u64).inverse().unwrap();
        builder.add_constraint(rational_expr - expected_result);

        // Validate constraints
    let validation_result = builder.validate_constraints();
        assert!(validation_result.is_ok(), "Rational function constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");

        println!("=== Inverse Rational Function Test ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // Should have multiple witness variables for complex calculations
        assert!(cs.num_witness_variables() >= 3, "Should have multiple witness variables for rational function");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Rational function constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_complex_constraint_with_public_inputs() {
        let inputs = vec! [
            BlsFr::from(2u64), // x_0 = 2 (public input)
            BlsFr::from(3u64), // x_1 = 3 (public input)
            BlsFr::from(4u64), // x_2 = 4 (public input)
        ];
        
    let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());

        // Constraint 1: x_0 + x_1 - 5 = 0 (i.e. 2 + 3 = 5)
        builder.add_constraint(
            ArithmeticExpression::x(0) + ArithmeticExpression::x(1) - BlsFr::from(5u64)
        );
        
        // Constraint 2: x_0 * x_1 - 6 = 0 (i.e. 2 * 3 = 6)
        builder.add_constraint(
            ArithmeticExpression::x(0) * ArithmeticExpression::x(1) - BlsFr::from(6u64)
        );

        // Constraint 3: (x_0 + x_1) * x_2 - 20 = 0 (i.e. (2 + 3) * 4 = 20)
        builder.add_constraint(
            (ArithmeticExpression::x(0) + ArithmeticExpression::x(1)) * ArithmeticExpression::x(2) - BlsFr::from(20u64)
        );
        
        builder.print_summary();

        // Validate all constraints
    let result = builder.validate_constraints();
        assert!(result.is_ok(), "All constraints should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        println!("=== Variable Count Analysis ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // Should have 3 public inputs + 1 constant
        assert!(cs.num_instance_variables() >= 3, "Should have public input variables");

        // Should have multiple witness variables for intermediate calculations (addition, multiplication results)
        assert!(cs.num_witness_variables() >= 3, "Should have multiple witness variables for intermediate results");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "All R1CS constraints should be satisfied");
    }

    #[test]
    fn test_arithmetic_nested_expression_witness_variables() {
        let inputs = vec! [
            BlsFr::from(1u64), // x_0 = 1
            BlsFr::from(2u64), // x_1 = 2  
            BlsFr::from(3u64), // x_2 = 3
            BlsFr::from(4u64), // x_3 = 4
        ];
            
        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(inputs.clone());

        // Complex nested expression: ((x_0 + x_1) * (x_2 + x_3)) - 21 = 0
        // Calculation: ((1 + 2) * (3 + 4)) - 21 = (3 * 7) - 21 = 21 - 21 = 0
        let complex_expr = (ArithmeticExpression::x(0) + ArithmeticExpression::x(1)) *
                          (ArithmeticExpression::x(2) + ArithmeticExpression::x(3)) -
                          BlsFr::from(21u64);
        
        builder.add_constraint(complex_expr);

        // Validate all constraints
    let result = builder.validate_constraints();
        assert!(result.is_ok(), "Complex constraint should be satisfied");

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        println!("=== Complex Expression Analysis ===");
        println!("Public inputs: {}", cs.num_instance_variables());
        println!("Witness variables: {}", cs.num_witness_variables());
        println!("Total constraints: {}", cs.num_constraints());

        // This expression should produce multiple witness variables for intermediate results:
        // - w1: x_0 + x_1 = 3
        // - w2: x_2 + x_3 = 7  
        // - w3: w1 * w2 = 21
        // - w4: w3 - 21 = 0
        assert!(cs.num_witness_variables() >= 3, "Should have multiple witness variables for nested calculations");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Complex nested constraint should be satisfied");
    }

    #[test]
    fn test_arithmetic_input_indices_method_with_inverse() {
        // Test input_indices method with inverse - Add explicit type annotations
        let expr: ArithmeticExpression<BlsFr> = ArithmeticExpression::x(0) + 
            ArithmeticExpression::inv(ArithmeticExpression::x(2) * ArithmeticExpression::x(1));
    let (pub_idx, pri_idx) = expr.input_indices();
    assert!(pri_idx.is_empty());

        // Should include input indices 0, 1, 2, sorted and deduplicated
    assert_eq!(pub_idx, vec![0, 1, 2]);

        // Test nested expression with inverse
        let nested_expr: ArithmeticExpression<BlsFr> = ArithmeticExpression::inv(
            ArithmeticExpression::x(3) + ArithmeticExpression::x(1)
        ) * (ArithmeticExpression::x(0) + ArithmeticExpression::c(BlsFr::from(5u64)));
    let (nested_pub_idx, nested_pri_idx) = nested_expr.input_indices();
    assert!(nested_pri_idx.is_empty());

        // Should include input indices 0, 1, 3
    assert_eq!(nested_pub_idx, vec![0, 1, 3]);
    }

    #[test]
    fn test_arithmetic_equal_vec_constraints() {
        // Test equal_vec: [a, b] == [c, d] => a == c && b == d
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
        
        // Add equal_vec constraint
        let result = builder.add_equal_vec_constraints(vec1, vec2);
        assert!(result.is_ok());

        // Validate constraints
        assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "equal_vec constraint should be satisfied");
        
        println!("equal_vec constraint test passed!");
    }

    #[test]
    fn test_arithmetic_equal_vec_constraints_mismatch() {
        // Test equal_vec with mismatched vector lengths
        let inputs = vec![BlsFr::from(1u64), BlsFr::from(2u64)];
        let mut builder = ConstraintSystemBuilder::<BlsFr>::new();
        builder.set_public_inputs(inputs);
        
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(0)]; // Different length
        
        let result = builder.add_equal_vec_constraints(vec1, vec2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Vector length mismatch"));
    }

    #[test]
    fn test_arithmetic_mul_vec_constraints() {
        // Test mul_vec: [a, b] * [c, d] = [e, f] => a*c = e && b*d = f
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
        
        // Add mul_vec constraint: vec1 * vec2 = result_vec
        let result = builder.add_mul_vec_constraint(vec1, vec2, result_vec);
        assert!(result.is_ok());

        // Validate constraints
        assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "mul_vec constraint should be satisfied");
        
        println!("mul_vec constraint test passed!");
    }

    #[test]
    fn test_arithmetic_mul_vec_constraints_with_constants() {
        // Test mul_vec with constants
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

    // Validate constraints
    assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "mul_vec with constants constraint should be satisfied");
        
        println!("mul_vec with constants constraint test passed!");
    }

    #[test]
    fn test_arithmetic_mul_vec_constraints_length_mismatch() {
        // Test mul_vec with mismatched vector lengths
        let inputs = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64)];
        let mut builder = ConstraintSystemBuilder::<BlsFr>::new();
        builder.set_public_inputs(inputs);
        
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(0)]; // Different length
        let vec3 = vec![ArithmeticExpression::x(2)]; // Different length
        
        let result = builder.add_mul_vec_constraint(vec1, vec2, vec3);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Vector length mismatch"));
    }

    #[test]
    fn test_arithmetic_inv_vec_constraints() {
        // Test inv_vec: for vector [a, b], its inverse vector is [1/a, 1/b]
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
        
        // Add inv_vec constraint
        let result = builder.add_inv_vec_constraint(vec, inv_vec);
        assert!(result.is_ok());

        // Validate constraints
        assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "inv_vec constraint should be satisfied");
        
        println!("inv_vec constraint test passed!");
    }

    #[test]
    fn test_arithmetic_inv_vec_constraints_single_element() {
        // Test single element inv_vec
        let a = BlsFr::from(7u64);
        let inv_a = a.inverse().unwrap(); // 1/7
        
        let inputs = vec![a, inv_a];
        let mut builder = ConstraintSystemBuilder::new();
    builder.set_public_inputs(inputs.clone());
        
        let vec = vec![ArithmeticExpression::x(0)];
        let inv_vec = vec![ArithmeticExpression::x(1)];
        
        let result = builder.add_inv_vec_constraint(vec, inv_vec);
        assert!(result.is_ok());

    // Validate constraints
    assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Single element inv_vec constraint should be satisfied");
        
        println!("Single element inv_vec constraint test passed!");
    }

    #[test]
    fn test_arithmetic_inv_vec_constraints_length_mismatch() {
        // Test inv_vec with mismatched vector lengths
        let inputs = vec![BlsFr::from(1u64), BlsFr::from(2u64)];
        let mut builder = ConstraintSystemBuilder::<BlsFr>::new();
    builder.set_public_inputs(inputs);
        
        let vec1 = vec![ArithmeticExpression::x(0), ArithmeticExpression::x(1)];
        let vec2 = vec![ArithmeticExpression::x(0)]; // Different length
        
        let result = builder.add_inv_vec_constraint(vec1, vec2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Vector length mismatch"));
    }

    #[test]
    fn test_arithmetic_combined_vec_constraints() {
        // Test combination of vector constraints
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
        
        // Add constraints:
        // 1. vec1 == vec2 (equal_vec)
        let result1 = builder.add_equal_vec_constraints(vec1.clone(), vec2.clone());
        assert!(result1.is_ok());
        
        // 2. vec1 * vec2 = mul_result (mul_vec)
        let result2 = builder.add_mul_vec_constraint(vec1, vec2, mul_result);
        assert!(result2.is_ok());

        // Validate constraints
        assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Combined vec constraints should be satisfied");
        
        println!("Combined vec constraints test passed!");
    }

    #[test]
    fn test_arithmetic_empty_vec_constraints() {
        // Test empty vector constraints (edge case)
        let inputs = vec![BlsFr::from(1u64)]; // Just a dummy input
        let mut builder = ConstraintSystemBuilder::new();
         builder.set_public_inputs(inputs.clone());
        
        let empty_vec1: Vec<ArithmeticExpression<BlsFr>> = vec![];
        let empty_vec2: Vec<ArithmeticExpression<BlsFr>> = vec![];
        let empty_vec3: Vec<ArithmeticExpression<BlsFr>> = vec![];
        
        // These should not add any constraints
        let result1 = builder.add_equal_vec_constraints(empty_vec1.clone(), empty_vec2.clone());
        assert!(result1.is_ok());
        
        let result2 = builder.add_mul_vec_constraint(empty_vec1.clone(), empty_vec2.clone(), empty_vec3.clone());
        assert!(result2.is_ok());
        
        let result3 = builder.add_inv_vec_constraint(empty_vec1, empty_vec2);
        assert!(result3.is_ok());

        // Add a simple constraint to make the system non-trivial
        builder.add_constraint(ArithmeticExpression::x(0) - BlsFr::from(1u64));

        // Validate constraints
        assert!(builder.validate_constraints().is_ok());

        // Generate R1CS constraints
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.generate_constraints(cs.clone()).expect("Failed to generate constraints");
        
        let is_satisfied = cs.is_satisfied().expect("Failed to check satisfaction");
        assert!(is_satisfied, "Empty vec constraints should be satisfied");
        
        println!("Empty vec constraints test passed!");
    }

    #[test]
    fn test_arithmetic_private_inputs_basic() {
        // Public inputs (instance): p0 = 2, p1 = 3
        let pub_inputs = vec![BlsFr::from(2u64), BlsFr::from(3u64)];
        // Private inputs (witness): w0 = 5, w1 = 1  (so w0 * w1 = 5 = p0 + p1)
        let pri_inputs = vec![BlsFr::from(5u64), BlsFr::from(1u64)];

        let mut builder = ConstraintSystemBuilder::new();
        builder.set_public_inputs(pub_inputs.clone())
               .set_private_inputs(pri_inputs.clone());

        // Constraint: w0 * w1 - (p0 + p1) = 0
        let constraint_expr = ArithmeticExpression::pri_input(0)
            * ArithmeticExpression::pri_input(1)
            - (ArithmeticExpression::pub_input(0) + ArithmeticExpression::pub_input(1));
        builder.add_constraint(constraint_expr.clone());

        // Local evaluation check using evaluate(pub, pri)
        let eval_val = (ArithmeticExpression::pri_input(0) * ArithmeticExpression::pri_input(1))
            .evaluate(&pub_inputs, &pri_inputs)
            .expect("evaluation should succeed");
        assert_eq!(eval_val, BlsFr::from(5u64));

        // Validate constraints (should be satisfied)
        assert!(builder.validate_constraints().is_ok(), "Private input constraint should hold");

        // R1CS generation
        let cs = ConstraintSystem::<BlsFr>::new_ref();
        builder.clone().generate_constraints(cs.clone()).expect("R1CS generation failed");

        // Expect: instance variables >= public inputs (+1 for ark internal 1), witness variables include 2 private + intermediates
        assert!(cs.num_instance_variables() >= pub_inputs.len(), "Should allocate public inputs");
        assert!(cs.num_witness_variables() >= pri_inputs.len(), "Should allocate private inputs as witnesses");
        let is_sat = cs.is_satisfied().expect("satisfaction check");
        assert!(is_sat, "Constraint system with private inputs must be satisfied");
    }

}