//! Mat operations for NN computation
//! 
use ark_ff::PrimeField;
use rayon::prelude::*;
use ark_std::rand::{Rng, SeedableRng};
use ark_std::rand::rngs::StdRng;

use mat::{DenseMatCM, DenseMatFieldCM};

use super::table_builder;

use crate::{MyInt, MyShortInt};


// generate a random matrix of bitwidth k (parallel implementation)
//
pub fn gen_rand_matrix<F>(rows: usize, cols: usize, bitwidth: usize) -> DenseMatCM<MyInt, F>
where
    F: PrimeField + Send + Sync,
{
    let max_value = (1 << (bitwidth - 1)) as MyInt - 1;
    let min_value = -max_value;

    // Use parallel iterator to generate columns concurrently
    let data = (0..cols)
        .into_par_iter()
        .map(|_| {
            // Each thread gets its own random RNG
            let mut rng = StdRng::from_entropy();
            (0..rows)
                .map(|_| {
                    let value = rng.gen_range(min_value..max_value);
                    MyInt::from(value)
                })
                .collect()
        })
        .collect();

    DenseMatCM::from_data(data)
}

pub fn shortint_to_myint_mat<F>(shortint_mat: &Vec<Vec<MyShortInt>>) -> DenseMatCM<MyInt, F>
where
    F: PrimeField + Send + Sync,
{
    let cols = shortint_mat.len();
    let rows = shortint_mat[0].len();

    let data = (0..cols)
    .into_par_iter()
    .map(|col_idx| {
        (0..rows)
            .map(|row_idx| {
                shortint_mat[col_idx][row_idx] as MyInt
            })
            .collect()
    })
    .collect();

    DenseMatCM::from_data(data)
}

pub fn myint_to_field_mat<F>(mat: &DenseMatCM<MyInt, F>) -> DenseMatFieldCM<F>
where
    F: PrimeField + Send + Sync,
{
    let cols = mat.data.len();
    let rows = mat.data[0].len();

    let data = (0..cols)
    .into_par_iter()
    .map(|col_idx| {
        (0..rows)
            .map(|row_idx| {
                F::from(mat.data[col_idx][row_idx])
            })
            .collect()
    })
    .collect();

    DenseMatFieldCM::from_data(data)
}

pub fn gen_rand_shortint_mat(rows: usize, cols: usize) -> Vec<Vec<MyShortInt>>
{
    let bitwidth = std::mem::size_of::<MyShortInt>() * 8;
    let max_value = (1 << (bitwidth - 1) - 1) as MyShortInt;
    let min_value = - max_value;

    // Use parallel iterator to generate columns concurrently
    (0..cols)
    .into_par_iter()
    .map(|col_idx| {
        // Each thread gets its own RNG seeded with column index
        let mut rng = StdRng::seed_from_u64(col_idx as u64);
        (0..rows)
            .map(|_| {
                let value = rng.gen_range(min_value..max_value);
                MyShortInt::from(value)
            })
            .collect()
    })
    .collect()
}

#[allow(dead_code)]
fn mat_myint_to_field<F: PrimeField>(mat: Vec<Vec<MyInt>>) -> Vec<Vec<F>> {

    mat.into_iter()
        .map(|col| col.into_iter().map(|x| F::from(x)).collect())
        .collect()
}


pub fn gen_rand_shortint_vec(len: usize) -> Vec<MyShortInt>
{
    let bitwidth = std::mem::size_of::<MyShortInt>() * 8;
    let max_value = (1 << (bitwidth - 1) - 1) as MyShortInt;
    let min_value = - max_value;

    let mut rng = StdRng::seed_from_u64(len as u64);
    (0..len)
    .map(|_| {
        let value = rng.gen_range(min_value..max_value);
        MyShortInt::from(value)
    })
    .collect()
}

pub fn flatten_and_concat<F>(mats: &Vec<DenseMatCM<MyInt, F>>) -> DenseMatCM<MyInt, F> 
where
    F: PrimeField + Send + Sync,
{
    let mut vec: Vec<MyInt> =  Vec::new();

    for mat in mats {
        for col in mat.data.iter() {
            vec.extend(col.iter().cloned());
        }
    }

    DenseMatCM::from_data(vec![vec])
}

pub fn flatten_to_field_vec<F>(mats: &Vec<Vec<MyInt>>) -> Vec<F>
where
    F: PrimeField + Send + Sync,
{
    let mut vec: Vec<F> =  Vec::new();

    for col in mats.iter() {
        vec.extend(col.par_iter().map(|&x| {
            if x >= 0 {
                F::from(x as u64)
            } else {
                -F::from((-x) as u64)
            }
        }).collect::<Vec<_>>());
    }
    vec
}

pub fn vec_myint_to_field<F>(vec: &Vec<MyInt>) -> Vec<F>
where
    F: PrimeField + Send + Sync,
{
    vec.par_iter().map(|&x| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-x) as u64)
        }
    }).collect()
}

// Compute the multiplication of two col_major myint dense matrices
// 
pub fn mat_mul_myint (a: &Vec<Vec<MyInt>>, b: &Vec<Vec<MyInt>>) -> Vec<Vec<MyInt>>
{
    let a_rows = a[0].len();
    let a_cols = a.len();
    let b_rows = b[0].len();
    let b_cols = b.len();
    
    // Check dimension compatibility: a_cols must equal b_rows
    assert_eq!(a_cols, b_rows, "Matrix multiplication dimension mismatch: A cols {} != B rows {}", a_cols, b_rows);

    let mut result = 
        vec![vec![0 as MyInt; a_rows]; b_cols];

    result.par_iter_mut().enumerate().for_each(
        |(j, col)| {
        for i in 0..a_rows {
            for k in 0..a_cols {
                col[i] = col[i] + a[k][i] * b[j][k];
            }
        }
    });

    result
}

// Element-wise addition of two col_major matrices
//
pub fn mat_add_myint (a: &Vec<Vec<MyInt>>, b: &Vec<Vec<MyInt>>) -> Vec<Vec<MyInt>>
{
    let (rows, cols) = (a[0].len(), a.len());
    let mut result = vec![vec![0 as MyInt; rows]; cols];

    result.par_iter_mut().enumerate().for_each(
        |(j, col)| {
            for i in 0..rows {
                col[i] = a[j][i] + b[j][i];
            }
        }
    );
    result
}

// Element-wise scalar multiplication
//
pub fn mat_scalar_mul_myint (a: &Vec<Vec<MyInt>>, scalar: MyInt) -> Vec<Vec<MyInt>>
{
    let (rows, cols) = (a[0].len(), a.len());
    let mut result = vec![vec![0 as MyInt; rows]; cols];

    result.par_iter_mut().enumerate().for_each(
        |(j, col)| {
            for i in 0..rows {
                col[i] = a[j][i] * scalar;
            }
        }
    );
    result
}

// Element-wise subtraction of two col_major matrices
//
pub fn mat_sub_myint (a: &Vec<Vec<MyInt>>, b: &Vec<Vec<MyInt>>) -> Vec<Vec<MyInt>>
{
    let (rows, cols) = (a[0].len(), a.len());
    let mut result = vec![vec![0 as MyInt; rows]; cols];

    result.par_iter_mut().enumerate().for_each(
        |(j, col)| {
            for i in 0..rows {
                col[i] = a[j][i] - b[j][i];
            }
        }
    );
    result
}

// Element-wise subtraction of two vectors
//
pub fn vec_sub_myint (a: &Vec<MyInt>, b: &Vec<MyInt>) -> Vec<MyInt>
{
    let mut result = vec![0 as MyInt; a.len()];

    result.par_iter_mut().enumerate().for_each(
        |(j, val)| {
            *val = a[j] - b[j];
        }
    );
    result
}

// Element-wise scalar multiplication of two col_major matrices
//
pub fn mat_mul_scalar_myint (a: &Vec<Vec<MyInt>>, scalar: MyInt) -> Vec<Vec<MyInt>>
{
    let (rows, cols) = (a[0].len(), a.len());
    let mut result = vec![vec![0 as MyInt; rows]; cols];

    result.par_iter_mut().enumerate().for_each(
        |(j, col)| {
            for i in 0..rows {
                col[i] = a[j][i] * scalar;
            }
        }
    );
    result
}

// Element-wise application of the quantized sigmoid function
//
pub fn element_wise_phi<F> (input: &DenseMatCM<MyInt, F>) -> DenseMatCM<MyInt, F>
where
    F: PrimeField + Send + Sync,
{
    let (m, n) = input.shape;
    let mut output = DenseMatCM::new(m, n);

    // Initialize output matrix data structure
    output.data = vec![vec![MyInt::default(); m]; n];

    // Parallelization over cols
    output.data.par_iter_mut().enumerate().for_each(|(j, col)| {
        for i in 0..m {
            col[i] = table_builder::phi(input.data[j][i], std::mem::size_of::<MyShortInt>() * 8); // Using 8-bit output bitwidth
        }
    });

    output
}





#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;


    #[test]
    fn test_element_wise_phi() {

        let m = 2;
        let n = 2;
        let mut input: DenseMatCM<MyInt, BlsFr> = DenseMatCM::new(m, n);
        
        let test_data = vec![
            vec![127, -128],  // Using MyInt (i32) values
            vec![0, -1],
        ];
        input.data = test_data.clone();
        
        let result = element_wise_phi(&input);
        
        assert_eq!(result.shape, (m, n));
        
        for j in 0..n {
            for i in 0..m {
                let expected = table_builder::phi(test_data[j][i], std::mem::size_of::<MyShortInt>() * 8);
                assert_eq!(result.data[j][i], expected);
            }
        }

    }

}

