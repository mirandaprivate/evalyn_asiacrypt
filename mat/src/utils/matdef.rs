//! Define sparse matrix operations
//! 
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use rayon::prelude::*;
use std::fmt::Debug;

use crate::utils::xi;

use super::linear;

use crate::MyInt;

pub trait ShortInt:
    Sized
    + Copy
    + Clone
    + Debug
    + Send
    + Sync
    + Eq
    + PartialEq
    + FromToI32
    + From<bool>
    + MulAB<Self>
    + GetBit
{}


// Sparse matrix in Kronecker format
// The smaller matrices are in col major format
// 
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct SparseKronecker<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub shape1: (usize, usize),
    pub shape2: (usize, usize),
    pub data: (Vec<Vec<I>>, Vec<Vec<I>>),
    _marker: PhantomData<F>,
}

impl<I, F> SparseKronecker<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub fn new(
        shape1_value: (usize, usize),
        shape2_value: (usize, usize),
    ) -> Self {
        Self {
            shape1: shape1_value,
            shape2: shape2_value,
            data: (Vec::new(), Vec::new()),
            _marker: PhantomData,
        }
    }

    pub fn set_data(&mut self, data: (Vec<Vec<I>>, Vec<Vec<I>>)) {
        self.data = data;
    }

    pub fn from_data(data: (Vec<Vec<I>>, Vec<Vec<I>>)) -> Self {
        let m1 = data.0.len();
        let n1 = data.0[0].len();
        let m2 = data.1.len();
        let n2 = data.1[0].len();

        Self {
            shape1: (m1, n1),
            shape2: (m2, n2),
            data,
            _marker: PhantomData,
        }
    }
}
// Sparse matrix
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct SparseMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub shape: (usize, usize),
    pub data: Vec<(usize, usize, I)>,
    _marker: PhantomData<F>,
}

impl<I, F> SparseMat<I, F> 
where
    I: ShortInt,
    F: PrimeField + From<I>,
{

    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m,n),
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn set_data(&mut self, data: Vec<(usize, usize, I)>) {
        self.data = data;
    }
}

// Sparse matrix
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct SparseFieldMat<F>
where
    F: PrimeField,
{
    pub shape: (usize, usize),
    pub data: Vec<(usize, usize, F)>,
}

impl<F> SparseFieldMat<F> 
where
    F: PrimeField,
{

    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m,n),
            data: Vec::new(),
        }
    }

    pub fn set_data(&mut self, data: Vec<(usize, usize, F)>) {
        self.data = data;
    }

    pub fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if m > l_vec.len() {
            panic!("The length of the left challenge is not enough");
        }

        let result: Vec<std::sync::Mutex<F>> = (0..n).map(|_| std::sync::Mutex::new(F::zero())).collect();

        self.data.par_iter().for_each(|entry| {
            if entry.0 < m && entry.1 < n{
                let row = entry.0 as usize;
                let col = entry.1 as usize;
                let value = entry.2;
                
                if let Ok(mut result_elem) = result[col].lock() {
                    *result_elem += value * l_vec[row];
                }
            }
        });

        result.into_iter()
            .map(|mutex_elem| mutex_elem.into_inner().unwrap())
            .collect()
    }

    pub fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n > r_vec.len() {
            panic!("The length of the right challenge is not enough");
        }

        // 使用 Vec<Mutex<F>> 为每个结果元素单独加锁
        let result: Vec<std::sync::Mutex<F>> = (0..m).map(|_| std::sync::Mutex::new(F::zero())).collect();

        self.data.par_iter().for_each(|entry| {
            let row = entry.0;
            let col = entry.1;
            let value = entry.2;
            
            if row < m && col < n {
                if let Ok(mut result_elem) = result[row].lock() {
                    *result_elem += value * r_vec[col];
                }
            }
        });

        // 提取最终结果
        result.into_iter()
            .map(|mutex_elem| mutex_elem.into_inner().unwrap())
            .collect()
    }

    pub fn to_dense(&self) -> DenseMatFieldCM::<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let mut data: Vec<Vec<F>> = vec![vec![F::zero(); m]; n];
        for entry in &self.data {
            if entry.0 < m && entry.1 < n {
                data[entry.1 as usize][entry.0 as usize] = entry.2;
            }
        }

    DenseMatFieldCM::<F> { shape: (m, n), data }
    }
}


// Dense matrix in Col Major format
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct DenseMatCM<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub shape: (usize, usize),
    pub data: Vec<Vec<I>>,
    _marker: PhantomData<F>,
}

impl<I, F> DenseMatCM<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m,n),
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn default() -> Self {
        Self::new(0,0)
    }

    pub fn set_data(&mut self, data: Vec<Vec<I>>) {
        self.data = data;
    }

    pub fn from_vec(&mut self, vec: &Vec<I>) {

        let m = self.shape.0;
        let n = self.shape.1;

        if vec.len() != (m * n) {
            panic!("Dimension mismatch");
        }

        let mut data = Vec::new();
        for j in 0..n {
            let col =
            vec[(j * m )..(j * m + m)]
            .to_vec();
            data.push(col);
        }

        self.data = data;

    }

    // Convert the matrix to a square matrix
    // Ready for commitment
    // 
    pub fn to_square_mat(&self) -> DenseMatCM::<I, F> {
        let new_data = linear::reshape_mat_cm_keep_projection_short::<I>(
            &self.data,
        );

        let m = new_data[0].len();
        let n = new_data.len();

        DenseMatCM::<I, F> {
            shape: (m, n),
            data: new_data,
            _marker: PhantomData,
        }
    }

    pub fn to_square_myint(&self) -> DenseMatCM::<MyInt, F> {
        let new_data =
        linear::reshape_mat_cm_to_myint_keep_projection_short::<I>(
            &self.data,
        );

        let m = new_data[0].len();
        let n = new_data.len();

        DenseMatCM::<MyInt, F> {
            shape: (m, n),
            data: new_data,
            _marker: PhantomData,
        }
    }

    pub fn from_data(data: Vec<Vec<I>>) -> Self {
        let m = data[0].len();
        let n = data.len();
        Self {
            shape: (m, n),
            data,
            _marker: PhantomData,
        }
    }

}


// Dense matrix in Col Major format
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct DenseMatFieldCM<F>
where
    F: PrimeField,
{
    pub shape: (usize, usize),
    pub data: Vec<Vec<F>>,
}

impl<F> DenseMatFieldCM<F>
where
    F: PrimeField,
{
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m, n),
            data: Vec::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(0, 0)
    }

    pub fn set_data(&mut self, data: Vec<Vec<F>>) {
        self.data = data;
    }

    pub fn from_vec(&mut self, vec: Vec<F>) {
        let m = self.shape.0;
        let n = self.shape.1;

        if vec.len() != (m * n) {
            panic!("Dimension mismatch");
        }

        if n == 1 {
            self.data = vec![vec];
        } else {
            let mut data = Vec::new();
            for j in 0..n {
                let col = vec[(j * m)..(j * m + m)].to_vec();
                data.push(col);
            }

            self.data = data;
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        if self.shape.1 == 1 {
            self.data[0].clone()
        } else {
            let mut result: Vec<F> = Vec::new();
            for j in 0..self.shape.1 {
                result.extend(self.data[j].clone().into_iter());
            }

            // let mut result = Vec::with_capacity(self.shape.0 * self.shape.1);
            // for j in 0..self.shape.1 {
            //     for i in 0..self.shape.0 {
            //         result.push(self.data[j][i]);
            //     }
            // }
            result
        }
    }

    pub fn into_vec(&mut self) -> Vec<F> {
        if self.shape.1 == 1 {
            self.data.pop().unwrap()
        } else {
            // flatten the matrix
            let mut result: Vec<F> = Vec::new();
            for j in 0..self.shape.1 {
                result.extend(self.data[j].clone().into_iter());
            }
            self.data = Vec::new();
            result
        }
    }

    // Convert the matrix to a square matrix
    // Ready for commitment
    //
    pub fn to_square_mat(&self) -> DenseMatFieldCM<F> {
        let new_data = linear::reshape_mat_cm_keep_projection::<F>(&self.data);

        let m = new_data[0].len();
        let n = new_data.len();

        DenseMatFieldCM::<F> {
            shape: (m, n),
            data: new_data,
        }
    }

    pub fn from_data(data: Vec<Vec<F>>) -> Self {
        let m = data[0].len();
        let n = data.len();
        Self {
            shape: (m, n),
            data,
        }
    }

    pub fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        linear::proj_left_cm::<F>(&self.data, &l_vec)
    }

    pub fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        linear::proj_right_cm::<F>(&self.data, &r_vec)
    }

    // Project left with challenges
    pub fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;

        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        let l_vec = xi::xi_from_challenges::<F>(&xl);
        linear::proj_left_cm::<F>(&self.data, &l_vec)
    }

    // Project right with challenges
    pub fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let n = self.shape.1;

        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let r_vec = xi::xi_from_challenges::<F>(&xr);
        linear::proj_right_cm::<F>(&self.data, &r_vec)
    }

    // Project with left and right challenges
    pub fn proj_lr_challenges(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        linear::proj_cm_on_tensor::<F>(&self.data, xl, xr)
    }

    // Matrix addition - delegates to parallel implementation in matop
    pub fn mat_add(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F>
    where
        F: Send + Sync,
    {
        self.par_add(other)
    }

    // Matrix multiplication - delegates to parallel implementation in matop
    pub fn mat_mul(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F>
    where
        F: Send + Sync,
    {
        self.par_mul(other)
    }

    // Matrix subtraction - delegates to parallel implementation in matop
    pub fn mat_sub(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F>
    where
        F: Send + Sync,
    {
        self.par_sub(other)
    }

    // Matrix subtraction - delegates to parallel implementation in matop
    pub fn hadamard_prod(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F>
    where
        F: Send + Sync,
    {
        self.par_hadamard(other)
    }
}

pub fn concat_mat_to_vec<F>(mat_list: &Vec<DenseMatFieldCM<F>>) -> DenseMatFieldCM<F>
where
    F: PrimeField,
{
    let mut result = Vec::new();
    for i in 0..mat_list.len() {
        result.extend(mat_list[i].to_vec());
    }

    let mat = DenseMatFieldCM::<F>::from_data(vec![result]);
    mat
}

pub fn concat_mat_to_square<F>(mat_list: &Vec<DenseMatFieldCM<F>>) -> DenseMatFieldCM<F>
where
    F: PrimeField,
{
    let mut result = Vec::new();
    for i in 0..mat_list.len() {
        result.extend(mat_list[i].to_vec());
    }

    let data = linear::reshape_mat_cm_keep_projection::<F>(&vec![result]);

    let m = data[0].len();
    let n = data.len();
    DenseMatFieldCM::<F> {
        shape: (m, n),
        data,
    }
}

// Rotation sparse matrix in index format
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct RotationMatIndexFormat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub shape: (usize, usize),
    // For a rotation matrix,
    // it is sufficient to store the indices of each row
    pub data: Vec<MyInt>,
    _marker: PhantomData<F>,
    _marker2: PhantomData<I>,
}

impl<I, F> RotationMatIndexFormat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m,n),
            data: Vec::new(),
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    pub fn set_data(&mut self, data: Vec<MyInt>) {
        self.data = data;
    }

    // Split into boolean matrix
    // start from the most important bit
    // 
    pub fn split_into_boolean(&self) -> (Vec<bool>, Vec<Vec<bool>>) {
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let kappa = n.ilog2() as usize;

        let mut result = Vec::new();

        for i in 0..kappa {
            let ii = kappa - i - 1;
            let cur = self.data.par_iter().map(|x|{
                if *x < 0 {
                    false
                } else {
                    ((*x >> ii) & 1) == 1
                }
            }).collect();

            result.push(cur);
        }

        let if_invalid: Vec<bool> =
        self.data.par_iter().map(|x|{
            if *x < 0 {
                true
            } else {
                false
            }
        }).collect();

        (if_invalid, result)
    }

    // Compute the right-projection via boolean matrix
    // useful for lookup proofs
    // 
    pub fn proj_right_via_boolean(&self, xr: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let kappa = n.ilog2() as usize;

        if kappa != xr.len() {
            println!("kappa: {}, xr.len(): {}", kappa, xr.len());
            panic!("The length of the right challenge is not consistent with the matrix");
        }

        

        let (if_invalid, boolean_mat) = self.split_into_boolean();

        let mut result = vec![F::one(); m];
        for i in 0..kappa {
            let boolean_vec = &boolean_mat[i];
            let factor = xr[i];
            let cur =
            boolean_vec.into_par_iter()
            .zip(if_invalid.par_iter())
            .map(|(x, y)| {
                if *y {
                    F::zero()
                } else if *x {
                    factor
                } else {
                    F::one()
                }
            }).collect::<Vec<F>>();
            result = linear::vec_element_wise_mul(&result, &cur);
        }

        result
    }

    pub fn from_data(data: Vec<MyInt>, n: usize) -> Self {
        let m = data.len();
        Self {
            shape: (m, n),
            data,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    pub  fn to_dense_bool(&self) ->
    DenseMatCM<bool, F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let mut data: Vec<Vec<bool>> = vec![vec![false; m]; n];

        for i in 0..m {
            if self.data[i] >= 0 {
                let idx = self.data[i] as usize;
                data[idx][i] = true;
            }
        }

        DenseMatCM::<bool, F> {
            shape: (m, n),
            data: data,
            _marker: PhantomData,
        }
    }

    
}


// Diagnoal block matrix
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct DiagBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub num_blocks: usize,
    pub block_shape: (usize, usize),
    pub data: Vec<Vec<Vec<I>>>,
    _marker: PhantomData<F>,
}

impl <I, F> DiagBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub fn new(num_blocks: usize, block_shape: (usize, usize)) -> Self {
        Self {
            num_blocks,
            block_shape,
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn set_data(&mut self, data: Vec<Vec<Vec<I>>>) {
        self.data = data;
    }

    pub fn from_dense_block(
        dense_block: &DenseBlockMat::<I, F>,
    ) -> Self {
        let block_shape = dense_block.block_shape;
        let num_blocks = dense_block.num_blocks;

        let mut result =
        Self::new(num_blocks, block_shape);

        result.set_data(dense_block.data.clone());

        result
    }

    pub fn from_data(data: Vec<Vec<Vec<I>>>) -> Self {
        let num_blocks = data.len();
        let block_shape =
        (data[0][0].len(), data[0].len());
        Self {
            num_blocks,
            block_shape,
            data,
            _marker: PhantomData,
        }
    }
    
}

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct DenseBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator,
{
    pub num_blocks: usize,
    pub block_shape: (usize, usize),
    pub data: Vec<Vec<Vec<I>>>,
    _marker: PhantomData<F>,
}

impl <I, F> DenseBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator,
{
    pub fn new(num_blocks: usize, block_shape: (usize, usize)) -> Self {
        Self {
            num_blocks,
            block_shape,
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn set_data(&mut self, data: Vec<Vec<Vec<I>>>) {
        self.data = data;
    }

    pub fn from_vec(&mut self, vec: &Vec<I>) {

        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;

        let num_blocks = self.num_blocks;

        if vec.len() != (m1 * n1 * num_blocks) {
            panic!("Dimension mismatch");
        }

        let m = m1 * num_blocks;

        let mut data = Vec::new();
        for i in 0..num_blocks {
            let mut block = Vec::new();
            for j in 0..n1 {
                let col =
                vec[(j * m + i * m1)..(j * m + i * m1 + m1)]
                .to_vec();
                block.push(col);
            }
            data.push(block);
        }

        self.num_blocks = num_blocks;
        self.data = data;

    }


    pub fn to_square_mat(&self) -> DenseMatCM::<I, F> {
        self.to_dense().to_square_mat()
    }

    pub fn to_square_myint(&self) -> DenseMatCM::<MyInt, F> {
        
        self.to_dense().to_square_myint()
    }
    
    pub fn from_data(data: Vec<Vec<Vec<I>>>) -> Self {
        let num_blocks = data.len();
        let block_shape =
        (data[0][0].len(), data[0].len());
        Self {
            num_blocks,
            block_shape,
            data,
            _marker: PhantomData,
        }
    }
}

// Define matrix projections for all kinds of matrices
pub trait MatOps<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    fn to_dense(&self) -> DenseMatCM::<I, F>;
    fn to_vec(&self) -> Vec<I>;
    fn to_field_vec(&self) -> Vec<F>;
    fn gen_rand(&mut self, k: usize);
    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F>;
    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F>;
    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F>;
    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F>;
    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F;
    fn get_shape(&self) -> (usize, usize);
    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>>;
    fn to_bool_neg(&self) -> DenseMatCM<I,F>;
    fn bit_decomposition(&self, k: usize)
    -> (DenseMatCM::<bool,F>,
        DenseMatCM<MyInt, F>,
        Vec<DenseMatCM<bool,F>>);
    // Truncating the least important k bit,
    // and then restirct the values between
    // -2^{k-1}  and 2^{k-1} 
    fn clamp(&self, k: usize) -> DenseMatCM<I,F>;
    fn clear(&mut self);
}

pub fn inner_product_generic<I, F> (
    a: &[I],
    b: &[F],
) -> F
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator,
{   
    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| F::from(*a) * b).sum()
}

// Projection of a col major matrix to the left vector
// Col Major
pub fn proj_left_cm_generic<I, F> (
    mat: &Vec<Vec<I>>,
    l_vec: &Vec<F>,
) -> Vec<F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    let n = mat.len();
    let m = mat[0].len();

    if m > l_vec.len() {
        panic!("The length of the left vector is not enough");
    }

    let l_vec = l_vec[0..m].to_vec();

    let mut result = Vec::new();
    for i in 0..n {
        let col = &mat[i];
        let ip_col = inner_product_generic::<I,F>(
            &col, &l_vec);
        result.push(ip_col);
    }
    result
}


// Projection of a col major matrix to the left vector
pub fn proj_right_cm_generic<I, F> (
    mat: &Vec<Vec<I>>,
    r_vec: &Vec<F>,
) -> Vec<F> 
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    let n = mat.len();
    let m = mat[0].len();

    if n > r_vec.len() {
        panic!("The length of the right vector is not enough");
    }

    let r_vec = r_vec[0..n].to_vec();
    
    let mut result = Vec::new();
    for i in 0..m {
        let row =
        mat.par_iter()
        .map(|x| x[i])
        .collect::<Vec<I>>();
        
        let ip_row = inner_product_generic::<I,F>(
            &row, &r_vec);
        result.push(ip_row);
    }
    result
}

pub fn proj_lr_cm_generic<I, F> (
    mat: &Vec<Vec<I>>,
    xl: &Vec<F>,
    xr: &Vec<F>,
) -> F
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    let la = proj_left_cm_generic::<I, F>(&mat, &xl);
    let result = linear::inner_product::<F>(&la, &xr);
    result
}

impl<I, F> MatOps<I, F> for DenseMatCM<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{

    fn clear(&mut self) {
        self.data = Vec::new();
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        self.clone()
    }

    fn to_vec(&self) -> Vec<I> {
        self.data.par_iter().flatten()
        .map(|x| *x)
        .collect()
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        self.data.iter().map(|x|{
            x.par_iter().map(|y|{
                scalar * F::from(*y)
            }).collect()
        }).collect()
    }

    // Convert the matrix to a field vector
    // following col major order
    // vec(a) \bullet tensor(xr, xl) = l a r
    // 
    fn to_field_vec(&self) -> Vec<F> {
        self.data.par_iter().flatten()
        .map(|x| F::from(*x))
        .collect()
    }

    fn gen_rand(&mut self, k: usize) {
        let low_bound = - (1 << (k-1)) + 1;
        let up_bound = (1 << (k-1)) - 1;

        let data = (0..self.shape.1)
        .into_iter().map(|_|{
            use rand::Rng;
            (0..self.shape.0).into_par_iter().map(|_|{
                let mut rng = rand::rng();
                I::from_myint(rng.random_range(low_bound..up_bound))
            }).collect()
        }).collect();
        
        self.set_data(data);
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
    
        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }
    
    
        let l_vec = xi::xi_from_challenges::<F>(&xl);
    
        proj_left_cm_generic::<I, F>(&self.data, &l_vec)
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        proj_left_cm_generic::<I, F>(&self.data, l_vec)
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let n = self.shape.1;
    
    
        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }
    
        let r_vec = xi::xi_from_challenges::<F>(&xr);
    
        proj_right_cm_generic::<I, F>(&self.data, &r_vec)
    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        proj_right_cm_generic::<I, F>(&self.data, r_vec)
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m = self.shape.0;
        let n = self.shape.1;
    
        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }
    
        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }
    
        let l_vec = xi::xi_from_challenges::<F>(&xl);
        let r_vec = xi::xi_from_challenges::<F>(&xr);
    
        let result = proj_lr_cm_generic::<I, F>(&self.data, &l_vec, &r_vec);
        result
    }

    fn to_bool_neg(&self) -> DenseMatCM<I, F> {
        let data =
        self.data.iter().map(|x|{
            x.par_iter().map(|y|{
                if y == &I::from_myint(0) {
                    I::from_myint(1)
                } else {
                    I::from_myint(0)
                }
            }).collect()
        }).collect();

        DenseMatCM::<I, F> {
            shape: self.shape,
            data,
            _marker: PhantomData,
        }
    }

    fn bit_decomposition(&self, k: usize)
        -> (
            DenseMatCM::<bool,F>,
            DenseMatCM::<MyInt, F>,
            Vec<DenseMatCM::<bool,F>>,
        ) {

        let (sign_data, abs_data)
        : (Vec<Vec<bool>>, Vec<Vec<MyInt>>) =
        self.data.par_iter().map(|x|{
            x.iter().map(|y|{
                let ymyint = y.to_myint();
                (ymyint >= 0, ymyint.abs() as MyInt)
            }).unzip()
        }).unzip();


        // let sign_data: Vec<Vec<bool>> =
        // self.data.par_iter().map(|x|{
        //     x.par_iter().map(|y|{
        //         y.to_myint() >= 0
        //     }).collect()
        // }).collect();

        // let abs_data: Vec<Vec<myint>> =
        // self.data.par_iter().map(|x|{
        //     x.par_iter().map(|y|{
        //         y.to_myint().abs()
        //     }).collect()
        // }).collect();

        // println!("\n raw_data: {:?} \n", self.to_dense().data[0]);
        // println!("sign_data: {:?} \n", sign_data[0]);


        let sign_mat = DenseMatCM::<bool,F> {
            shape: self.shape,
            data: sign_data,
            _marker: PhantomData,
        };

        let abs_mats =
        DenseMatCM::<MyInt, F> {
            shape: self.shape,
            data: abs_data,
            _marker: PhantomData,
        };

        let mut bit_mats = Vec::new();

        for kappa in 0..k {
            let bit_data =
            abs_mats.data.par_iter().map(|x|{
                x.iter().map(|y|{
                    let bit = ((y >> kappa) & 1) == 1;
                    bit
                }).collect()
            }).collect();

            let bit_mat = DenseMatCM::<bool,F> {
                shape: self.shape,
                data: bit_data,
                _marker: PhantomData,
            };

            bit_mats.push(bit_mat);
        }

        (sign_mat, abs_mats, bit_mats)

    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        let low_bound = - 1 << (2 * k-1);
        let up_bound = 1 << (2 * k-1);
        let th = 1 << k;

        let low_bound_value = -1 << (k-1);
        let up_bound_value = 1 << (k-1);

        let data =
        self.data.iter().map(|x|{
            x.par_iter().map(|y|{
                if y.to_myint() <= low_bound {
                    I::from_myint(low_bound_value)
                } else if y.to_myint() >= up_bound {
                    I::from_myint(up_bound_value)
                } else {
                    I::from_myint(y.to_myint() / th)
                }
            }).collect()
        }).collect();
        
        DenseMatCM::<I,F> {
            shape: self.shape,
            data,
            _marker: PhantomData,
        }
    }
}


impl<I, F> MatOps<I, F> for SparseKronecker<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{

    fn clear(&mut self) {
        self.data = (Vec::new(), Vec::new());
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        let m1 = self.shape1.0;
        let n1 = self.shape1.1;
        let m2 = self.shape2.0;
        let n2 = self.shape2.1;

        let m = m1 * m2;
        let n = n1 * n2;

        let mat1 = &self.data.0;
        let mat2 = &self.data.1;

        let mut data: Vec<Vec<I>> = vec![vec![I::from_myint(0); m]; n];
        for i1 in 0..m1 {
            for j1 in 0..n1 {
                for i2 in 0..m2 {
                    for j2 in 0..n2 {
                        data[j1*n2+j2][i1*m2+i2] =
                        mat1[j1][i1].mul_mix(mat2[j2][i2]);
                    }
                }
            }
        }

        DenseMatCM::<I,F> {
            shape: (m, n),
            data,
            _marker: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        let m1 = self.shape1.0;
        let n1 = self.shape1.1;
        let m2 = self.shape2.0;
        let n2 = self.shape2.1;

        let m = m1 * m2;
        let n = n1 * n2;

        let mat1 = &self.data.0;
        let mat2 = &self.data.1;

        let mut data: Vec<Vec<F>> = vec![vec![F::zero(); m]; n];
        for i1 in 0..m1 {
            for j1 in 0..n1 {
                for i2 in 0..m2 {
                    for j2 in 0..n2 {
                        data[j1*n2+j2][i1*m2+i2] =
                        F::from(mat1[j1][i1].mul_mix(mat2[j2][i2]))
                        .mul(&scalar);
                    }
                }
            }
        }

        data
    }

    fn to_vec(&self) -> Vec<I> {
        self.to_dense().to_vec()
    }

    fn to_field_vec(&self) -> Vec<F> {
        self.to_dense().to_field_vec()
    }

    fn get_shape(&self) -> (usize, usize) {
        let m1 = self.shape1.0;
        let n1 = self.shape1.1;
        let m2 = self.shape2.0;
        let n2 = self.shape2.1;

        (m1 * m2, n1 * n2)
    }

    fn gen_rand(&mut self, k: usize) {
        let m1 = self.shape1.0;
        let n1 = self.shape1.1;
        let m2 = self.shape2.0;
        let n2 = self.shape2.1;

        let mut mat_1 =
        DenseMatCM::<I, F>::new(m1, n1);
        mat_1.gen_rand(k);
        let mut mat_2 =
        DenseMatCM::<I, F>::new(m2, n2);
        mat_2.gen_rand(k);
        
        self.set_data((mat_1.data, mat_2.data));
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {
        let m1 = self.shape1.0;
        let m2 = self.shape2.0;

        let m = m1 * m2;
    
        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }
    
        let logm1 = m1.ilog2() as usize;
        let logm2 = m2.ilog2() as usize;

        let mat1 = &self.data.0;
        let mat2 = &self.data.1;

        let xl1 = &xl[0..logm1].to_vec();
        let xl2 = &xl[logm1..(logm1 + logm2)].to_vec();


        let l_vec1 = xi::xi_from_challenges::<F>(&xl1);
        let l_vec2 = xi::xi_from_challenges::<F>(&xl2);


        let la1 = proj_left_cm_generic::<I, F>(
            mat1, &l_vec1
        );

        let la2 = proj_left_cm_generic::<I, F>(
            mat2, &l_vec2
        );

        let result = linear::vec_kronecker(&la1,&la2);
        result
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
       self.to_dense().proj_left(l_vec)
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let n1 = self.shape1.1;
        let n2 = self.shape2.1;

        let n = n1 * n2;

      
        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }


        let logn1 = n1.ilog2() as usize;
        let logn2 = n2.ilog2() as usize;

        let mat1 = &self.data.0;
        let mat2 = &self.data.1;

        let xr1 = &xr[0..logn1].to_vec();
        let xr2 = &xr[logn1..(logn1 + logn2)].to_vec();

        let r_vec1 = xi::xi_from_challenges::<F>(&xr1);
        let r_vec2 = xi::xi_from_challenges::<F>(&xr2);

        let ar1 = proj_right_cm_generic::<I, F>(
            mat1, &r_vec1
        );

        let ar2 = proj_right_cm_generic::<I, F>(
            mat2, &r_vec2
        );

        let result = linear::vec_kronecker(&ar1,&ar2);
        result
    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        self.to_dense().proj_right(r_vec)
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m1 = self.shape1.0;
        let n1 = self.shape1.1;
        let m2 = self.shape2.0;
        let n2 = self.shape2.1;

        let m = m1 * m2;
        let n = n1 * n2;

        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }
    
        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let logm1 = m1.ilog2() as usize;
        let logm2 = m2.ilog2() as usize;
        let logn1 = n1.ilog2() as usize;
        let logn2 = n2.ilog2() as usize;

        let mat1 = &self.data.0;
        let mat2 = &self.data.1;

        let xl1 = &xl[0..logm1].to_vec();
        let xl2 = &xl[logm1..(logm1 + logm2)].to_vec();
        let xr1 = &xr[0..logn1].to_vec();
        let xr2 = &xr[logn1..(logn1 + logn2)].to_vec();

        let l_vec1 = xi::xi_from_challenges::<F>(&xl1);
        let l_vec2 = xi::xi_from_challenges::<F>(&xl2);
        let r_vec1 = xi::xi_from_challenges::<F>(&xr1);
        let r_vec2 = xi::xi_from_challenges::<F>(&xr2);

        let lar1 = proj_lr_cm_generic::<I, F>(
            mat1, &l_vec1,&r_vec1
        );

        let lar2 = proj_lr_cm_generic::<I, F>(
            mat2, &l_vec2,&r_vec2
        );

        let result = lar1 * lar2;
        result
    }

    fn to_bool_neg(&self) -> DenseMatCM<I, F> {
        self.to_dense().to_bool_neg()
    }

    fn bit_decomposition(&self, k: usize)
        -> (DenseMatCM::<bool,F>,
            DenseMatCM<MyInt, F>,
            Vec<DenseMatCM<bool,F>>) {
        
        self.to_dense().bit_decomposition(k)
    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        self.to_dense().clamp(k)
    }

}

impl<I, F> MatOps<I, F> for DenseBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    fn clear(&mut self) {
        self.data = Vec::new();
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.block_shape.1;

        let mut data: Vec<Vec<I>> = vec![vec![I::from_myint(0); m]; n];
        for i in 0..self.num_blocks {
            for j in 0..self.block_shape.1 {
                for ii in 0..self.block_shape.0 {
                    data[j][i*self.block_shape.0 + ii] =
                    self.data[i][j][ii];
                }
            }
        }

        DenseMatCM::<I, F> {
            shape: (m, n),
            data: data,
            _marker: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.block_shape.1;

        let mut data: Vec<Vec<F>> = vec![vec![F::zero(); m]; n];
        for i in 0..self.num_blocks {
            for j in 0..self.block_shape.1 {
                for ii in 0..self.block_shape.0 {
                    data[j][i*self.block_shape.0 + ii] =
                    F::from(self.data[i][j][ii]).mul(&scalar);
                }
            }
        }

        data
    }

    fn to_vec(&self) -> Vec<I> {
        self.to_dense().to_vec()
    }


    fn to_field_vec(&self) -> Vec<F> {
        self.to_dense().to_field_vec()
    }

    fn get_shape(&self) -> (usize, usize) {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.block_shape.1;
        (m, n)
    }

    fn gen_rand(&mut self, k: usize) {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;


        let data = (0..self.num_blocks)
        .into_par_iter().map(|_|{
            let mut dense_mat =
            DenseMatCM::<I, F>::new(m1, n1);
            dense_mat.gen_rand(k);
            dense_mat.data
        }).collect();

        self.set_data(data);
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let m = num * m1;
        let n = n1;

        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        let logm1 = m1.ilog2() as usize;
        let lognum = num.ilog2() as usize;

        let xl1 = &xl[0..lognum].to_vec();
        let xi_l1 = xi::xi_from_challenges::<F>(&xl1);
        let xl2 = &xl[lognum..(logm1 + lognum)].to_vec();

        let mut result = vec![F::zero();n];

        (0..num).into_par_iter().map(|i|{
            let factor = xi_l1[i];
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_left_challenges(&xl2);
            linear::vec_scalar_mul(&cur, &factor)
        }).collect::<Vec<Vec<F>>>()
        .iter().for_each(|cur|{
            result = linear::vec_addition(&result, cur);
        });

        result
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        self.to_dense().proj_left(l_vec)
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let n = n1;

        
        if n > (1 << xr.len()) {
            panic!("The length of the left challenge is not enough");
        }

        (0..num).into_par_iter().map(|i|{
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_right_challenges(&xr);
            cur
        }).flatten().collect()

    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        self.to_dense().proj_right(r_vec)
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let m = num * m1;
        let n = n1;

        
        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let logm1 = m1.ilog2() as usize;
        let lognum = num.ilog2() as usize;

        let xl1 = &xl[0..lognum].to_vec();
        let xi_l1 = xi::xi_from_challenges::<F>(&xl1);
        let xl2 = &xl[lognum..(logm1 + lognum)].to_vec();


        (0..num).into_par_iter().map(|i|{
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_lr(&xl2, &xr);
            cur * xi_l1[i]
        }).sum()


    }

    fn to_bool_neg(&self) -> DenseMatCM<I,F> {
        self.to_dense().to_bool_neg()
    }

    fn bit_decomposition(&self, k: usize)
        -> (DenseMatCM::<bool,F>,
            DenseMatCM<MyInt, F>,
            Vec<DenseMatCM<bool,F>>) {
        self.to_dense().bit_decomposition(k)
    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        self.to_dense().clamp(k)
    }

}


impl<I, F> MatOps<I, F> for DiagBlockMat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    fn clear(&mut self) {
        self.data = Vec::new();
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.num_blocks * self.block_shape.1;

        let mut data: Vec<Vec<I>> = vec![vec![I::from_myint(0); m]; n];
        for i in 0..self.num_blocks {
            for ii in 0..self.block_shape.0 {
                for jj in 0..self.block_shape.1 {
                    data[i*self.block_shape.1 + jj][i*self.block_shape.0 + ii] =
                    self.data[i][jj][ii];    
                }
            }
        }
    

        DenseMatCM::<I, F> {
            shape: (m, n),
            data: data,
            _marker: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.num_blocks * self.block_shape.1;

        let mut data: Vec<Vec<F>> = vec![vec![F::zero(); m]; n];
        for i in 0..self.num_blocks {
            for ii in 0..self.block_shape.0 {
                for jj in 0..self.block_shape.1 {
                    data[i*self.block_shape.0 + jj][i*self.block_shape.0 + ii] =
                    F::from(self.data[i][jj][ii])
                    .mul(&scalar);    
                }
            }
        }

        data
    }

    fn to_vec(&self) -> Vec<I> {
        self.to_dense().to_vec()
    }

    fn to_field_vec(&self) -> Vec<F> {
        self.to_dense().to_field_vec()
    }

    fn get_shape(&self) -> (usize, usize) {
        let m = self.num_blocks * self.block_shape.0;
        let n = self.num_blocks * self.block_shape.1;
        (m, n)
    }

    fn gen_rand(&mut self, k: usize) {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;


        let data = (0..self.num_blocks)
        .into_par_iter().map(|_|{
            let mut dense_mat =
            DenseMatCM::<I, F>::new(m1, n1);
            dense_mat.gen_rand(k);
            dense_mat.data
        }).collect();

        self.set_data(data);
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let m = num * m1;

        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        let logm1 = m1.ilog2() as usize;
        let lognum = num.ilog2() as usize;

        let xl1 = &xl[0..lognum].to_vec();
        let xi_l1 = xi::xi_from_challenges::<F>(&xl1);
        let xl2 = &xl[lognum..(logm1 + lognum)].to_vec();


        let result =(0..num).into_par_iter().map(|i|{
            let factor = xi_l1[i];
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_left_challenges(&xl2);
            linear::vec_scalar_mul(&cur, &factor)
        }).flatten().collect::<Vec<F>>();

        result
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let m = num * m1;
        
        if m > l_vec.len() {
            panic!("The length of the left challenge is not enough");
        }

        let result =(0..num).into_par_iter().map(|i|{
            let l_vec_cur = l_vec[(i*m1)..(i*m1+m1)].to_vec();
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_left(&l_vec_cur);
            cur
        }).flatten().collect::<Vec<F>>();

        result
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let n = num * n1;

        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let logn1 = n1.ilog2() as usize;
        let lognum = num.ilog2() as usize;

        let xr1 = &xr[0..lognum].to_vec();
        let xi_r1 = xi::xi_from_challenges::<F>(&xr1);
        let xr2 = &xr[lognum..(logn1 + lognum)].to_vec();


        let result =(0..num).into_par_iter().map(|i|{
            let factor = xi_r1[i];
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_right_challenges(&xr2);
            linear::vec_scalar_mul(&cur, &factor)
        }).flatten().collect::<Vec<F>>();

        result
    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let n = num * n1;

        if n > r_vec.len() {
            panic!("The length of the right challenge is not enough");
        }

        let result =(0..num).into_par_iter().map(|i|{
            let r_vec_cur = r_vec[(i*n1)..(i*n1+n1)].to_vec();
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_right(&r_vec_cur);
            cur
        }).flatten().collect::<Vec<F>>();

        result   
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m1 = self.block_shape.0;
        let n1 = self.block_shape.1;
        let num = self.num_blocks;

        let m = num * m1;
        let n = num * n1;

        
        if m > (1 << xr.len()) {
            panic!("The length of the left challenge is not enough");
        }

        if n > (1 << xr.len()) {
            panic!("The length of the left challenge is not enough");
        }

        let logm1 = m1.ilog2() as usize;
        let lognum = num.ilog2() as usize;
        let logn1 = n1.ilog2() as usize;
 
        let xl1 = &xl[0..lognum].to_vec();
        let xi_l1 = xi::xi_from_challenges::<F>(&xl1);
        let xl2 = &xl[lognum..(logm1 + lognum)].to_vec();
        let xr1 = &xr[0..lognum].to_vec();
        let xi_r1 = xi::xi_from_challenges::<F>(&xr1);
        let xr2 = &xr[lognum..(logn1 + lognum)].to_vec();


        (0..num).into_par_iter().map(|i|{
            let dense_mat = DenseMatCM::<I, F>{
                shape: (m1, n1),
                data: self.data[i].clone(),
                _marker: PhantomData,
            };
            let cur = dense_mat.proj_lr(&xl2, &xr2);
            cur * xi_l1[i] * xi_r1[i]
        }).sum()


    }

    fn to_bool_neg(&self) -> DenseMatCM<I,F> {
        self.to_dense().to_bool_neg()
    }

    fn bit_decomposition(&self, k: usize)
    -> (DenseMatCM::<bool,F>,
        DenseMatCM<MyInt, F>,
        Vec<DenseMatCM<bool,F>>) {
        self.to_dense().bit_decomposition(k)    
    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        self.to_dense().clamp(k)
    }

}


impl<I, F> MatOps<I, F> for SparseMat<I,F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    fn clear(&mut self) {
        self.data = Vec::new();
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let mut data: Vec<Vec<I>> = vec![vec![I::from_myint(0); m]; n];
        for entry in &self.data {
            if entry.0 < m && entry.1 < n {
                data[entry.1 as usize][entry.0 as usize] = entry.2;
            }
        }

        DenseMatCM::<I,F> {
            shape: (m, n),
            data: data,
            _marker: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        self.to_dense().scalar_mul(scalar)
    }

    fn to_vec(&self) -> Vec<I> {
        self.to_dense().to_vec()
    }

    fn to_field_vec(&self) -> Vec<F> {
        self.to_dense().to_field_vec()
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if m > l_vec.len() {
            panic!("The length of the left challenge is not enough");
        }

        let result: Vec<std::sync::Mutex<F>> = (0..n).map(|_| std::sync::Mutex::new(F::zero())).collect();

        self.data.par_iter().for_each(|entry| {
            if entry.0 < m && entry.1 < n{
                let row = entry.0 as usize;
                let col = entry.1 as usize;
                let value = entry.2;
                
                if let Ok(mut result_elem) = result[col].lock() {
                    *result_elem += F::from(value) * l_vec[row];
                }
            }
        });

        result.into_iter()
            .map(|mutex_elem| mutex_elem.into_inner().unwrap())
            .collect()
    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n > r_vec.len() {
            panic!("The length of the right challenge is not enough");
        }

        // 使用 Vec<Mutex<F>> 为每个结果元素单独加锁
        let result: Vec<std::sync::Mutex<F>> = (0..m).map(|_| std::sync::Mutex::new(F::zero())).collect();

        self.data.par_iter().for_each(|entry| {
            let row = entry.0;
            let col = entry.1;
            let value = entry.2;
            
            if row < m && col < n {
                if let Ok(mut result_elem) = result[row].lock() {
                    *result_elem += F::from(value) * r_vec[col];
                }
            }
        });

        // 提取最终结果
        result.into_iter()
            .map(|mutex_elem| mutex_elem.into_inner().unwrap())
            .collect()
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {

        let xi_l = xi::xi_from_challenges::<F>(&xl);

        self.proj_left(&xi_l)
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {

        let xi_r = xi::xi_from_challenges::<F>(&xr);

        self.proj_right(&xi_r)
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m = self.shape.0;
        let n = self.shape.1;

        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let xi_l = xi::xi_from_challenges::<F>(&xl);
        let ar = self.proj_right_challenges(xr);
        linear::inner_product::<F>(&ar, &xi_l)
    }

    fn to_bool_neg(&self) -> DenseMatCM<I,F> {
        self.to_dense().to_bool_neg()
    }

    fn gen_rand(&mut self, _: usize) {
        let m = self.shape.0;
        let n = self.shape.1; // 改为 usize 而不是 MyInt

        let data = (0..m).into_par_iter().map(|_|{
            use rand::Rng;
            let mut rng = rand::rng();
            let col = rng.random_range(0..n); // col 现在是 usize
            (col, rng.random_range(0..n), I::from_myint(1)) // 返回 (usize, usize, I)
        }).collect();

        self.set_data(data);
    }


    fn bit_decomposition(&self, k: usize)
    -> (DenseMatCM::<bool,F>,
        DenseMatCM<MyInt, F>,
        Vec<DenseMatCM<bool,F>>) {
        self.to_dense().bit_decomposition(k) 
    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        self.to_dense().clamp(k)
    }

}


impl<I, F> MatOps<I, F> for RotationMatIndexFormat<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    fn clear(&mut self) {
        self.data = Vec::new();
    }

    fn to_dense(&self) -> DenseMatCM::<I, F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n & (n-1) != 0 {
            panic!("The number of columns is not a power of 2");
        }

        let mut data: Vec<Vec<I>> = vec![vec![I::from_myint(0); m]; n];
        for i in 0..m {
            if self.data[i] >= 0 {
                let idx = self.data[i] as usize;
                data[idx][i] = I::from_myint(1);
            }
        }

        DenseMatCM::<I, F> {
            shape: (m, n),
            data: data,
            _marker: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: F) -> Vec<Vec<F>> {
        self.to_dense().scalar_mul(scalar)
    }

    fn to_vec(&self) -> Vec<I> {
        self.to_dense().to_vec()
    }

    fn to_field_vec(&self) -> Vec<F> {
        self.to_dense().to_field_vec()
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }


    fn gen_rand(&mut self, _: usize) {
        let m = self.shape.0;
        let n = self.shape.1 as MyInt;
        
        let data = (0..m).into_par_iter().map(|_|{
            use rand::Rng;
            let mut rng = rand::rng();
            rng.random_range(0..n)
        }).collect();

        self.set_data(data);
    }

    fn proj_left_challenges(&self, xl: &Vec<F>) -> Vec<F> {

        let xi_l = xi::xi_from_challenges::<F>(&xl);
        
        self.proj_left(&xi_l)
    }

    fn proj_left(&self, l_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if m > l_vec.len() {
            panic!("The length of the left challenge is not enough");
        }

        let result = std::sync::Arc::new(
            std::sync::Mutex::new(
                vec![F::zero(); n]
            )
        );

        (0..m).into_par_iter().for_each(|i| {
            let idx = self.data[i];
            if idx >= 0 {
                let idx_usize = idx as usize;
                // Lock the mutex to safely modify the result vector
                if let Ok(mut result_lock) = result.lock() {
                    result_lock[idx_usize] =
                    result_lock[idx_usize] + l_vec[i];
                }
            }
        });

        // Unlock the mutex and return the result
        std::sync::Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }

    fn proj_right_challenges(&self, xr: &Vec<F>) -> Vec<F> {
        let xi_r = xi::xi_from_challenges::<F>(&xr);

        self.proj_right(&xi_r)
    }

    fn proj_right(&self, r_vec: &Vec<F>) -> Vec<F> {
        let m = self.shape.0;
        let n = self.shape.1;

        if n > r_vec.len() {
            panic!("The length of the right challenge is not enough");
        }

        (0..m).into_par_iter().map(|i|{
            let idx = self.data[i];
            if idx >= 0 {
                let idx = idx as usize;
                r_vec[idx]
            } else {
                F::zero()
            }
        }).collect::<Vec<F>>()
    }

    fn proj_lr(&self, xl: &Vec<F>, xr: &Vec<F>) -> F {
        let m = self.shape.0;
        let n = self.shape.1;
        
        if m > (1 << xl.len()) {
            panic!("The length of the left challenge is not enough");
        }

        if n > (1 << xr.len()) {
            panic!("The length of the right challenge is not enough");
        }

        let xi_l = xi::xi_from_challenges::<F>(&xl);

        let ar = self.proj_right_challenges(xr);
        let result = linear::inner_product::<F>(&ar, &xi_l);
        result
    }

    fn to_bool_neg(&self) -> DenseMatCM<I,F> {
        self.to_dense().to_bool_neg()
    }

    fn bit_decomposition(&self, k: usize)
    -> (DenseMatCM::<bool,F>,
        DenseMatCM<MyInt, F>,
        Vec<DenseMatCM<bool,F>>) {
        self.to_dense().bit_decomposition(k)
    }

    fn clamp(&self, k: usize) -> DenseMatCM<I,F> {
        self.to_dense().clamp(k)
    }
}


impl ShortInt for MyInt {}
impl ShortInt for bool {}

pub trait FromToI32 {
    fn from_myint(num: MyInt) -> Self;
    fn to_myint(&self) -> MyInt;
}

impl FromToI32 for MyInt {
    fn from_myint(num: MyInt) -> Self {
        num
    }
    fn to_myint(&self) -> MyInt {
        *self
    }
}

impl FromToI32 for bool {
    fn from_myint(num: MyInt) -> Self {
        num % 2 == 1
    }

    fn to_myint(&self) -> MyInt {
        if *self {
            1
        } else {
            0
        }
    }
}

pub trait MulAB<B> {
    fn mul_mix(self, b: B) -> Self;
}

impl MulAB<MyInt> for MyInt {
    fn mul_mix(self, b: MyInt) -> MyInt {
        self * b
    }
}

impl MulAB<bool> for MyInt {
    fn mul_mix(self, b: bool) -> MyInt {
        if b {
            self
        } else {
            0
        }
    }
}

impl MulAB<bool> for bool {
    fn mul_mix(self, b: bool) -> bool {
        if self && b {
            true
        } else {
            false
        }
    }
}

// Get the kappa-th bit in le format
pub trait GetBit {
    fn get_bit(&self, kappa: usize) -> bool;
}

impl GetBit for MyInt {
    fn get_bit(&self, kappa: usize) -> bool {
        let mask:MyInt = 1 << kappa;
        (self & mask) != 0
    }
}

impl GetBit for bool {
    fn get_bit(&self, kappa: usize) -> bool {
        if kappa == 0 {
            *self
        } else {
            false
        }
    }
}

// Reshape a vector into a col major matrix
pub fn reshape_vec<T> (
    vec: &Vec<T>,
    shape: (usize, usize)
) -> Vec<Vec<T>> 
where T: Clone,
{
    let m = shape.0;
    let n = shape.1;

    if m * n != vec.len() {
        panic!("The shape is not compatible with the vector length");
    }

    let mut mat = Vec::new();

    for i in 0..n {
        let row = vec[i*m..(i+1)*m].to_vec();
        mat.push(row);
    }

    mat
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_matop(){
        use ark_bls12_381::Fr;
        use ark_std::UniformRand;

        let rng = &mut ark_std::rand::thread_rng();


        let mut mat_kronecker =
        SparseKronecker::<MyInt, Fr>::new(
            (4,8), (8,16)
        );

        mat_kronecker.gen_rand(8);

        let dense_kronecker =
        mat_kronecker.to_dense();

        let xl =
        (0..5).map(|_| Fr::rand(rng)).collect();
        let xr =
        (0..7).map(|_| Fr::rand(rng)).collect();

        let v1 =
        mat_kronecker.proj_lr(
            &xl, &xr
        );

        let v2 =
        dense_kronecker.proj_lr(
            &xl, &xr
        );

        let la1 =
        mat_kronecker.proj_left_challenges(&xl);
        let la2 =
        dense_kronecker.proj_left_challenges(&xl);
        let xi_l = xi::xi_from_challenges::<Fr>(&xl);
        let la3 =
        mat_kronecker.proj_left(&xi_l);

        let ar1 =
        mat_kronecker.proj_right_challenges(&xr);
        let ar2 =
        dense_kronecker.proj_right_challenges(&xr);
        let xi_r = xi::xi_from_challenges::<Fr>(&xr);
        let ar3 =
        mat_kronecker.proj_right(&xi_r);

        assert_eq!(v1, v2);
        assert_eq!(la1, la2);
        assert_eq!(ar1, ar2);
        assert_eq!(la1, la3);
        assert_eq!(ar1, ar3);

        let x = [xr,xl].concat();
        let xi =
        xi::xi_from_challenges::<Fr>(&x);

        let vec =
        dense_kronecker.to_field_vec();
        let v3 =
        linear::inner_product::<Fr>(&vec, &xi);

        assert_eq!(v1, v3);

        // println!("v1: {:?}, v2: {:?}", v1, v2);

        let mut mat_dense_block =
        DenseBlockMat::<MyInt, Fr>::new(4, (4,8));

        mat_dense_block.gen_rand(8);

        let dense_block =
        mat_dense_block.to_dense();

        let xl_dense =
        (0..4).map(|_| Fr::rand(rng)).collect();
        let xr_dense =
        (0..3).map(|_| Fr::rand(rng)).collect();

        let v1_dense = mat_dense_block.proj_lr(
            &xl_dense, &xr_dense
        );

        let v2_dense = dense_block.proj_lr(
            &xl_dense, &xr_dense
        );

        assert_eq!(v1_dense, v2_dense);

        let la1_dense =
        mat_dense_block.proj_left_challenges(&xl_dense);
        let la2_dense =
        dense_block.proj_left_challenges(&xl_dense);

        let ar1_dense =
        mat_dense_block.proj_right_challenges(&xr_dense);
        let ar2_dense =
        dense_block.proj_right_challenges(&xr_dense);

        assert_eq!(la1_dense, la2_dense);
        assert_eq!(ar1_dense, ar2_dense);

        let mut mat_diag_block =
        DenseBlockMat::<MyInt, Fr>::new(4, (4,8));

        mat_diag_block.gen_rand(8);

        let diag_block =
        mat_diag_block.to_dense();

        let xl_diag =
        (0..4).map(|_| Fr::rand(rng)).collect();
        let xr_diag =
        (0..5).map(|_| Fr::rand(rng)).collect();

        let v1_diag =
        mat_diag_block.proj_lr(
            &xl_diag, &xr_diag
        );

        let v2_diag =
        diag_block.proj_lr(
            &xl_diag, &xr_diag
        );

        assert_eq!(v1_diag, v2_diag);

        let la1_diag =
        mat_dense_block.proj_left_challenges(&xl_diag);
        let la2_diag =
        dense_block.proj_left_challenges(&xl_diag);

        let ar1_diag =
        mat_dense_block.proj_right_challenges(&xr_diag);
        let ar2_diag =
        dense_block.proj_right_challenges(&xr_diag);

        assert_eq!(la1_diag, la2_diag);
        assert_eq!(ar1_diag, ar2_diag);

      

        let mut mat_rotation =
        RotationMatIndexFormat::<MyInt, Fr>::new(4, 8);

        mat_rotation.gen_rand(8);

        let dense_rotation =
        mat_rotation.to_dense();

        let xl_rot =
        (0..2).map(|_| Fr::rand(rng)).collect();
        let xr_rot =
        (0..3).map(|_| Fr::rand(rng)).collect();

        let v1_rot =
        mat_rotation.proj_lr(
            &xl_rot, &xr_rot
        );
        let v2_rot =
        dense_rotation.proj_lr(
            &xl_rot, &xr_rot
        );

        assert_eq!(v1_rot, v2_rot);

        let la1_rot =
        mat_rotation.proj_left_challenges(&xl_rot);
        let la2_rot =
        dense_rotation.proj_left_challenges(&xl_rot);

        let ar1_rot =
        mat_rotation.proj_right_challenges(&xr_rot);
        let ar2_rot =
        dense_rotation.proj_right_challenges(&xr_rot);

        let ar3_rot =
        mat_rotation.proj_right_via_boolean(&xr_rot);

        assert_eq!(la1_rot, la2_rot);
        assert_eq!(ar1_rot, ar2_rot);
        assert_eq!(ar1_rot, ar3_rot);

        println!("All passed");


        let mut mat_a =
        DenseMatCM::<MyInt, Fr>::new(4, 8);
        mat_a.gen_rand(8);

        let (mat_sign, mat_abs, mat_bool) =
        mat_a.bit_decomposition(8);

        let vec_a = mat_a.to_vec();
        let vec_sign = mat_sign.to_vec();
        let vec_abs = mat_abs.to_vec();
        let vec_bool_1 = mat_bool[1].to_vec();

        let (vec_sign_check,
            vec_abs_check,
            vec_bool_check
        ) = linear::split_into_boolean(&vec_a, 8);

        assert_eq!(vec_sign, vec_sign_check);
        assert_eq!(vec_abs, vec_abs_check);
        assert_eq!(vec_bool_1, vec_bool_check[1]);

        let mut mat_a =
        DenseMatCM::<MyInt, Fr>::new(4, 8);
        mat_a.gen_rand(17);

        let vec_a = mat_a.to_vec();

        let clamp =
        mat_a.clamp(8);

        let vec_a_clamp = clamp.to_vec();

        let (vec_sign_check,
            _,
            vec_bool_check
        ) = linear::split_into_boolean(&vec_a, 17);
        
        let mut cur_vec =vec![0; vec_a.len()];


        for kappa in 8..17 {
            let factor = 1 << (kappa - 8);
            let cur_bit_vec = vec_bool_check[kappa].clone();
            cur_vec =
            (0..vec_a.len()).into_par_iter().map(|i|{
                let cur_bit = cur_bit_vec[i];
                let cur_val = cur_vec[i];
                cur_val + cur_bit.to_myint() * factor
            }).collect::<Vec<MyInt>>();
        }

        let vec_clamp_check: Vec<MyInt> =
        cur_vec.par_iter()
        .zip(vec_sign_check.par_iter())
        .map(|(x, s)|{
            if *s {
                if *x > 128 {
                    1 << (8 -1)
                } else {
                    *x
                }
            } else {
                if *x > 128 {
                    -(1 << (8 -1))
                } else {
                    - *x
                }
            }
        }).collect();
        println!("vec_a: {:?}", vec_a);

        assert_eq!(vec_a_clamp, vec_clamp_check);

                // ==============================================
        // 添加 SparseMat 测试
        // ==============================================
        println!("=== Testing SparseMat Operations ===");

        // 创建 SparseMat (16x32 = power of 2 dimensions)
        let mut mat_sparse = SparseMat::<MyInt, Fr>::new(16, 32);

        // 手动添加一些测试数据
        let test_data = vec![
            (0, 0, 1),   (0, 5, 3),   (1, 2, -2),
            (2, 7, 4),   (3, 1, -1),  (5, 10, 2),
            (7, 15, 5),  (10, 20, -3), (12, 25, 6),
            (15, 31, 7)
        ];
        mat_sparse.set_data(test_data);

        // 转换为密集矩阵用于对比
        let dense_sparse = mat_sparse.to_dense();

        println!("SparseMat shape: {:?}", mat_sparse.get_shape());
        println!("SparseMat non-zero entries: {}", mat_sparse.data.len());

        // 测试向量转换
        let sparse_vec = mat_sparse.to_vec();
        let dense_vec = dense_sparse.to_vec();
        assert_eq!(sparse_vec, dense_vec, "Vector conversion failed");

        let sparse_field_vec = mat_sparse.to_field_vec();
        let dense_field_vec = dense_sparse.to_field_vec();
        assert_eq!(sparse_field_vec, dense_field_vec, "Field vector conversion failed");

        // 生成测试向量
        let xl_sparse: Vec<Fr> = (0..4).map(|_| Fr::rand(rng)).collect(); // log2(16) = 4
        let xr_sparse: Vec<Fr> = (0..5).map(|_| Fr::rand(rng)).collect(); // log2(32) = 5

        // 测试左投影
        let la1_sparse = mat_sparse.proj_left_challenges(&xl_sparse);
        let la2_sparse = dense_sparse.proj_left_challenges(&xl_sparse);
        assert_eq!(la1_sparse, la2_sparse, "Left challenges projection failed");

        let xi_l_sparse = xi::xi_from_challenges::<Fr>(&xl_sparse);
        let la3_sparse = mat_sparse.proj_left(&xi_l_sparse);
        assert_eq!(la1_sparse, la3_sparse, "Left projection consistency failed");

        // 测试右投影
        let ar1_sparse = mat_sparse.proj_right_challenges(&xr_sparse);
        let ar2_sparse = dense_sparse.proj_right_challenges(&xr_sparse);
        assert_eq!(ar1_sparse, ar2_sparse, "Right challenges projection failed");

        let xi_r_sparse = xi::xi_from_challenges::<Fr>(&xr_sparse);
        let ar3_sparse = mat_sparse.proj_right(&xi_r_sparse);
        assert_eq!(ar1_sparse, ar3_sparse, "Right projection consistency failed");

        // 测试双侧投影
        let v1_sparse = mat_sparse.proj_lr(&xl_sparse, &xr_sparse);
        let v2_sparse = dense_sparse.proj_lr(&xl_sparse, &xr_sparse);
        assert_eq!(v1_sparse, v2_sparse, "LR projection failed");

        // 测试标量乘法
        let scalar = Fr::from(3u64);
        let scaled_sparse = mat_sparse.scalar_mul(scalar);
        let scaled_dense = dense_sparse.scalar_mul(scalar);
        assert_eq!(scaled_sparse, scaled_dense, "Scalar multiplication failed");

        // 测试布尔否定
        let bool_neg_sparse = mat_sparse.to_bool_neg();
        let bool_neg_dense = dense_sparse.to_bool_neg();
        assert_eq!(bool_neg_sparse.to_vec(), bool_neg_dense.to_vec(), "Boolean negation failed");

        // 测试位分解
        let (sign_sparse, abs_sparse, bit_sparse) = mat_sparse.bit_decomposition(4);
        let (sign_dense, abs_dense, bit_dense) = dense_sparse.bit_decomposition(4);
        
        assert_eq!(sign_sparse.to_vec(), sign_dense.to_vec(), "Sign decomposition failed");
        assert_eq!(abs_sparse.to_vec(), abs_dense.to_vec(), "Abs decomposition failed");
        for (i, (bit_s, bit_d)) in bit_sparse.iter().zip(bit_dense.iter()).enumerate() {
            assert_eq!(bit_s.to_vec(), bit_d.to_vec(), "Bit decomposition {} failed", i);
        }

        // 测试截断
        let clamped_sparse = mat_sparse.clamp(2);
        let clamped_dense = dense_sparse.clamp(2);
        assert_eq!(clamped_sparse.to_vec(), clamped_dense.to_vec(), "Clamp operation failed");

        // 测试清空操作
        let mut test_sparse = mat_sparse.clone();
        test_sparse.clear();
        assert_eq!(test_sparse.data.len(), 0, "Clear operation failed");

        println!("✅ SparseMat basic operations tests passed");

    }
}