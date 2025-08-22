//! Parallel matrix operations for DenseMatFieldCM
//! 
use ark_ff::PrimeField;
use rayon::prelude::*;
use crate::utils::matdef::DenseMatFieldCM;

impl<F> DenseMatFieldCM<F>
where
    F: PrimeField + Send + Sync,
{
    // Parallel matrix addition
    // Returns self + other
    pub fn par_add(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F> {
        if self.shape != other.shape {
            panic!("Matrix dimensions must be the same for addition: {:?} vs {:?}", self.shape, other.shape);
        }

        let (m, n) = self.shape;
        
        // 按列并行计算矩阵加法
        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(self.data[j][i] + other.data[j][i]);
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel matrix subtraction
    // Returns self - other
    pub fn par_sub(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F> {
        if self.shape != other.shape {
            panic!("Matrix dimensions must be the same for subtraction: {:?} vs {:?}", self.shape, other.shape);
        }

        let (m, n) = self.shape;
        
        // 按列并行计算矩阵减法
        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(self.data[j][i] - other.data[j][i]);
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel matrix multiplication
    // Returns self * other
    pub fn par_mul(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F> {
        let (m, k) = self.shape;
        let (k2, n) = other.shape;

        if k != k2 {
            panic!("Matrix dimensions are not compatible for multiplication: {:?} * {:?}", self.shape, other.shape);
        }

        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = vec![F::zero(); m];
            for i in 0..m {
                for l in 0..k {
                    col[i] += self.data[l][i] * other.data[j][l];
                }
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel Hadamard product (element-wise multiplication)
    // Returns self ⊙ other (element-wise multiplication)
    pub fn par_hadamard(&self, other: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F> {
        if self.shape != other.shape {
            panic!("Matrix dimensions must be the same for Hadamard product: {:?} vs {:?}", self.shape, other.shape);
        }

        let (m, n) = self.shape;
        
        // 按列并行计算 Hadamard 乘积
        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(self.data[j][i] * other.data[j][i]);
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel scalar multiplication
    // Returns scalar * self
    pub fn par_scalar_mul(&self, scalar: F) -> DenseMatFieldCM<F> {
        let (m, n) = self.shape;
        
        // Col major
        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(scalar * self.data[j][i]);
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel transpose
    // Returns self^T
    pub fn par_transpose(&self) -> DenseMatFieldCM<F> {
        let (m, n) = self.shape;
        
        let result_data: Vec<Vec<F>> = (0..m).into_par_iter().map(|i| {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push(self.data[j][i]);
            }
            row
        }).collect();

        DenseMatFieldCM {
            shape: (n, m), // 转置后的形状
            data: result_data,
        }
    }

    // Parallel element-wise operation with a custom function
    // Applies the given function to each corresponding pair of elements
    pub fn par_elementwise_op<Op>(&self, other: &DenseMatFieldCM<F>, op: Op) -> DenseMatFieldCM<F>
    where
        Op: Fn(F, F) -> F + Send + Sync,
    {
        if self.shape != other.shape {
            panic!("Matrix dimensions must be the same for element-wise operation: {:?} vs {:?}", self.shape, other.shape);
        }

        let (m, n) = self.shape;
        
        let result_data: Vec<Vec<F>> = (0..n).into_par_iter().map(|j| {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(op(self.data[j][i], other.data[j][i]));
            }
            col
        }).collect();

        DenseMatFieldCM::from_data(result_data)
    }

    // Parallel reduction to compute matrix sum (sum of all elements)
    pub fn par_sum(&self) -> F {
        self.data.par_iter().map(|col| {
            col.par_iter().cloned().reduce(|| F::zero(), |a, b| a + b)
        }).reduce(|| F::zero(), |a, b| a + b)
    }

    // Parallel computation of matrix norm squared (Frobenius norm squared)
    pub fn par_norm_squared(&self) -> F {
        self.data.par_iter().map(|col| {
            col.par_iter().map(|&x| x * x).reduce(|| F::zero(), |a, b| a + b)
        }).reduce(|| F::zero(), |a, b| a + b)
    }

    // Parallel matrix-vector multiplication
    // Returns self * vec
    pub fn par_mv_mul(&self, vec: &[F]) -> Vec<F> {
        let (m, n) = self.shape;
        
        if n != vec.len() {
            panic!("Vector dimension {} does not match matrix columns {}", vec.len(), n);
        }

        (0..m).into_par_iter().map(|i| {
            let mut sum = F::zero();
            for j in 0..n {
                sum += self.data[j][i] * vec[j];
            }
            sum
        }).collect()
    }

    // Parallel vector-matrix multiplication
    // Returns vec^T * self
    pub fn par_vm_mul(&self, vec: &[F]) -> Vec<F> {
        let (m, n) = self.shape;
        
        if m != vec.len() {
            panic!("Vector dimension {} does not match matrix rows {}", vec.len(), m);
        }

        // 按列并行计算向量-矩阵乘法
        (0..n).into_par_iter().map(|j| {
            let mut sum = F::zero();
            for i in 0..m {
                sum += vec[i] * self.data[j][i];
            }
            sum
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;

    #[test]
    fn test_par_add() {
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_a.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(2u64)], // Column 0
            vec![BlsFr::from(3u64), BlsFr::from(4u64)], // Column 1
        ]);

        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_b.set_data(vec![
            vec![BlsFr::from(5u64), BlsFr::from(6u64)], // Column 0
            vec![BlsFr::from(7u64), BlsFr::from(8u64)], // Column 1
        ]);

        let result = mat_a.par_add(&mat_b);
        
        assert_eq!(result.data[0][0], BlsFr::from(6u64)); // 1 + 5
        assert_eq!(result.data[0][1], BlsFr::from(8u64)); // 2 + 6
        assert_eq!(result.data[1][0], BlsFr::from(10u64)); // 3 + 7
        assert_eq!(result.data[1][1], BlsFr::from(12u64)); // 4 + 8
    }

    #[test]
    fn test_par_mul() {
        // Test with actual matrix multiplication to verify column-major order
        // mat_a = [[1, 2], [3, 4]] (row-major representation)
        // In column-major: data[0] = [1, 3], data[1] = [2, 4]
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_a.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(3u64)], // Column 0: [1, 3]
            vec![BlsFr::from(2u64), BlsFr::from(4u64)], // Column 1: [2, 4]
        ]);

        // mat_b = [[5, 6], [7, 8]] (row-major representation)
        // In column-major: data[0] = [5, 7], data[1] = [6, 8]
        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_b.set_data(vec![
            vec![BlsFr::from(5u64), BlsFr::from(7u64)], // Column 0: [5, 7]
            vec![BlsFr::from(6u64), BlsFr::from(8u64)], // Column 1: [6, 8]
        ]);

        let result = mat_a.par_mul(&mat_b);
        
        // Expected result: [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // In column-major: data[0] = [19, 43], data[1] = [22, 50]
        assert_eq!(result.data[0][0], BlsFr::from(19u64)); // (1*5 + 2*7) = 19
        assert_eq!(result.data[0][1], BlsFr::from(43u64)); // (3*5 + 4*7) = 43
        assert_eq!(result.data[1][0], BlsFr::from(22u64)); // (1*6 + 2*8) = 22
        assert_eq!(result.data[1][1], BlsFr::from(50u64)); // (3*6 + 4*8) = 50
        
        // Verify shape is correct
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_par_mul_rectangular() {
        // Test with rectangular matrices: 2x3 * 3x2 = 2x2
        // mat_a = [[1, 2, 3], [4, 5, 6]] (row-major representation)
        // In column-major: data[0] = [1, 4], data[1] = [2, 5], data[2] = [3, 6]
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(2, 3);
        mat_a.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(4u64)], // Column 0: [1, 4]
            vec![BlsFr::from(2u64), BlsFr::from(5u64)], // Column 1: [2, 5]
            vec![BlsFr::from(3u64), BlsFr::from(6u64)], // Column 2: [3, 6]
        ]);

        // mat_b = [[7, 8], [9, 10], [11, 12]] (row-major representation)
        // In column-major: data[0] = [7, 9, 11], data[1] = [8, 10, 12]
        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(3, 2);
        mat_b.set_data(vec![
            vec![BlsFr::from(7u64), BlsFr::from(9u64), BlsFr::from(11u64)], // Column 0: [7, 9, 11]
            vec![BlsFr::from(8u64), BlsFr::from(10u64), BlsFr::from(12u64)], // Column 1: [8, 10, 12]
        ]);

        let result = mat_a.par_mul(&mat_b);
        
        // Expected result: [[1, 2, 3], [4, 5, 6]] * [[7, 8], [9, 10], [11, 12]] = [[58, 64], [139, 154]]
        // Calculation:
        // result[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // result[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // result[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // result[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(result.data[0][0], BlsFr::from(58u64));
        assert_eq!(result.data[0][1], BlsFr::from(139u64));
        assert_eq!(result.data[1][0], BlsFr::from(64u64));
        assert_eq!(result.data[1][1], BlsFr::from(154u64));
        
        // Verify shape is correct (2x3 * 3x2 = 2x2)
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_par_hadamard() {
        let mut mat_a = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_a.set_data(vec![
            vec![BlsFr::from(2u64), BlsFr::from(3u64)], // Column 0
            vec![BlsFr::from(4u64), BlsFr::from(5u64)], // Column 1
        ]);

        let mut mat_b = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat_b.set_data(vec![
            vec![BlsFr::from(6u64), BlsFr::from(7u64)], // Column 0
            vec![BlsFr::from(8u64), BlsFr::from(9u64)], // Column 1
        ]);

        let result = mat_a.par_hadamard(&mat_b);
        
        assert_eq!(result.data[0][0], BlsFr::from(12u64)); // 2 * 6
        assert_eq!(result.data[0][1], BlsFr::from(21u64)); // 3 * 7
        assert_eq!(result.data[1][0], BlsFr::from(32u64)); // 4 * 8
        assert_eq!(result.data[1][1], BlsFr::from(45u64)); // 5 * 9
    }

    #[test]
    fn test_par_transpose() {
        let mut mat = DenseMatFieldCM::<BlsFr>::new(2, 3);
        mat.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(4u64)], // Column 0
            vec![BlsFr::from(2u64), BlsFr::from(5u64)], // Column 1
            vec![BlsFr::from(3u64), BlsFr::from(6u64)], // Column 2
        ]);

        let result = mat.par_transpose();
        
        assert_eq!(result.shape, (3, 2)); // Transposed shape
        assert_eq!(result.data[0][0], BlsFr::from(1u64));
        assert_eq!(result.data[0][1], BlsFr::from(2u64));
        assert_eq!(result.data[0][2], BlsFr::from(3u64));
        assert_eq!(result.data[1][0], BlsFr::from(4u64));
        assert_eq!(result.data[1][1], BlsFr::from(5u64));
        assert_eq!(result.data[1][2], BlsFr::from(6u64));
    }

    #[test]
    fn test_par_mv_mul() {
        let mut mat = DenseMatFieldCM::<BlsFr>::new(2, 2);
        mat.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(3u64)], // Column 0
            vec![BlsFr::from(2u64), BlsFr::from(4u64)], // Column 1
        ]);

        let vec = vec![BlsFr::from(5u64), BlsFr::from(6u64)];
        let result = mat.par_mv_mul(&vec);
        
        // [1 2] * [5] = [17]
        // [3 4]   [6]   [39]
        assert_eq!(result[0], BlsFr::from(17u64)); // 1*5 + 2*6
        assert_eq!(result[1], BlsFr::from(39u64)); // 3*5 + 4*6
    }
}
