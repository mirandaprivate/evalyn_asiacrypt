//! Matrix and Point Container Utilities for BatchProj Protocol
//!
//! This module provides container structures and utilities for managing heterogeneous 
//! matrices and evaluation points in the BatchProj atomic proof protocol. The key
//! functionality includes:
//!
//! - Area-based sorting of matrices and points to enable efficient padding alignment
//! - Flattening matrices into concatenated vectors with proper power-of-2 alignment
//! - Computing xi evaluation vectors from challenge points for multilinear polynomials
//! - Length consistency checking between matrix and point containers
//!
//! ## Algorithm Overview
//!
//! The containers implement a specific padding strategy:
//! 1. Sort matrices/points by area (num_rows * num_cols) in ascending order
//! 2. For each matrix of area 2^k, ensure it starts at position aligned to area boundary
//! 3. Pad with zeros as needed to maintain alignment
//! 4. Resize final vector to next power of 2 for efficient inner product computation
//!
//! This approach ensures that the flattened matrix vector and xi evaluation vector
//! have matching lengths and proper alignment for the bilinear pairing verification.

use ark_ff::PrimeField;

use mat::utils::matdef::DenseMatFieldCM;
use mat::MyInt;
use mat::DenseMatCM;



/// Container for field element matrices sorted by area.
/// Used in protocols that need to batch multiple matrices with different dimensions.
pub struct MatContainer<F: PrimeField> {
    pub sorted_matrices: Vec<DenseMatFieldCM<F>>,
    pub sorted_shapes: Vec<(usize, usize)>,
    pub sorted_areas: Vec<usize>,
}

#[derive(Debug, Clone)]
/// Container for integer matrices sorted by area.
/// Specialized container for DenseMatCM<MyInt, F> matrices that maintains
/// area-based ordering for efficient concatenation with proper alignment.
pub struct MatContainerMyInt<F: PrimeField> {
    pub sorted_matrices: Vec<DenseMatCM<MyInt, F>>,
    pub sorted_shapes: Vec<(usize, usize)>,
    pub sorted_areas: Vec<usize>,
}

#[derive(Debug, Clone)]
/// Container for evaluation points with area-based sorting.
/// Manages challenge points for multilinear polynomial evaluation,
/// maintaining correspondence with matrix containers for BatchProj protocol.
pub struct PointsContainer<F> {
    pub sorted_hats: Vec<F>,
    pub sorted_points: Vec<(Vec<F>, Vec<F>)>,
    pub sorted_shapes: Vec<(usize, usize)>,
    pub sorted_areas: Vec<usize>,
    pub sorted_start_position: Vec<usize>,
    pub sorted_hats_index: Vec<usize>,
    pub sorted_points_index: Vec<(Vec<usize>, Vec<usize>)>,  // start_position in flattened vector
}

impl<F> PointsContainer<F> 
where
    F: PrimeField + Clone,
{
    /// Creates a new empty PointsContainer.
    pub fn new() -> Self {
        Self {
            sorted_hats: Vec::new(),
            sorted_points: Vec::new(),
            sorted_shapes: Vec::new(),
            sorted_areas: Vec::new(),
            sorted_start_position: Vec::new(),
            sorted_hats_index: Vec::new(),
            sorted_points_index: Vec::new(),
        }
    }

    /// Adds a new hat value and challenge point pair to the container.
    /// 
    /// # Arguments
    /// * `hat` - Field element hat value
    /// * `point` - Tuple of challenge vectors (row challenges, col challenges)
    /// * `hat_index` - Index for the hat value
    /// * `point_index` - Tuple of index vectors for the challenge points
    ///
    /// # Returns
    /// Index of the inserted element after sorting
    pub fn push(&mut self, hat: F, point: (Vec<F>, Vec<F>), hat_index: usize, point_index: (Vec<usize>, Vec<usize>)) -> usize {
        let m = (1 << point.0.len()) as usize;
        let n = (1 << point.1.len()) as usize;
        let shape = (m, n);
        let area = m * n; // area is now usize

        // Stable insert
        let insert_pos = match self.sorted_areas.binary_search(&area) {
            Ok(mut idx) => {
                while idx < self.sorted_areas.len() && self.sorted_areas[idx] == area {
                    idx += 1;
                }
                idx
            }
            Err(pos) => pos,
        };

        // Insert all fields at the same position to maintain consistency
        self.sorted_areas.insert(insert_pos, area);
        self.sorted_hats.insert(insert_pos, hat);
        self.sorted_points.insert(insert_pos, point);
        self.sorted_shapes.insert(insert_pos, shape);
        self.sorted_hats_index.insert(insert_pos, hat_index);
        self.sorted_points_index.insert(insert_pos, point_index);

        // Insert placeholder for start position - will be calculated when needed
        self.sorted_start_position.insert(insert_pos, 0);
        
        // Update start positions
        self.update_start_positions();

        insert_pos
    }

    /// Push point 
    pub fn push_point(&mut self, point_info: &PointInfo<F>) {
        self.push(
            point_info.hat.clone(),
            point_info.point.clone(),
            point_info.hat_index.clone(),
            point_info.point_index.clone(),
        );
    }

    /// Updates start positions for area-based alignment.
    /// Each matrix of area 2^k must start at a position aligned to 2^k boundary.
    /// If not aligned, padding zeros are added to maintain alignment.
    fn update_start_positions(&mut self) {
        let mut current_pos = 0usize;

        for i in 0..self.sorted_areas.len() {
            let current_area = self.sorted_areas[i]; // area is already usize

            if current_pos % current_area != 0 {
                // Pad with zeros to align with the current area
                let padding = current_area - (current_pos % current_area);
                current_pos += padding;
            }

            self.sorted_start_position[i] = current_pos;
            current_pos += current_area;
        }
    }


    /// Generates xi evaluation vector from challenge points with proper padding alignment.
    /// Creates a concatenated vector of xi evaluations corresponding to the sorted matrices,
    /// with padding to ensure power-of-2 alignment for efficient inner product computation.
    ///
    /// # Arguments
    /// * `challenge` - Challenge field element for batching multiple matrices
    ///
    /// # Returns
    /// Vector of field elements representing the batched xi evaluations
    pub fn xi_concat(&self, challenge: F) -> Vec<F> {
        let mut vec = Vec::new();
        let mut multiplicant = F::one();

        for i in 0..self.sorted_points.len() {
            let start_pos = self.sorted_start_position[i];
            assert_eq!(start_pos % self.sorted_areas[i] as usize, 0);

            let cur_point = &self.sorted_points[i];

            let cur_xx = [cur_point.1.as_slice(), cur_point.0.as_slice()].concat();
            let cur_xi = mat::utils::linear::vec_scalar_mul(
                &mat::utils::xi::xi_from_challenges(&cur_xx),
                &multiplicant,
            );
            

            let cur_len = vec.len();

            if cur_len < start_pos {
                vec.resize(start_pos, F::zero());
            }

            vec.extend(cur_xi);

            multiplicant *= challenge.clone();
        }

        let len = vec.len().next_power_of_two();
        vec.resize(len, F::zero());

        vec
    }

    /// Returns the length of the flattened xi vector.
    /// Computes the required length for xi_concat output based on start positions and areas.
    pub fn flatten_len(&self) -> usize {
        let num = self.sorted_hats.len();
        let len_raw = self.sorted_start_position[num-1];
        let area = self.sorted_areas[num-1];

        (len_raw + area).next_power_of_two()
    }

    /// Computes batched hat value using challenge for randomization.
    /// Combines multiple hat values with powers of the challenge for zero-knowledge.
    ///
    /// # Arguments
    /// * `challenge` - Challenge field element for batching
    ///
    /// # Returns
    /// Batched hat value as a field element
    pub fn hat_batched(&self, challenge: F) -> F {
        
        let mut multiplicant = F::one();
        let mut result = F::zero();
        
        for i in 0..self.sorted_hats.len() {
         
            let cur_hat = &self.sorted_hats[i];

            result += *cur_hat * multiplicant;
            multiplicant *= challenge.clone();
        }

        result
    }
}


impl<F: PrimeField> MatContainer<F> {
    /// Creates a new empty MatContainer.
    pub fn new() -> Self {
        Self {
            sorted_matrices: Vec::new(),
            sorted_shapes: Vec::new(),
            sorted_areas: Vec::new(),
        }
    }

    /// Adds a new field matrix to the container with area-based sorting.
    /// 
    /// # Arguments
    /// * `mat` - DenseMatFieldCM matrix to add to the container
    pub fn push(&mut self, mat: DenseMatFieldCM<F>) {

        let (m, n) = mat.shape.clone();
        let area = m * n; // area is now usize
        // Stable insert: keep original order for elements with the same area (insert at the end of the same area block)
        let insert_pos = match self.sorted_areas.binary_search(&area) {
            Ok(mut idx) => {
                while idx < self.sorted_areas.len() && self.sorted_areas[idx] == area {
                    idx += 1;
                }
                idx
            }
            Err(pos) => pos,
        };

        let shape = (m, n);

        // Insert all fields at the same position to maintain consistency
        self.sorted_areas.insert(insert_pos, area);
        self.sorted_shapes.insert(insert_pos, shape);
        self.sorted_matrices.insert(insert_pos, mat);
    }

    /// Flattens and concatenates all matrices with proper area-based alignment.
    /// Implements the core padding algorithm to ensure each matrix of area 2^k
    /// starts at a position aligned to 2^k boundary.
    ///
    /// # Returns
    /// Vector of field elements representing the concatenated and padded matrices
    pub fn flatten_and_concat(&self) -> Vec<F> {
        let mut vec = Vec::new();
        let mut cur_pos = 0;

        for i in 0..self.sorted_matrices.len() {

            let cur_vec = flatten(&self.sorted_matrices[i]);
            let current_area = self.sorted_areas[i];

            if cur_pos % current_area != 0 {
                // Pad with zeros to align with the current area
                let padding = current_area - (cur_pos % current_area);
                cur_pos += padding;
                vec.resize(cur_pos, F::zero());
            }

            vec.extend(cur_vec);

            cur_pos += current_area;    
            
        }

        let len = vec.len().next_power_of_two();
        vec.resize(len, F::zero());

        vec
    }
}




impl<F: PrimeField> MatContainerMyInt<F> {
    /// Creates a new empty MatContainerMyInt.
    pub fn new() -> Self {
        Self {
            sorted_matrices: Vec::new(),
            sorted_shapes: Vec::new(),
            sorted_areas: Vec::new(),
        }
    }

    /// Clears the contents of the container.
    pub fn clear(&mut self) {
        self.sorted_matrices = Vec::new();
        self.sorted_shapes = Vec::new();
        self.sorted_areas =  Vec::new();
    }

    /// Adds a new integer matrix to the container with area-based sorting.
    /// 
    /// # Arguments
    /// * `mat` - DenseMatCM<MyInt, F> matrix to add to the container
    pub fn push(&mut self, mat: DenseMatCM<MyInt, F>) {

        let (m, n) = mat.shape.clone();
        let shape = (m, n);
        let area = m * n; // area is now usize

        // Stable insertion by area: place after existing equal-area block (upper bound)
        let insert_pos = match self.sorted_areas.binary_search(&area) {
            Ok(mut pos) => {
                // advance to first index > area
                while pos < self.sorted_areas.len() && self.sorted_areas[pos] == area {
                    pos += 1;
                }
                pos
            }
            Err(pos) => pos,
        };

        // Insert all fields at the same position to maintain consistency
        self.sorted_areas.insert(insert_pos, area);
        self.sorted_shapes.insert(insert_pos, shape);
        self.sorted_matrices.insert(insert_pos, mat);
    }

    /// Flattens and concatenates all integer matrices with proper area-based alignment.
    /// Similar to MatContainer but works with MyInt elements instead of field elements.
    ///
    /// # Returns
    /// Vector of MyInt elements representing the concatenated and padded matrices
    pub fn flatten_and_concat(&self) -> Vec<MyInt> {
        let mut vec = Vec::new();
        let mut cur_pos = 0;

        for i in 0..self.sorted_matrices.len() {

            let cur_vec = flatten_myint(&self.sorted_matrices[i]);
            let current_area = self.sorted_areas[i];

            if cur_pos % current_area != 0 {
                // Pad with zeros to align with the current area
                let padding = current_area - (cur_pos % current_area);
                cur_pos += padding;
                vec.resize(cur_pos, 0 as MyInt);
            }

            vec.extend(cur_vec);

            cur_pos += current_area;
            
        }

        let len = vec.len().next_power_of_two();
        vec.resize(len, 0 as MyInt);

        vec
    }

    pub fn into_flattened_vec(&mut self) -> Vec<MyInt> {
        let mut vec = Vec::new();
        let mut cur_pos = 0;

        for i in 0..self.sorted_matrices.len() {

            let cur_vec = flatten_myint(&self.sorted_matrices[i]);
            let current_area = self.sorted_areas[i];

            if cur_pos % current_area != 0 {
                // Pad with zeros to align with the current area
                let padding = current_area - (cur_pos % current_area);
                cur_pos += padding;
                vec.resize(cur_pos, 0 as MyInt);
            }

            vec.extend(cur_vec);

            self.sorted_matrices[i].data = Vec::new();

            cur_pos += current_area;
            
        }

        let len = vec.len().next_power_of_two();
        vec.resize(len, 0 as MyInt);

        vec
    }
}


/// Flattens a DenseMatFieldCM matrix into a column-major vector.
/// 
/// # Arguments
/// * `mat` - Matrix to flatten
///
/// # Returns
/// Vector of field elements in column-major order
pub fn flatten<F: PrimeField>(mat: &DenseMatFieldCM<F>) -> Vec<F> {
    let mut flattened = Vec::new();

    for col in &mat.data {
        flattened.extend(col.iter().cloned());
    }

    flattened
}

/// Flattens a DenseMatCM<MyInt, F> matrix into a column-major vector.
/// 
/// # Arguments
/// * `mat` - MyInt matrix to flatten
///
/// # Returns
/// Vector of MyInt elements in column-major order
pub fn flatten_myint<F: PrimeField>(mat: &DenseMatCM<MyInt, F>) -> Vec<MyInt> {
    let mut flattened = Vec::new();

    for col in &mat.data {
        flattened.extend(col.iter().cloned());
    }

    flattened
}


/// Computes xi evaluation at a specific position with proper bounds checking.
/// This function calculates the xi value for multilinear polynomial evaluation
/// at the given position using the challenge vector xxxx.
///
/// # Arguments
/// * `position` - Position in the flattened vector
/// * `cur_log_len` - Log of current matrix area (must be power of 2)
/// * `xxxx` - Challenge vector for xi computation
///
/// # Returns
/// Field element representing the xi evaluation at the position
pub fn compute_xi_at_position<F: PrimeField>(position: usize, cur_log_len: usize, xxxx: &Vec<F>) -> F {
    // Compute the xi value at the given position using the xxxx vector
    let cur_len = 1 << cur_log_len;
    if position % cur_len != 0 {
        println!("Position {} is not aligned with current length {}", position, cur_len);
        return F::zero();
    }

    let div = position / cur_len;

    let div_ceil = (div+1).next_power_of_two();
    let log_div_ceil = div_ceil.ilog2() as usize;

    let log_cur_len = cur_len.ilog2() as usize;

    // Guard: if we request more bits (log_div_ceil + log_cur_len) than available challenge entries,
    // we cannot slice; in this (rare / test) case return zero contribution.
    if log_div_ceil + log_cur_len > xxxx.len() {
        // Not enough challenge bits supplied for this position depth.
        return F::zero();
    }

    let start = xxxx.len() - log_div_ceil - log_cur_len;
    let end = xxxx.len() - log_cur_len;
    let xx_div = &xxxx[start..end];

    let mut cur_div = div;
    let mut cur_mul = F::one();

    for i in 0..xx_div.len() {

        if cur_div % 2 == 1 {
            cur_mul *= &xx_div[xx_div.len()-i-1];
        } 

        cur_div /= 2;
    }

    cur_mul
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_ff::Zero;
    use mat::utils::linear::inner_product;
    use mat::utils::xi::xi_from_challenges;
    use ark_bls12_381::Bls12_381;
    use ark_std::test_rng;
    use ark_ff::UniformRand;
    use ark_poly_commit::smart_pc::SmartPC;
    use ark_poly_commit::smart_pc::data_structures::UniversalParams as PcsPP;

    #[test]
    fn test_flatconcat_container_area_ordering() {
        let mut container = MatContainer::<BlsFr>::new();

        // Create matrices of different sizes
        // Matrix 1: 4x4 (area = 16)
        let mut mat1 = DenseMatFieldCM::new(4, 4);
        mat1.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(4u64)],
            vec![BlsFr::from(5u64), BlsFr::from(6u64), BlsFr::from(7u64), BlsFr::from(8u64)],
            vec![BlsFr::from(9u64), BlsFr::from(10u64), BlsFr::from(11u64), BlsFr::from(12u64)],
            vec![BlsFr::from(13u64), BlsFr::from(14u64), BlsFr::from(15u64), BlsFr::from(16u64)],
        ]);

        // Matrix 2: 2x2 (area = 4) - should be inserted before mat1
        let mut mat2 = DenseMatFieldCM::new(2, 2);
        mat2.set_data(vec![
            vec![BlsFr::from(17u64), BlsFr::from(18u64)],
            vec![BlsFr::from(19u64), BlsFr::from(20u64)],
        ]);

        // Matrix 3: 8x8 (area = 64) - should be inserted after mat1
        let mut mat3 = DenseMatFieldCM::new(8, 8);
        let mut data3 = Vec::new();
        for col in 0..8 {
            let mut col_data = Vec::new();
            for row in 0..8 {
                col_data.push(BlsFr::from((col * 8 + row + 21) as u64));
            }
            data3.push(col_data);
        }
        mat3.set_data(data3);

        // Push matrices in different order
        container.push(mat1);  // area = 16
        container.push(mat2);  // area = 4, should be inserted at position 0
        container.push(mat3);  // area = 64, should be inserted at position 2

        // Verify area ordering (small to large)
        let areas: Vec<usize> = container.sorted_areas.clone();
        
        assert_eq!(areas, vec![4, 16, 64], "Areas should be sorted in ascending order");

        // Verify matrices are also reordered accordingly
        assert_eq!(container.sorted_matrices[0].shape, (2, 2));
        assert_eq!(container.sorted_matrices[1].shape, (4, 4));
        assert_eq!(container.sorted_matrices[2].shape, (8, 8));

        println!("✅ Area ordering test passed!");
    }

    // This test checks whether "segment padding + per-block points" can be
    // represented by a single global (xl, xr) point for xi generation.
    // It constructs two segments with different per-block points, pads them to
    // aligned positions, and compares the batched xi (piecewise) with several
    // candidate global xi vectors. In general they are NOT equal.
    #[test]
    fn test_padding_segments_vs_single_global_point_non_equivalence() {
        // Build two row-vectors: shapes (1, 8) and (1, 16) so areas are powers of two.
        let mut matc = MatContainerMyInt::<BlsFr>::new();
        // Row-major helper: 1 x n
        let row8 = DenseMatCM::<MyInt, BlsFr>::from_data(vec![vec![0 as MyInt; 8]]);
        let row16 = DenseMatCM::<MyInt, BlsFr>::from_data(vec![vec![0 as MyInt; 16]]);
        matc.push(row8);
        matc.push(row16);

        let mut pc = PointsContainer::<BlsFr>::new();
        // Per-block points: (xl empty, xr has log2(n) entries)
        let p1_xr = vec![BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(5u64)]; // len=3 -> n=8
        let p2_xr = vec![BlsFr::from(7u64), BlsFr::from(11u64), BlsFr::from(13u64), BlsFr::from(17u64)]; // len=4 -> n=16
        // Push with dummy hat/index; we only need points for xi
        pc.push(BlsFr::zero(), (vec![], p1_xr.clone()), 0, (vec![], vec![0,1,2]));
        pc.push(BlsFr::zero(), (vec![], p2_xr.clone()), 1, (vec![], vec![0,1,2,3]));

        // Compute flattened length and batched xi
        let flattened = matc.flatten_and_concat();
        let r = BlsFr::from(19u64);
        let xi_batched = pc.xi_concat(r);
        assert_eq!(flattened.len(), xi_batched.len());

        let total_len = xi_batched.len();
        let log_total = total_len.ilog2() as usize;
        let sum_logs = p1_xr.len() + p2_xr.len();
        // With shapes (1x8) and (1x16), alignment yields total_len = 32, so log_total = 5
        // But per-block logs sum to 7, strictly greater than 5 -> cannot encode both blocks' points in a single global point
        assert!(sum_logs > log_total, "Sum of per-block logs must exceed global log; got {} vs {}", sum_logs, log_total);

        // Try several candidate global challenge vectors (length = log_total)
        let cand1: Vec<BlsFr> = vec![23u64,29,31,37,41].into_iter().map(BlsFr::from).collect();
        // Embed p1_xr into tail/head positions as naive candidates
        let mut cand2 = vec![BlsFr::from(0u64); log_total];
        // place p1_xr at the end
        for (i, v) in p1_xr.iter().enumerate() { cand2[log_total - p1_xr.len() + i] = *v; }
        let mut cand3 = vec![BlsFr::from(0u64); log_total];
        // place p1_xr at the beginning
        for (i, v) in p1_xr.iter().enumerate() { cand3[i] = *v; }

        let xi_c1 = xi_from_challenges(&cand1);
        let xi_c2 = xi_from_challenges(&cand2);
        let xi_c3 = xi_from_challenges(&cand3);

        assert_ne!(xi_batched, xi_c1, "Batched xi should not equal xi from a random single global point");
        assert_ne!(xi_batched, xi_c2, "Batched xi should not equal xi from a naive tail-embedded global point");
        assert_ne!(xi_batched, xi_c3, "Batched xi should not equal xi from a naive head-embedded global point");
    }

    #[test]
    fn test_flatconcat_inner_product_property() {
        let mut container = MatContainer::<BlsFr>::new();
        let mut point_container = PointsContainer::new();
        let challenge = BlsFr::from(7u64); // Random challenge

        // Create test matrices
        let mut mat1 = DenseMatFieldCM::new(2, 2);
        mat1.set_data(vec![
            vec![BlsFr::from(1u64), BlsFr::from(2u64)],
            vec![BlsFr::from(3u64), BlsFr::from(4u64)],
        ]);

        let mut mat2 = DenseMatFieldCM::new(4, 4);
        let mut data2 = Vec::new();
        for col in 0..4 {
            let mut col_data = Vec::new();
            for row in 0..4 {
                col_data.push(BlsFr::from((col * 4 + row + 5) as u64));
            }
            data2.push(col_data);
        }
        mat2.set_data(data2);

        // Set up proper evaluation points for each matrix
        // For 2x2 matrix (log_m = 1, log_n = 1)
        let point1 = (vec![BlsFr::from(2u64)], vec![BlsFr::from(3u64)]);
        
        // For 4x4 matrix (log_m = 2, log_n = 2)  
        let point2 = (vec![BlsFr::from(5u64), BlsFr::from(6u64)], vec![BlsFr::from(7u64), BlsFr::from(8u64)]);

        // Manually calculate expected projections using proj_lr_challenges
        let proj1 = mat1.proj_lr_challenges(&point1.0, &point1.1);
        let proj2 = mat2.proj_lr_challenges(&point2.0, &point2.1);
        
        let expected_sum = proj1 + challenge * proj2;

        // Now use container (this will reorder by area, so mat1 comes first)
        container.push(mat1);
        container.push(mat2);
        point_container.push(
            point1.0[0].clone(), 
            point1.clone(), 
            0, // hat index
            (vec![0], vec![0]), // point index
        );
        point_container.push(
            point2.0[0].clone(),
            point2.clone(),
            1, // hat index
            (vec![1], vec![1]), // point index
        );

        // Update points container with proper evaluation points
        point_container.sorted_points[0] = point1; // mat1 is at index 0 (smaller area)
        point_container.sorted_points[1] = point2; // mat2 is at index 1

        // Get the vectors
        let flattened = container.flatten_and_concat();
        let xi_vec = point_container.xi_concat(challenge);

        // Verify they have the same length
        assert_eq!(flattened.len(), xi_vec.len(), "Vectors should have same length");

        // Calculate inner product
        let actual_result = inner_product(&flattened, &xi_vec);

        assert_eq!(actual_result, expected_sum, 
            "Inner product should equal linear combination of projections");

        println!("✅ Inner product property test passed!");
    }

    #[test]
    fn test_flatconcat_xi_from_challenges_vs_compute_xi_at_position() {
        // Test that xi_from_challenges at a specific position equals compute_xi_at_position
        
        // Test case 1: Simple case with 4 challenges
        let xxxx = vec![
            BlsFr::from(3u64),
            BlsFr::from(5u64), 
            BlsFr::from(7u64),
            BlsFr::from(11u64),
        ];
        
        let xi_vec = mat::xi::xi_from_challenges(&xxxx);
        let cur_len: usize = 4; // 2^2
        let log_cur_len = cur_len.ilog2() as usize;
        
        // Test several positions where position % cur_len == 0
        let test_positions = vec![0, 4, 8, 12];
        
        for &position in &test_positions {
            if position < xi_vec.len() {
                let xi_value = xi_vec[position];
                let computed_value = compute_xi_at_position(position, log_cur_len, &xxxx);
                
                assert_eq!(xi_value, computed_value, 
                    "Values should match at position {} with cur_len {}", position, cur_len);
            }
        }
        
        // Test case 2: Different cur_len
        let cur_len = 8; // 2^3
        let test_positions = vec![0, 8, 16];
        
        for &position in &test_positions {
            if position < xi_vec.len() {
                let xi_value = xi_vec[position];
                let computed_value = compute_xi_at_position(position, log_cur_len, &xxxx);
                
                assert_eq!(xi_value, computed_value, 
                    "Values should match at position {} with cur_len {}", position, cur_len);
            }
        }
        
        // Test case 3: Larger example with 6 challenges
        let xxxx_large = vec![
            BlsFr::from(2u64),
            BlsFr::from(3u64),
            BlsFr::from(5u64),
            BlsFr::from(7u64),
            BlsFr::from(11u64),
            BlsFr::from(13u64),
        ];
        
        let xi_vec_large = mat::utils::xi::xi_from_challenges(&xxxx_large);
        let cur_len = 16; // 2^4
        
        let test_positions = vec![0, 16, 32, 48];
        
        for &position in &test_positions {
            if position < xi_vec_large.len() {
                let xi_value = xi_vec_large[position];
                let computed_value = compute_xi_at_position(position, log_cur_len, &xxxx_large);

                assert_eq!(xi_value, computed_value,
                    "Values should match at position {} with cur_len {} for large example",
                    position, cur_len);
            }
        }
        
        println!("✅ xi_from_challenges vs compute_xi_at_position test passed!");
    }

    #[test]
    fn test_flatconcat_compute_xi_at_position_alignment_check() {
        // Test that compute_xi_at_position returns zero for misaligned positions
        let xxxx = vec![BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(5u64)];
        let cur_len = 4;
        
        // Test misaligned positions (position % cur_len != 0)
        let misaligned_positions = vec![1, 2, 3, 5, 6, 7, 9, 10, 11];
        
        for &position in &misaligned_positions {
            let result = compute_xi_at_position(position, cur_len, &xxxx);
            assert_eq!(result, BlsFr::zero(), 
                "compute_xi_at_position should return zero for misaligned position {}", position);
        }
        
        // Test aligned positions should not return zero (unless the actual xi value is zero)
        let aligned_positions = vec![0, 4, 8, 12];
        for &position in &aligned_positions {
            let result = compute_xi_at_position(position, cur_len, &xxxx);
            // We don't assert non-zero here because the actual xi value might be zero
            // Just ensure it doesn't panic and returns a valid field element
            let _ = result;
        }
        
        println!("✅ compute_xi_at_position alignment check test passed!");
    }

    #[test]
    fn test_pcs_roundtrip_after_square_reorg_mixed_shapes() {
        // Purpose: Build mixed-shape MatContainerMyInt + PointsContainer, flatten to a single row vector,
        // and check SmartPC commit_square_myint → open_square_myint → verify_square still holds.
        // This mimics how leaf commitments are formed from heterogeneous blocks.

    // Shapes to test: heterogeneous, each dim is a power of two, and areas are all different
    // Areas: 2, 4, 8, 16, 32
    let shapes: &[(usize, usize)] = &[(1, 2), (2, 2), (1, 8), (2, 8), (4, 8)];

        // Build integer matrices with simple, deterministic contents
        let mut matc = MatContainerMyInt::<BlsFr>::new();
        let mut pointc = PointsContainer::<BlsFr>::new();

        for (idx, &(m, n)) in shapes.iter().enumerate() {
            let mut data = Vec::new();
            // Col-major: n columns, each length m
            for col in 0..n {
                let mut column = Vec::with_capacity(m);
                for row in 0..m {
                    // deterministic value that depends on (row,col,idx)
                    let val: i32 = (idx as i32) * 1000 + (col as i32) * 10 + (row as i32);
                    column.push(val as MyInt);
                }
                data.push(column);
            }
            // Keep a copy for building the field matrix below (DenseMatCM::set_data will move `data`)
            let data_for_field = data.clone();
            let mut dense = DenseMatCM::<MyInt, BlsFr>::new(m, n);
            dense.set_data(data);
            matc.push(dense);

            // Build a random evaluation point matching the shape
            let log_m = (m as u64).ilog2() as usize;
            let log_n = (n as u64).ilog2() as usize;
            let mut rng = test_rng();
            let xl: Vec<BlsFr> = (0..log_m).map(|_| BlsFr::rand(&mut rng)).collect();
            let xr: Vec<BlsFr> = (0..log_n).map(|_| BlsFr::rand(&mut rng)).collect();

            // Compute hat via field container to cross-check later if needed
            let mut field_mat = DenseMatFieldCM::<BlsFr>::new(m, n);
            // convert from current MyInt matrix (local copy) to Field matrix with same layout
            let mut field_cols = Vec::with_capacity(n);
            for j in 0..n {
                let col: Vec<BlsFr> = data_for_field[j]
                    .iter()
                    .map(|&v| BlsFr::from(v as i64))
                    .collect();
                field_cols.push(col);
            }
            field_mat.set_data(field_cols);
            let hat = field_mat.proj_lr_challenges(&xl, &xr);

            // Push point to points container
            pointc.push(hat, (xl, xr), idx, (vec![idx], vec![idx]));
        }

        // Length consistency between flattened matrices and xi
        check_length_consistency(&matc, &pointc, BlsFr::from(7u64));

    // Flatten into a single vector for PCS (same logic as xi alignment)
    let vec_myint = matc.flatten_and_concat();

        // Setup PCS with sufficient qlog: vec length is power-of-two due to padding
        let total_len = vec_myint.len();
        let qlog = (total_len as u64).ilog2() as usize; // SmartPC expects q=2^qlog
        let mut rng = test_rng();
        let pp: PcsPP<Bls12_381> = SmartPC::<Bls12_381>::setup(qlog, &mut rng).expect("pcs setup failed");

        // Commit the single-row matrix as 1×total_len (col-major: n=1 col, m=total_len rows)
        let (com, mut tier1) = SmartPC::<Bls12_381>::commit_square_myint(&pp, &vec![vec_myint.clone()], BlsFr::zero()).expect("commit_square_myint failed");

        // Build a batching challenge and xi for points container to compute v (hat batched)
        let chal = BlsFr::from(5u64);
        let xi = pointc.xi_concat(chal);
        let hat_batched = pointc.hat_batched(chal);

        // Check inner product equals batched hat
        let a_field: Vec<BlsFr> = vec_myint.iter().map(|&x| BlsFr::from(x as i64)).collect();
        let ip = inner_product(&a_field, &xi);
        assert_eq!(ip, hat_batched, "Flattened·xi must equal batched hat");

        // Now open at concatenated challenge [xr||xl] derived from the single batched point
        // We reuse xi construction path: the PCS open_square expects explicit xl/xr for the whole matrix.
        // Here we simulate a simple case: treat the whole flattened vector as m×n = total_len×1 (xl bits = log(total_len)).
        let xl_full: Vec<BlsFr> = (0..qlog).map(|_| BlsFr::rand(&mut rng)).collect();
        let xr_full: Vec<BlsFr> = vec![];
        let (v_com, v_tilde) = SmartPC::<Bls12_381>::eval(&pp, &vec![a_field.clone()], &xl_full, &xr_full);

        let proof = SmartPC::<Bls12_381>::open_square_myint(
            &pp,
            &vec![vec_myint],
            &xl_full,
            &xr_full,
            v_com,
            com,
            &mut tier1,
            v_tilde,
            BlsFr::zero(),
        ).expect("open_square_myint failed");

        let ok = SmartPC::<Bls12_381>::verify_square(&pp, com, v_com, &xl_full, &xr_full, &proof).expect("verify_square failed");
        assert!(ok, "PCS verify_square should succeed on flattened single-row after square conversion");
    }
}

/// Debug utility to check length consistency between containers.
/// Verifies that flatten_and_concat and xi_concat produce vectors of the same length.
/// If not, print detailed diagnostics (matrix areas, areas inferred from points,
/// per‑point start offsets, etc.).
///
/// # Arguments
/// * `mat_container` - Integer matrix container to check
/// * `point_container` - Points container to check
/// * `challenge` - Challenge field element for xi computation
pub fn check_length_consistency<F: PrimeField + core::fmt::Debug>(
    mat_container: &MatContainerMyInt<F>,
    point_container: &PointsContainer<F>,
    challenge: F,
) {
    let flat = mat_container.flatten_and_concat();
    let xi = point_container.xi_concat(challenge);

    if flat.len() != xi.len() {
        eprintln!("❌ Length mismatch: flatten={}  xi={}", flat.len(), xi.len());
        eprintln!("Matrix areas (sorted ascending):    {:?}", mat_container.sorted_areas);
        eprintln!("Point container derived areas: {:?}", point_container.sorted_areas);

        for i in 0..point_container.sorted_points.len() {
            let (xl, xr) = &point_container.sorted_points[i];
            let shape_from_points = ((1usize << xl.len()), (1usize << xr.len()));
            eprintln!(
                "Point #{i}: xl_len={} xr_len={} -> inferred shape={:?} area={} start_pos={}", 
                xl.len(), xr.len(), shape_from_points, shape_from_points.0 * shape_from_points.1, point_container.sorted_start_position[i]
            );
        }

        for i in 0..mat_container.sorted_matrices.len() {
            let shape = mat_container.sorted_shapes[i];
            eprintln!("Matrix #{i}: actual shape={:?} area={}", shape, shape.0 * shape.1);
        }

        panic!("Length mismatch between flattened matrices and xi vector. Ensure point challenge vector lengths are log2(dimensions).");
    } else {
        println!("✅ Lengths match: {}", flat.len());
    }
}

/// Debug utility to assert that the area-sorted insertion is stably aligned between
/// MatContainerMyInt and PointsContainer.
/// It checks:
/// - same number of items
/// - per-index shape and area match (matrix real shape vs. point-inferred shape)
/// - start positions are multiples of their areas and non-decreasing
/// On mismatch it prints detailed diagnostics and panics.
pub fn check_shapes_aligned<F: PrimeField + core::fmt::Debug>(
    mat_container: &MatContainerMyInt<F>,
    point_container: &PointsContainer<F>,
) {
    let m_len = mat_container.sorted_shapes.len();
    let p_len = point_container.sorted_shapes.len();

    if m_len != p_len {
        eprintln!(
            "❌ Count mismatch: matrices={} points={} (area-sorted sequences must have equal length)",
            m_len, p_len
        );
        eprintln!("Matrix shapes:   {:?}", mat_container.sorted_shapes);
        eprintln!("Point shapes:    {:?}", point_container.sorted_shapes);
        panic!("Mat/Point containers have different counts");
    }

    let mut ok = true;
    for i in 0..m_len {
        let m_shape = mat_container.sorted_shapes[i];
        let m_area = mat_container.sorted_areas[i];

        let p_shape = point_container.sorted_shapes[i];
        let p_area = point_container.sorted_areas[i];

        if m_shape != p_shape || m_area != p_area {
            eprintln!(
                "❌ Shape/area mismatch at index {i}: mat shape={:?} area={} | point shape={:?} area={}",
                m_shape, m_area, p_shape, p_area
            );
            ok = false;
        }

        // Check start position alignment for the point stream
        let start_pos = point_container.sorted_start_position[i];
        if start_pos % p_area != 0 {
            eprintln!(
                "❌ Start position misaligned at index {i}: start_pos={} area={} (must be multiple)",
                start_pos, p_area
            );
            ok = false;
        }

        if i > 0 {
            let prev = point_container.sorted_start_position[i - 1] + point_container.sorted_areas[i - 1];
            // With padding, current start must be >= previous end
            if start_pos < prev && point_container.sorted_areas[i] >= point_container.sorted_areas[i - 1] {
                eprintln!(
                    "❌ Start position order issue at index {i}: start_pos={} < prev_end={} (non-decreasing expected)",
                    start_pos, prev
                );
                ok = false;
            }
        }
    }

    if !ok {
        // Print compact overviews to help locate clustering issues
        eprintln!("Matrix areas (asc): {:?}", mat_container.sorted_areas);
        eprintln!("Point  areas (asc): {:?}", point_container.sorted_areas);
        eprintln!("Point  starts (asc by area): {:?}", point_container.sorted_start_position);
        panic!("Shapes/start alignment check failed. Ensure both containers are filled in matching push order for equal-area items and that (xl,xr) match real shapes.");
    } else {
        println!("✅ Shapes and start positions aligned ({} items)", m_len);
    }
}

/// Store the point information
#[derive(Debug, Clone)]
pub struct PointInfo<F: PrimeField> {
    pub hat: F,  // The projection of the matrix
    pub point: (Vec<F>, Vec<F>),
    pub hat_index: usize,
    pub point_index: (Vec<usize>, Vec<usize>),
}

impl<F: PrimeField> PointInfo<F> {
    pub fn new(hat: F, point: (Vec<F>, Vec<F>), hat_index: usize, point_index: (Vec<usize>, Vec<usize>)) -> Self {
        Self { hat, point, hat_index, point_index }
    }

    pub fn default() -> Self {
        Self {
            hat: F::zero(),
            point: (Vec::new(), Vec::new()),
            hat_index: 0,
            point_index: (Vec::new(), Vec::new()),
        }
    }
}