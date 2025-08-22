use ark_ff::PrimeField;

use atomic_protocol::{AtomicMatProtocol,MatOp};
use mat::utils::matdef::DenseMatCM;

/// The Node in an Matrix DAG
/// For efficiency, the Node is stored in shortInt
/// 
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct MatNode<I, F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    pub op: MatOp,
    pub mat: DenseMatCM<I, F>,
    pub input_nodes: Vec<Box<MatNode<I, F>>>,
    pub hat: F,
    pub point: (Vec<F>, Vec<F>),
    pub hat_index: usize,
    pub point_index: (Vec<usize>, Vec<usize>),
    pub ready: (bool, bool)
}
