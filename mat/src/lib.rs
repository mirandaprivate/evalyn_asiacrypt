/// Implementations for zkSmart
/// 
#[allow(unused)]
#[macro_use]
extern crate derivative;
#[macro_use]
extern crate ark_std;


pub mod data_structures;
pub mod protocols;
pub mod utils;


pub use crate::utils::linear;
pub use crate::utils::matdef::{DenseMatCM,DenseMatFieldCM,ShortInt};
pub use crate::utils::xi;

pub type MyInt = ark_poly_commit::MyInt;
pub type MyShortInt = i8;