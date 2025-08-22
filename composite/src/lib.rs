pub mod protocols;
pub mod utils;

// Re-export PointInfo for external users (previously attempted via utils::point)
pub use atomic_proof::PointInfo;


pub type MyInt = mat::MyInt;
pub type MyShortInt = mat::MyShortInt;