//! Atomic proof library for zero-knowledge proofs
pub mod atomic_pop;
pub mod atomic_protocol;
pub mod pop;
pub mod protocols;
pub mod utils;

pub use pop::{
    ArithmeticExpression,
    mlpcs::{MLPCS, MLPCSProverKey, MLPCSCommitment, MLPCSProof},
    groth16::Groth16Prover,
};
pub use atomic_protocol::{AtomicMatProtocol, AtomicMatProtocolInput, MatOp};

pub use utils::{MatContainer, MatContainerMyInt, PointsContainer, PointInfo};

pub use protocols::{
    MatAdd,
    BatchPoint,
    BatchProj,
    BatchProjField,
    Concat,
    MatEq,
    GrandProd,
    Hadamard,
    LinComb,
    LiteBullet,
    MatSub,
    EqZero,
};
