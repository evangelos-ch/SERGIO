pub mod gene;
pub mod grn;
pub mod interaction;
pub mod mrs;
pub mod sim;

use pyo3::prelude::*;

#[pymodule]
fn sergio_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<gene::Gene>()?;
    m.add_class::<interaction::Interaction>()?;
    m.add_class::<grn::GRN>()?;
    m.add_class::<mrs::MrProfile>()?;
    m.add_class::<sim::Sim>()?;
    Ok(())
}
