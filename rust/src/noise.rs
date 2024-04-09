use std::ops::Not;

use ndarray::{s, Array, Array2, ArrayView2, Axis, NewAxis};
use ndarray_rand::{
    rand::thread_rng,
    rand_distr::{Bernoulli, Distribution, LogNormal},
    RandomExt,
};
use ndarray_stats::{interpolate::Midpoint, QuantileExt};
use noisy_float::types::N64;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{pyclass, pyfunction, Python};

#[pyclass]
pub enum NoiseSetting {
    DS1,
    DS2,
    DS3,
    DS4,
    DS5,
    DS6,
    DS7,
    DS8,
    DS13,
    DS14,
}

pub fn add_outlier_effect(expr: &mut Array2<f64>, p: f64, mu: f64, sigma: f64) {
    let outlier_indicator = Array::random_using(
        expr.len_of(Axis(0)),
        Bernoulli::new(p).unwrap(),
        &mut thread_rng(),
    );
    let outlier_factors = Array::random_using(
        expr.len_of(Axis(0)),
        LogNormal::new(mu, sigma).unwrap(),
        &mut thread_rng(),
    );
    expr.axis_iter_mut(Axis(0))
        .zip(outlier_factors)
        .enumerate()
        .for_each(|(i, (mut gene_expr, outlier_factor))| {
            if outlier_indicator[i] {
                gene_expr.map_inplace(|x| *x *= outlier_factor)
            }
        });
}

pub fn add_lib_size_effect(expr: &mut Array2<f64>, mu: f64, sigma: f64) {
    let lib_factors = Array::random_using(
        expr.len_of(Axis(1)),
        LogNormal::new(mu, sigma).unwrap(),
        &mut thread_rng(),
    );
    let norm_constant = expr.sum_axis(Axis(0));
    let cell_factors = &lib_factors / &norm_constant;
    expr.zip_mut_with(&cell_factors.slice(s![NewAxis, ..]), |x, c| *x *= c);
}

pub fn add_dropout(expr: &mut Array2<f64>, k: f64, q: N64) {
    let mut log_expr = expr.map(|x| (x + 1.0).ln());
    let log_mid_point = log_expr
        .quantile_axis_skipnan_mut(Axis(0), q / 100.0, &Midpoint)
        .unwrap();
    log_expr.zip_mut_with(&log_mid_point, |x, mid_point| {
        *x = 1.0 / (1.0 + (-1.0 * k * (*x - mid_point)).exp());
    });
    expr.zip_mut_with(&log_expr, |x, p| {
        if Bernoulli::new(*p).unwrap().sample(&mut thread_rng()).not() {
            *x = 0.0;
        }
    });
}

pub fn add_technical_noise_custom(
    expr: &ArrayView2<f64>,
    outlier_mu: f64,
    library_mu: f64,
    library_sigma: f64,
    dropout_k: f64,
    dropout_q: N64,
) -> Array2<f64> {
    let mut data_copy = expr.to_owned();
    add_outlier_effect(&mut data_copy, 0.01, outlier_mu, 1.0);
    add_lib_size_effect(&mut data_copy, library_mu, library_sigma);
    add_dropout(&mut data_copy, dropout_k, dropout_q);
    data_copy
}

pub fn add_technical_noise(expr: &ArrayView2<f64>, setting: &NoiseSetting) -> Array2<f64> {
    match setting {
        NoiseSetting::DS1 => add_technical_noise_custom(expr, 0.8, 4.8, 0.3, 20.0, N64::new(82.0)),
        NoiseSetting::DS2 => add_technical_noise_custom(expr, 0.8, 6.0, 0.4, 12.0, N64::new(80.0)),
        NoiseSetting::DS3 => add_technical_noise_custom(expr, 0.8, 7.0, 0.4, 8.0, N64::new(80.0)),
        NoiseSetting::DS4 => add_technical_noise_custom(expr, 3.0, 6.0, 0.3, 8.0, N64::new(74.0)),
        NoiseSetting::DS5 => add_technical_noise_custom(expr, 3.0, 6.0, 0.4, 8.0, N64::new(82.0)),
        NoiseSetting::DS6 => add_technical_noise_custom(expr, 5.0, 4.5, 0.7, 8.0, N64::new(45.0)),
        NoiseSetting::DS7 => add_technical_noise_custom(expr, 3.0, 4.4, 0.8, 8.0, N64::new(85.0)),
        NoiseSetting::DS8 => add_technical_noise_custom(expr, 4.5, 10.8, 0.55, 2.0, N64::new(92.0)),
        NoiseSetting::DS13 => add_technical_noise_custom(expr, 0.8, 3.6, 0.4, 8.0, N64::new(70.0)),
        NoiseSetting::DS14 => add_technical_noise_custom(expr, 0.8, 5.0, 0.4, 4.0, N64::new(80.0)),
    }
}

#[pyfunction]
#[pyo3(name = "add_technical_noise")]
pub fn py_add_technical_noise<'py>(
    py: Python<'py>,
    expr: PyReadonlyArray2<f64>,
    setting: &NoiseSetting,
) -> &'py PyArray2<f64> {
    let rust_array = expr.as_array();
    let noisy_data = add_technical_noise(&rust_array, setting);
    noisy_data.to_pyarray(py)
}

mod tests {
    use super::*;

    #[test]
    fn test_noise() {
        let arr1 = Array::random_using(
            (100, 500),
            LogNormal::new(7.0, 1.0).unwrap(),
            &mut thread_rng(),
        );
        let noisy_data = add_technical_noise(&arr1.view(), &NoiseSetting::DS6);
        assert_ne!(arr1, noisy_data);
    }
}
