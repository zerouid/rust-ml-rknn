use anyhow::Result;
use std::ffi::CStr;

use rknn_api_sys::*;

fn get_type_string(t: rknn_tensor_type) -> &'static str {
    match t {
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32 => "FP32",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT16 => "FP16",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_INT8 => "INT8",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_UINT8 => "UINT8",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_INT16 => "INT16",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_UINT16 => "UINT16",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_INT32 => "INT32",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_UINT32 => "UINT32",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_INT64 => "INT64",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_BOOL => "BOOL",
        rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_INT4 => "INT4",
        _ => "UNKNOW",
    }
}

fn get_qnt_type_string(t: rknn_tensor_qnt_type) -> &'static str {
    match t {
        rknn_api_sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => "NONE",
        rknn_api_sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => "DFP",
        rknn_api_sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => "AFFINE",
        _ => "UNKNOW",
    }
}

fn get_format_string(fmt: rknn_tensor_format) -> &'static str {
    match fmt {
        rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NCHW => "NCHW",
        rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NHWC => "NHWC",
        rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NC1HWC2 => "NC1HWC2",
        rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_UNDEFINED => "UNDEFINED",
        _ => "UNKNOW",
    }
}

pub fn safe_string(chars: &[::std::os::raw::c_char]) -> anyhow::Result<String> {
    let cstr = CStr::from_bytes_until_nul(chars)?;
    // Get copy-on-write Cow<'_, str>, then guarantee a freshly-owned String allocation
    Ok(String::from_utf8_lossy(cstr.to_bytes()).to_string())
}

pub trait DumpVals {
    fn dump(&self) -> Result<String>;
}

impl DumpVals for &rknn_tensor_attr {
    fn dump(&self) -> Result<String> {
        let dims = self.dims[0..self.n_dims as usize].to_vec();
        Ok(format!("  index={}, name={}, n_dims={}, dims={:?}, n_elems={}, size={}, w_stride={}, size_with_stride={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
           self.index, safe_string(&self.name)?, self.n_dims, dims, self.n_elems, self.size, self.w_stride,
           self.size_with_stride, get_format_string(self.fmt), get_type_string(self.type_),
           get_qnt_type_string(self.qnt_type), self.zp, self.scale))
    }
}

impl DumpVals for &rknn_input_range {
    fn dump(&self) -> Result<String> {
        let dims: Vec<Vec<u32>> = self.dyn_range[0..self.shape_number as usize]
            .iter()
            .map(|d| d[0..self.n_dims as usize].to_vec())
            .collect();
        Ok(format!(
            "  index={}, name={}, shape_number={}, range={:?}, fmt={}",
            self.index,
            safe_string(&self.name)?,
            self.shape_number,
            dims,
            get_format_string(self.fmt)
        ))
    }
}

pub trait DumpStats {
    fn dump_stats(&self, units: &str) -> String;
}

impl DumpStats for Vec<f64> {
    fn dump_stats(&self, units: &str) -> String {
        if !self.is_empty() {
            let count = self.len() as f64;
            let mean = self.iter().sum::<f64>() / count;
            let variance = self
                .iter()
                .map(|value| {
                    let diff = mean - *value;
                    diff * diff
                })
                .sum::<f64>()
                / count;

            let sd = variance.sqrt();
            let min = self.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
            let max = self.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            format!(
                "avg: {mean:.2}{units}, min {min:.2}{units}, max {max:.2}{units}, sd {sd:.2}{units}"
            )
        } else {
            "avg: N/A , min N/A, max N/A, sd N/A".to_string()
        }
    }
}

#[macro_export]
macro_rules! time_bench {
    { $times:expr,$x:block } => {
        println!("Begin perf ...");
            let mut observations = Vec::new();
            for i in 0..$times {
                let now = std::time::Instant::now();

                $x

                let duration = now.elapsed();
                println!(
                    "{}: Elapse Time = {:?}, FPS = {:.2}",
                    i,
                    duration,
                    1.0 / duration.as_secs_f32()
                );
                observations.push(duration.as_millis() as f64);
            }
            println!("{}",observations.dump_stats("ms"));
    };
}
