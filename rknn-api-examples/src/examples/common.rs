use anyhow::{bail, Ok, Result};
use image::DynamicImage;
use rknn_api_sys::{
    rknn_context, rknn_destroy, rknn_init, rknn_input, rknn_input_output_num, rknn_input_range,
    rknn_inputs_set, rknn_output, rknn_outputs_get, rknn_outputs_release, rknn_query,
    rknn_query_cmd, rknn_run, rknn_sdk_version, rknn_set_core_mask, rknn_set_input_shapes,
    rknn_tensor_attr, RKNN_SUCC,
};
use std::ffi::CString;

use crate::examples::utils::safe_string;

#[macro_export]
macro_rules! call_rknn_api {
    ( $x:expr ) => {{
        let ret = unsafe { ($x) };
        if ret != RKNN_SUCC as i32 {
            bail!("error! ret={}", ret)
        }
        Ok(()) as Result<()>
    }};
}

pub struct RKNNContext {
    ctx: rknn_context,
    pub n_input: u32,
    pub n_output: u32,
}

#[repr(u32)]
#[derive(clap::ValueEnum, Copy, Clone, Debug)]
pub enum RknnCoreMask {
    Auto = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_AUTO,
    Npu0 = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_0,
    Npu1 = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_1,
    Npu2 = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_2,
    Npu0_1 = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_0_1,
    Npu0_1_2 = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_0_1_2,
    All = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_ALL,
    Undefined = rknn_api_sys::_rknn_core_mask_RKNN_NPU_CORE_UNDEFINED,
}

pub struct SdkVersion {
    pub api_verion: String,
    pub driver_verion: String,
}

impl RKNNContext {
    pub fn load_model(model_path: &str) -> Result<Self> {
        let mut ctx: rknn_context = 0;
        let c_string = CString::new(model_path).expect("CString::new failed");
        let c_string_ptr = c_string.as_ptr() as *mut ::std::os::raw::c_void;
        call_rknn_api!(rknn_init(
            &mut ctx,
            c_string_ptr,
            0,
            0,
            std::ptr::null_mut()
        ))?;

        let mut io_num: rknn_input_output_num = rknn_input_output_num::default();
        let io_num_ptr = &mut io_num as *mut rknn_input_output_num as *mut ::std::os::raw::c_void;
        call_rknn_api!(rknn_query(
            ctx,
            rknn_api_sys::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
            io_num_ptr,
            std::mem::size_of::<rknn_input_output_num>() as u32,
        ))?;
        Ok(RKNNContext {
            ctx: ctx,
            n_input: io_num.n_input,
            n_output: io_num.n_output,
        })
    }

    pub fn get_sdk_version(&self) -> Result<SdkVersion> {
        let mut sdk_ver: rknn_sdk_version = rknn_sdk_version::default();
        let sdk_ver_ptr = &mut sdk_ver as *mut rknn_sdk_version as *mut ::std::os::raw::c_void;
        call_rknn_api!(rknn_query(
            self.ctx,
            rknn_api_sys::_rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
            sdk_ver_ptr,
            std::mem::size_of::<rknn_sdk_version>() as u32,
        ))?;
        Ok(SdkVersion {
            api_verion: safe_string(&sdk_ver.api_version)?,
            driver_verion: safe_string(&sdk_ver.drv_version)?,
        })
    }

    pub fn get_input_attrs(&self) -> Result<Vec<rknn_tensor_attr>> {
        self.get_attrs(
            self.n_input,
            rknn_api_sys::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
        )
    }

    pub fn get_output_attrs(&self) -> Result<Vec<rknn_tensor_attr>> {
        self.get_attrs(
            self.n_output,
            rknn_api_sys::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
        )
    }

    fn get_attrs(&self, num: u32, cmd: rknn_query_cmd) -> Result<Vec<rknn_tensor_attr>> {
        let mut vec = Vec::with_capacity(num as usize);
        for i in 0..num {
            let mut attrs: rknn_tensor_attr = rknn_tensor_attr {
                index: i,
                ..Default::default()
            };
            let attrs_ptr = &mut attrs as *mut rknn_tensor_attr as *mut ::std::os::raw::c_void;
            call_rknn_api!(rknn_query(
                self.ctx,
                cmd,
                attrs_ptr,
                std::mem::size_of::<rknn_tensor_attr>() as u32,
            ))?;
            vec.push(attrs)
        }
        Ok(vec)
    }

    pub fn get_input_range(&self) -> Result<Vec<rknn_input_range>> {
        let mut vec = Vec::with_capacity(self.n_input as usize);
        for i in 0..self.n_input {
            let mut ranges: rknn_input_range = rknn_input_range {
                index: i,
                ..Default::default()
            };
            let ranges_ptr = &mut ranges as *mut rknn_input_range as *mut ::std::os::raw::c_void;
            call_rknn_api!(rknn_query(
                self.ctx,
                rknn_api_sys::_rknn_query_cmd_RKNN_QUERY_INPUT_DYNAMIC_RANGE,
                ranges_ptr,
                std::mem::size_of::<rknn_input_range>() as u32,
            ))?;

            vec.push(ranges)
        }
        Ok(vec)
    }

    pub fn set_input_shapes(&self, shapes: &mut Vec<rknn_tensor_attr>) -> Result<()> {
        call_rknn_api!(rknn_set_input_shapes(
            self.ctx,
            self.n_input,
            shapes.as_mut_ptr()
        ))?;
        Ok(())
    }

    pub fn set_core_mask(&self, core_mask: &RknnCoreMask) -> Result<()> {
        call_rknn_api!(rknn_set_core_mask(self.ctx, *core_mask as u32))?;
        Ok(())
    }

    pub(crate) fn set_inputs(
        &self,
        input_attrs: &Vec<rknn_tensor_attr>,
        resized_images: &Vec<DynamicImage>,
    ) -> Result<()> {
        let mut inputs: Vec<rknn_input> = resized_images
            .iter()
            .enumerate()
            .map(|(i, img)| {
                let height =
                    if input_attrs[i].fmt == rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NHWC {
                        input_attrs[i].dims[1]
                    } else {
                        input_attrs[i].dims[2]
                    };
                let width =
                    if input_attrs[i].fmt == rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NHWC {
                        input_attrs[i].dims[2]
                    } else {
                        input_attrs[i].dims[3]
                    };
                let mut buf = img
                    .resize(width, height, image::imageops::FilterType::Nearest)
                    .to_rgb8()
                    .into_raw();
                rknn_input {
                    index: i as u32,
                    pass_through: 0,
                    fmt: rknn_api_sys::_rknn_tensor_format_RKNN_TENSOR_NHWC,
                    type_: rknn_api_sys::_rknn_tensor_type_RKNN_TENSOR_UINT8,
                    buf: buf.as_mut_ptr() as *mut ::std::os::raw::c_void,
                    size: buf.len() as u32,
                }
            })
            .collect();
        call_rknn_api!(rknn_inputs_set(self.ctx, self.n_input, inputs.as_mut_ptr()))?;
        Ok(())
    }

    pub fn get_outputs(&self) -> Result<Vec<rknn_output>> {
        let mut outputs: Vec<rknn_output> = (0..self.n_output)
            .map(|i| rknn_output {
                want_float: 1,
                is_prealloc: 0,
                index: i,
                ..Default::default()
            })
            .collect();

        call_rknn_api!(rknn_outputs_get(
            self.ctx,
            self.n_output,
            outputs.as_mut_ptr(),
            std::ptr::null_mut()
        ))?;
        Ok(outputs)
    }
    /// This function causes double attempt to free the memory.
    /// Assume that Rust takes care of memory management and don't use it.
    fn _release_outputs(&self, mut outputs: Vec<rknn_output>) -> Result<()> {
        call_rknn_api!(rknn_outputs_release(
            self.ctx,
            self.n_output,
            outputs.as_mut_ptr()
        ))?;
        Ok(())
    }
    pub fn run(&self) -> Result<()> {
        call_rknn_api!(rknn_run(self.ctx, std::ptr::null_mut()))?;
        Ok(())
    }
}

impl Drop for RKNNContext {
    fn drop(&mut self) {
        println!("destroying RKNNContext");
        unsafe {
            rknn_destroy(self.ctx);
        };
    }
}
