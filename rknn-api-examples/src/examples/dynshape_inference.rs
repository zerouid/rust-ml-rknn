use anyhow::Result;
use clap::Parser;
use image::DynamicImage;

use crate::{
    examples::{
        common::*,
        utils::{DumpStats, DumpVals},
    },
    time_bench,
};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Debug, Parser)]
pub struct Example {
    /// The path to the model file (*.rknn)
    #[arg(short, long)]
    model_path: String,

    /// The path to the input image file
    #[arg(short, long, value_parser)]
    input_paths: Vec<String>,

    /// The number of loops
    #[arg(short, long, default_value_t = 1)]
    loop_count: u8,

    #[arg(short, long, value_enum, default_value_t = RknnCoreMask::Npu0)]
    core_mask: RknnCoreMask,

    /// The path to the output image file
    #[arg(short, long)]
    output_dir: Option<String>,
}

impl Example {
    pub fn execute(&self) -> Result<()> {
        let ctx = RKNNContext::load_model(&self.model_path)?;
        println!("\x1b[34;4m Load model sucess\x1b[0m");
        let ver = &ctx.get_sdk_version()?;
        println!(
            "\x1b[34;4m model input num: {}, output num: {}\x1b[0m",
            ctx.n_input, ctx.n_output
        );
        println!("{}{}", ver.api_verion, ver.driver_verion);

        if self.input_paths.len() != ctx.n_input as usize {
            panic!(
                "inconsistent number of input paths ({}) expected ({})",
                self.input_paths.len(),
                ctx.n_input
            );
        }

        println!("\x1b[34;4m input tensors:\x1b[0m");
        let mut input_attrs = ctx.get_input_attrs()?;
        for attr in &input_attrs {
            println!("{}", attr.dump()?);
        }

        println!("\x1b[34;4m output tensors:\x1b[0m");
        let output_attrs = ctx.get_output_attrs()?;
        for attr in &output_attrs {
            println!("{}", attr.dump()?);
        }

        println!("\x1b[34;4m dynamic inputs shape range:\x1b[0m");
        let shape_range = ctx.get_input_range()?;
        for range in &shape_range {
            println!("{}", range.dump()?);
        }

        println!("\x1b[34;4m load input images\x1b[0m");
        let images: Vec<DynamicImage> = self
            .input_paths
            .iter()
            .map(|p| {
                image::ImageReader::open(p)
                    .expect("failed to open")
                    .decode()
                    .expect("failed to decode")
            })
            .collect();

        for s in 0..shape_range[0].shape_number {
            println!(
                "\x1b[34;4m setting dynamic shape {}:{:?}\x1b[0m",
                s,
                shape_range[0].dyn_range[s as usize]
                    .iter()
                    .take(shape_range[0].n_dims as usize)
                    .collect::<Vec<_>>()
            );

            for i in 0..ctx.n_input {
                for j in 0..input_attrs[i as usize].n_dims {
                    input_attrs[i as usize].dims[j as usize] =
                        shape_range[i as usize].dyn_range[s as usize][j as usize];
                }
            }
            ctx.set_input_shapes(&mut input_attrs)?;

            let cur_input_attrs = ctx.get_input_attrs()?;
            println!("\x1b[34;4m current input tensors:\x1b[0m");
            for attr in &cur_input_attrs {
                println!("{}", attr.dump()?);
            }

            let cur_output_attrs = ctx.get_output_attrs()?;
            println!("\x1b[34;4m current output tensors:\x1b[0m");
            for attr in &cur_output_attrs {
                println!("{}", attr.dump()?);
            }

            ctx.set_core_mask(&self.core_mask)?;
            ctx.set_inputs(&cur_input_attrs, &images)?;

            time_bench!(self.loop_count, {
                ctx.run()?;
            });

            let outputs = ctx.get_outputs()?;
            let len = outputs[0].size as usize / size_of::<f32>();
            let results = unsafe { Vec::from_raw_parts(outputs[0].buf as *mut f32, len, len) };
            let mut results_pairs: Vec<(usize, f32)> =
                results.iter().enumerate().map(|(i, f)| (i, *f)).collect();
            results_pairs.sort_by(|(_, f1), (_, f2)| f2.total_cmp(f1));
            println!("\x1b[34;4m --- Top5 ---\x1b[0m");
            for (i, f) in results_pairs.iter().take(5) {
                println!("{}: {:.2}%", i, f);
            }
        }
        Ok(())
    }
}
