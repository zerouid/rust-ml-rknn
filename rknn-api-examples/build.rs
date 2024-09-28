use anyhow::{Context, Result};
use std::{
    fs::{self, File},
    io::copy,
    path::Path,
};

const DOWNLOAD_BASE_URL: &'static str =
    "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/refs/heads/master/";

const TEST_DATA: [(&'static str, &'static str); 3] = [
    (
        "rknpu2/examples/rknn_common_test/model/cat_224x224.jpg",
        "test-data/cat_224x224.jpg",
    ),
    (
        "rknpu2/examples/rknn_common_test/model/dog_224x224.jpg",
        "test-data/dog_224x224.jpg",
    ),
    (
        "rknn-toolkit-lite2/examples/resnet18/space_shuttle_224.jpg",
        "test-data/space_shuttle_224.jpg",
    ),
];

const MODELS: [(fn(&str) -> String, fn(&str) -> String); 3] = [
    (
        |feat| {
            format!(
                "rknpu2/examples/rknn_dynamic_shape_input_demo/model/{}/mobilenet_v2.rknn",
                feat.to_uppercase()
            )
        },
        |feat| format!("models/mobilenet_v2_for_{}.rknn", feat),
    ),
    (
        |feat| {
            format!(
                "rknn-toolkit-lite2/examples/resnet18/resnet18_for_{}.rknn",
                feat
            )
        },
        |feat| format!("models/resnet18_for_{}.rknn", feat),
    ),
    (
        |feat| {
            format!(
                "rknpu2/examples/rknn_common_test/model//{}/mobilenet_v1.rknn",
                feat.to_uppercase()
            )
        },
        |feat| format!("models/mobilenet_v1_for_{}.rknn", feat),
    ),
];

fn download_image(url: &str, file_path: &str) -> Result<()> {
    let file_path = Path::new(file_path);
    if !Path::exists(&file_path) {
        // Send an HTTP GET request to the URL
        let mut response = reqwest::blocking::get(url)
            .context("Failed to make HTTP request")?
            .error_for_status()
            .map_err(|err| {
                anyhow::anyhow!("Failed to fetch data from {}: {}", url, err.to_string())
            })?;
        // Ensure folder exists
        fs::create_dir_all(file_path.parent().unwrap())?;
        // Create a new file to write the downloaded image to
        let mut file = File::create(file_path)?;

        // Copy the contents of the response to the file
        copy(&mut response, &mut file)?;
    }

    Ok(())
}

fn main() {
    for (src, dst) in TEST_DATA {
        let url = format!("{}{}", DOWNLOAD_BASE_URL, src);
        match download_image(&url, &dst) {
            Ok(_) => (),
            Err(err) => panic!("Failed to download {}. Err: {}", &url, &err),
        }
    }

    let selected_features = [
        #[cfg(feature = "rk3562")]
        "rk3562",
        #[cfg(feature = "rk3566_rk3568")]
        "rk3566_rk3568",
        #[cfg(feature = "rk3576")]
        "rk3576",
        #[cfg(feature = "rk3588")]
        "rk3588",
    ];

    for feat in selected_features {
        for (format_src, format_dst) in MODELS {
            let url = format!("{}{}", DOWNLOAD_BASE_URL, format_src(feat));
            let file_path = format_dst(feat);
            match download_image(&url, &file_path) {
                Ok(_) => (),
                Err(err) => panic!("Failed to download {}. Err: {}", &url, &err),
            }
        }
    }
}
