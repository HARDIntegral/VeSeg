use ndarray::Array2;

const RADIUS_SMOOTHING_WINDOW: isize = 3;
const RADIUS_SMOOTHING_PASSES: usize = 2;
const CLOSING_RADIUS: isize = 1;
const BLUR_RADIUS: isize = 1;
const RECONSTRUCTION_THRESHOLD: f32 = 0.45;

/// Reconstruct a cleaner vessel mask from skeleton centerlines and local radius estimates.
///
/// The skeleton gives the vessel centerlines, and the distance map gives the
/// approximate vessel radius at each skeleton pixel. Reconstruction draws local
/// radius-weighted disks along the skeleton, then applies small morphology and
/// blur passes to reduce stair-step artifacts.
pub fn reconstruct_vessels(skeleton: &Array2<bool>, distance_map: &Array2<f32>) -> Array2<bool> {
    let shape = skeleton.dim();

    assert_eq!(
        distance_map.dim(),
        shape,
        "skeleton and distance_map must have the same shape"
    );

    // Main reconstruction pipeline:
    // radius sampling -> radius smoothing -> disk drawing -> closing -> blur -> threshold.
    let radii = smooth_skeleton_radii(skeleton, distance_map);
    let reconstruction = draw_radius_weighted_centerlines(skeleton, &radii);
    let closed = close_mask(&reconstruction);
    let blurred = blur_mask(&closed);

    threshold_float_mask(&blurred)
}

fn smooth_skeleton_radii(skeleton: &Array2<bool>, distance_map: &Array2<f32>) -> Array2<f32> {
    let (height, width) = skeleton.dim();
    let mut radii = Array2::<f32>::zeros((height, width));

    // Only skeleton pixels get radius values because only centerlines reconstruct vessels.
    for row in 0..height {
        for col in 0..width {
            if skeleton[(row, col)] {
                radii[(row, col)] = distance_map[(row, col)];
            }
        }
    }

    // Smooth radii along neighboring skeleton pixels so vessel widths change gradually.
    for _ in 0..RADIUS_SMOOTHING_PASSES {
        radii = smooth_radius_pass(skeleton, &radii);
    }

    radii
}

fn smooth_radius_pass(skeleton: &Array2<bool>, previous: &Array2<f32>) -> Array2<f32> {
    let (height, width) = skeleton.dim();
    let mut smoothed = previous.clone();

    for row in 0..height {
        for col in 0..width {
            if !skeleton[(row, col)] {
                continue;
            }

            let mut total = 0.0;
            let mut count = 0.0;

            for_each_disk_neighbor(
                row,
                col,
                height,
                width,
                RADIUS_SMOOTHING_WINDOW,
                |neighbor_row, neighbor_col| {
                    if skeleton[(neighbor_row, neighbor_col)] {
                        total += previous[(neighbor_row, neighbor_col)];
                        count += 1.0;
                    }
                },
            );

            if count > 0.0 {
                smoothed[(row, col)] = total / count;
            }
        }
    }

    smoothed
}

fn draw_radius_weighted_centerlines(skeleton: &Array2<bool>, radii: &Array2<f32>) -> Array2<bool> {
    let (height, width) = skeleton.dim();
    let mut reconstruction = Array2::<bool>::from_elem((height, width), false);

    // Each skeleton point becomes a local disk. Overlapping disks form vessel tubes.
    for row in 0..height {
        for col in 0..width {
            if !skeleton[(row, col)] {
                continue;
            }

            let radius = radii[(row, col)];

            if radius > 0.0 {
                draw_disk(&mut reconstruction, row, col, radius);
            }
        }
    }

    reconstruction
}

// Binary closing = dilation followed by erosion.
// It fills tiny gaps and rounds sharp corners after disk reconstruction.
fn close_mask(mask: &Array2<bool>) -> Array2<bool> {
    let dilated = dilate_mask(mask, CLOSING_RADIUS);
    erode_mask(&dilated, CLOSING_RADIUS)
}

fn dilate_mask(mask: &Array2<bool>, radius: isize) -> Array2<bool> {
    let (height, width) = mask.dim();
    let mut dilated = Array2::<bool>::from_elem((height, width), false);

    // Expand every vessel pixel into its local circular neighborhood.
    for row in 0..height {
        for col in 0..width {
            if !mask[(row, col)] {
                continue;
            }

            for_each_disk_neighbor(
                row,
                col,
                height,
                width,
                radius,
                |neighbor_row, neighbor_col| {
                    dilated[(neighbor_row, neighbor_col)] = true;
                },
            );
        }
    }

    dilated
}

fn erode_mask(mask: &Array2<bool>, radius: isize) -> Array2<bool> {
    let (height, width) = mask.dim();
    let mut eroded = Array2::<bool>::from_elem((height, width), false);

    // Shrink the dilated mask back while preserving newly closed small gaps.
    for row in 0..height {
        for col in 0..width {
            if !mask[(row, col)] {
                continue;
            }

            eroded[(row, col)] = disk_is_filled(mask, row, col, radius);
        }
    }

    eroded
}

fn disk_is_filled(mask: &Array2<bool>, row: usize, col: usize, radius: isize) -> bool {
    let (height, width) = mask.dim();
    let mut filled = true;

    for_each_disk_neighbor(
        row,
        col,
        height,
        width,
        radius,
        |neighbor_row, neighbor_col| {
            if !mask[(neighbor_row, neighbor_col)] {
                filled = false;
            }
        },
    );

    filled
}

fn blur_mask(mask: &Array2<bool>) -> Array2<f32> {
    let (height, width) = mask.dim();
    let mut blurred = Array2::<f32>::zeros((height, width));

    // Light Gaussian-style blur softens stair-step boundaries before thresholding.
    for row in 0..height {
        for col in 0..width {
            let mut weighted_total = 0.0;
            let mut weight_sum = 0.0;

            for_each_gaussian_neighbor(
                row,
                col,
                height,
                width,
                |neighbor_row, neighbor_col, weight| {
                    weighted_total += mask[(neighbor_row, neighbor_col)] as u8 as f32 * weight;
                    weight_sum += weight;
                },
            );

            if weight_sum > 0.0 {
                blurred[(row, col)] = weighted_total / weight_sum;
            }
        }
    }

    blurred
}

// Convert blurred floating-point values back into a binary vessel mask.
fn threshold_float_mask(mask: &Array2<f32>) -> Array2<bool> {
    let (height, width) = mask.dim();
    let mut thresholded = Array2::<bool>::from_elem((height, width), false);

    for row in 0..height {
        for col in 0..width {
            thresholded[(row, col)] = mask[(row, col)] >= RECONSTRUCTION_THRESHOLD;
        }
    }

    thresholded
}

// Visit all pixels inside a circular neighborhood without allocating a Vec.
// This is used in radius smoothing, dilation, and erosion.
fn for_each_disk_neighbor<F>(
    center_row: usize,
    center_col: usize,
    height: usize,
    width: usize,
    radius: isize,
    mut visit: F,
) where
    F: FnMut(usize, usize),
{
    let center_row = center_row as isize;
    let center_col = center_col as isize;
    let radius_squared = (radius * radius) as f32;

    for row_offset in -radius..=radius {
        for col_offset in -radius..=radius {
            let distance_squared = (row_offset * row_offset + col_offset * col_offset) as f32;

            if distance_squared > radius_squared {
                continue;
            }

            let row = center_row + row_offset;
            let col = center_col + col_offset;

            if row < 0 || col < 0 {
                continue;
            }

            let row = row as usize;
            let col = col as usize;

            if row < height && col < width {
                visit(row, col);
            }
        }
    }
}

// Visit blur neighbors with Gaussian weights without allocating a temporary Vec.
fn for_each_gaussian_neighbor<F>(
    center_row: usize,
    center_col: usize,
    height: usize,
    width: usize,
    mut visit: F,
) where
    F: FnMut(usize, usize, f32),
{
    let center_row = center_row as isize;
    let center_col = center_col as isize;
    let sigma = BLUR_RADIUS.max(1) as f32;
    let two_sigma_squared = 2.0 * sigma * sigma;

    for row_offset in -BLUR_RADIUS..=BLUR_RADIUS {
        for col_offset in -BLUR_RADIUS..=BLUR_RADIUS {
            let row = center_row + row_offset;
            let col = center_col + col_offset;

            if row < 0 || col < 0 {
                continue;
            }

            let row = row as usize;
            let col = col as usize;

            if row >= height || col >= width {
                continue;
            }

            let distance_squared = (row_offset * row_offset + col_offset * col_offset) as f32;
            let weight = (-distance_squared / two_sigma_squared).exp();

            visit(row, col, weight);
        }
    }
}

// Reconstruct one local vessel cross-section by drawing a filled disk.
fn draw_disk(mask: &mut Array2<bool>, center_row: usize, center_col: usize, radius: f32) {
    let (height, width) = mask.dim();
    let radius_ceiling = radius.ceil() as isize;
    let radius_squared = radius * radius;
    let center_row = center_row as isize;
    let center_col = center_col as isize;

    // Rasterize the disk by checking whether neighboring pixels fall inside radius².
    for row_offset in -radius_ceiling..=radius_ceiling {
        for col_offset in -radius_ceiling..=radius_ceiling {
            let distance_squared = (row_offset * row_offset + col_offset * col_offset) as f32;

            if distance_squared > radius_squared {
                continue;
            }

            let row = center_row + row_offset;
            let col = center_col + col_offset;

            if row < 0 || col < 0 {
                continue;
            }

            let row = row as usize;
            let col = col as usize;

            if row < height && col < width {
                mask[(row, col)] = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn reconstructs_single_center_pixel_with_radius() {
        let skeleton = array![
            [false, false, false, false, false],
            [false, false, false, false, false],
            [false, false, true, false, false],
            [false, false, false, false, false],
            [false, false, false, false, false],
        ];

        let mut distance_map = Array2::<f32>::zeros((5, 5));
        distance_map[(2, 2)] = 1.0;

        let reconstruction = reconstruct_vessels(&skeleton, &distance_map);
        let count = reconstruction.iter().filter(|&&pixel| pixel).count();

        assert!(count >= 1);
        assert!(reconstruction[(2, 2)]);
    }

    #[test]
    fn ignores_zero_radius_skeleton_pixels() {
        let skeleton = array![
            [false, false, false],
            [false, true, false],
            [false, false, false],
        ];
        let distance_map = Array2::<f32>::zeros((3, 3));

        let reconstruction = reconstruct_vessels(&skeleton, &distance_map);
        let count = reconstruction.iter().filter(|&&pixel| pixel).count();

        assert_eq!(count, 0);
    }
}
