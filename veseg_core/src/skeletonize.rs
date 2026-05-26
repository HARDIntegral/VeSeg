use ndarray::Array2;
use rayon::prelude::*;

/// Iteratively thin a vessel mask until only 1-pixel-wide centerlines remain.
///
/// This uses Zhang-Suen thinning. Each iteration runs two removal passes. Pixels
/// are collected first and removed afterward so one removal does not influence
/// another pixel during the same pass.
pub fn skeletonize(mask: &Array2<bool>) -> Array2<bool> {
    let mut image = mask.clone();
    let (height, width) = image.dim();

    if height < 3 || width < 3 {
        return image;
    }

    loop {
        let first_pass = collect_pixels_to_remove(&image, ThinningPass::First);
        let mut changed = !first_pass.is_empty();
        remove_pixels(&mut image, &first_pass);

        let second_pass = collect_pixels_to_remove(&image, ThinningPass::Second);
        changed |= !second_pass.is_empty();
        remove_pixels(&mut image, &second_pass);

        if !changed {
            break;
        }
    }

    image
}

#[derive(Debug, Clone, Copy)]
enum ThinningPass {
    First,
    Second,
}

// Find all removable pixels for one Zhang-Suen pass.
// The scan is parallelized by row because each pixel only reads from the current image.
fn collect_pixels_to_remove(image: &Array2<bool>, pass: ThinningPass) -> Vec<(usize, usize)> {
    let (height, width) = image.dim();

    (1..height - 1)
        .into_par_iter()
        .map(|row| collect_row_pixels_to_remove(image, row, width, pass))
        .reduce(Vec::new, append_pixels)
}

// Process one row of candidate pixels.
// Keeping this separate makes the parallel collection easier to read.
fn collect_row_pixels_to_remove(
    image: &Array2<bool>,
    row: usize,
    width: usize,
    pass: ThinningPass,
) -> Vec<(usize, usize)> {
    let mut pixels = Vec::new();

    for col in 1..width - 1 {
        if should_remove_pixel(image, row, col, pass) {
            pixels.push((row, col));
        }
    }

    pixels
}

fn append_pixels(
    mut left: Vec<(usize, usize)>,
    mut right: Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    left.append(&mut right);
    left
}

// Remove all selected pixels simultaneously to avoid order bias.
fn remove_pixels(image: &mut Array2<bool>, pixels: &[(usize, usize)]) {
    for &(row, col) in pixels {
        image[(row, col)] = false;
    }
}

// Zhang-Suen thinning rules: remove pixels without breaking vessel connectivity.
fn should_remove_pixel(image: &Array2<bool>, row: usize, col: usize, pass: ThinningPass) -> bool {
    if !image[(row, col)] {
        return false;
    }

    let neighborhood = Neighborhood::from_image(image, row, col);
    let neighbor_count = neighborhood.count_neighbors();

    // Too few neighbors means endpoint-like structure, too many means dense interior.
    if !(2..=6).contains(&neighbor_count) {
        return false;
    }

    // One transition means removing this pixel should preserve one connected component.
    if neighborhood.count_transitions() != 1 {
        return false;
    }

    match pass {
        ThinningPass::First => neighborhood.can_remove_first_pass(),
        ThinningPass::Second => neighborhood.can_remove_second_pass(),
    }
}

#[derive(Debug, Clone, Copy)]
struct Neighborhood {
    p2: bool,
    p3: bool,
    p4: bool,
    p5: bool,
    p6: bool,
    p7: bool,
    p8: bool,
    p9: bool,
}

impl Neighborhood {
    // Store neighboring pixels in Zhang-Suen order:
    // p2 starts above center, then proceeds clockwise to p9.
    fn from_image(image: &Array2<bool>, row: usize, col: usize) -> Self {
        Self {
            p2: image[(row - 1, col)],
            p3: image[(row - 1, col + 1)],
            p4: image[(row, col + 1)],
            p5: image[(row + 1, col + 1)],
            p6: image[(row + 1, col)],
            p7: image[(row + 1, col - 1)],
            p8: image[(row, col - 1)],
            p9: image[(row - 1, col - 1)],
        }
    }

    fn as_array(&self) -> [bool; 8] {
        [
            self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9,
        ]
    }

    // Count how many surrounding pixels are still part of the vessel.
    fn count_neighbors(&self) -> usize {
        self.as_array().iter().filter(|&&pixel| pixel).count()
    }

    // Count 0→1 transitions around the neighborhood ring.
    // This is the connectivity-preserving part of Zhang-Suen thinning.
    fn count_transitions(&self) -> usize {
        let neighbors = self.as_array();
        let mut transitions = 0;

        for index in 0..8 {
            let current = neighbors[index];
            let next = neighbors[(index + 1) % 8];

            if !current && next {
                transitions += 1;
            }
        }

        transitions
    }

    // First pass removes one side of thick structures.
    fn can_remove_first_pass(&self) -> bool {
        !((self.p2 && self.p4 && self.p6) || (self.p4 && self.p6 && self.p8))
    }

    // Second pass removes the complementary side so thinning stays centered.
    fn can_remove_second_pass(&self) -> bool {
        !((self.p2 && self.p4 && self.p8) || (self.p2 && self.p6 && self.p8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn preserves_empty_mask() {
        let mask = Array2::<bool>::from_elem((5, 5), false);
        let skeleton = skeletonize(&mask);

        assert_eq!(skeleton, mask);
    }

    #[test]
    fn thins_solid_block() {
        let mask = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let skeleton = skeletonize(&mask);
        let count = skeleton.iter().filter(|&&pixel| pixel).count();

        assert!(count < 9);
        assert!(count > 0);
    }
}
