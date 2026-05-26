use ndarray::Array2;
use std::collections::VecDeque;

const BACKGROUND_SOURCE: isize = -1;

/// Estimate local vessel radius from a binary vessel mask.
///
/// Background pixels start as BFS sources. Their coordinates propagate into the
/// vessel mask, and each vessel pixel stores the Euclidean distance back to its
/// nearest discovered background pixel. At skeleton pixels, this distance is an
/// approximate local vessel radius in pixels.
pub fn distance_transform(mask: &Array2<bool>) -> Array2<f32> {
    let (height, width) = mask.dim();

    if height == 0 || width == 0 {
        return Array2::<f32>::zeros((height, width));
    }

    let mut nearest = NearestBackground::new(height, width);
    let mut queue = initialize_background_sources(mask, &mut nearest);
    propagate_background_sources(mask.dim(), &mut nearest, &mut queue);

    compute_distances(mask, &nearest)
}

struct NearestBackground {
    row: Array2<isize>,
    col: Array2<isize>,
    visited: Array2<bool>,
}

impl NearestBackground {
    fn new(height: usize, width: usize) -> Self {
        Self {
            row: Array2::<isize>::from_elem((height, width), BACKGROUND_SOURCE),
            col: Array2::<isize>::from_elem((height, width), BACKGROUND_SOURCE),
            visited: Array2::<bool>::from_elem((height, width), false),
        }
    }

    fn mark_source(&mut self, row: usize, col: usize) {
        self.visited[(row, col)] = true;
        self.row[(row, col)] = row as isize;
        self.col[(row, col)] = col as isize;
    }

    fn inherit_source(&mut self, row: usize, col: usize, source_row: usize, source_col: usize) {
        self.visited[(row, col)] = true;
        self.row[(row, col)] = self.row[(source_row, source_col)];
        self.col[(row, col)] = self.col[(source_row, source_col)];
    }

    fn source_at(&self, row: usize, col: usize) -> Option<(isize, isize)> {
        let source_row = self.row[(row, col)];
        let source_col = self.col[(row, col)];

        if source_row == BACKGROUND_SOURCE || source_col == BACKGROUND_SOURCE {
            None
        } else {
            Some((source_row, source_col))
        }
    }
}

fn initialize_background_sources(
    mask: &Array2<bool>,
    nearest: &mut NearestBackground,
) -> VecDeque<(usize, usize)> {
    let (height, width) = mask.dim();
    let mut queue = VecDeque::<(usize, usize)>::new();

    // All background pixels start the BFS at the same time.
    // That makes the first source reaching a vessel pixel its nearest boundary estimate.
    for row in 0..height {
        for col in 0..width {
            if !mask[(row, col)] {
                nearest.mark_source(row, col);
                queue.push_back((row, col));
            }
        }
    }

    queue
}

fn propagate_background_sources(
    shape: (usize, usize),
    nearest: &mut NearestBackground,
    queue: &mut VecDeque<(usize, usize)>,
) {
    let (height, width) = shape;

    // Propagate source coordinates instead of storing full paths.
    // This keeps the BFS cheap while still letting us compute radius afterward.
    while let Some((row, col)) = queue.pop_front() {
        for (next_row, next_col) in valid_neighbors(row, col, height, width) {
            if nearest.visited[(next_row, next_col)] {
                continue;
            }

            nearest.inherit_source(next_row, next_col, row, col);
            queue.push_back((next_row, next_col));
        }
    }
}

fn compute_distances(mask: &Array2<bool>, nearest: &NearestBackground) -> Array2<f32> {
    let (height, width) = mask.dim();
    let mut distances = Array2::<f32>::zeros((height, width));

    for row in 0..height {
        for col in 0..width {
            if !mask[(row, col)] {
                continue;
            }

            let Some((source_row, source_col)) = nearest.source_at(row, col) else {
                continue;
            };

            // Distance from vessel pixel to nearest background pixel ≈ local radius.
            distances[(row, col)] = euclidean_distance(row, col, source_row, source_col);
        }
    }

    distances
}

fn euclidean_distance(row: usize, col: usize, source_row: isize, source_col: isize) -> f32 {
    let dy = row as f32 - source_row as f32;
    let dx = col as f32 - source_col as f32;

    (dx * dx + dy * dy).sqrt()
}

// Return all valid 8-connected neighbors around one pixel.
// A small fixed array avoids allocating a Vec during every BFS expansion.
fn valid_neighbors(
    row: usize,
    col: usize,
    height: usize,
    width: usize,
) -> impl Iterator<Item = (usize, usize)> {
    const OFFSETS: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    OFFSETS
        .into_iter()
        .filter_map(move |(row_offset, col_offset)| {
            let next_row = row as isize + row_offset;
            let next_col = col as isize + col_offset;

            if next_row < 0 || next_col < 0 {
                return None;
            }

            let next_row = next_row as usize;
            let next_col = next_col as usize;

            if next_row < height && next_col < width {
                Some((next_row, next_col))
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Empty masks should produce zero radius everywhere.
    #[test]
    fn background_pixels_have_zero_distance() {
        let mask = Array2::<bool>::from_elem((3, 3), false);
        let distances = distance_transform(&mask);

        assert!(distances.iter().all(|&distance| distance == 0.0));
    }

    // Interior pixels should have larger radii than edge-adjacent pixels.
    #[test]
    fn center_of_solid_block_has_positive_distance() {
        let mask = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let distances = distance_transform(&mask);

        assert_eq!(distances[(0, 0)], 0.0);
        assert!(distances[(2, 2)] > distances[(1, 1)]);
    }
}
