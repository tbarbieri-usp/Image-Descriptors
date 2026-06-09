import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Iterable

import pandas as pd
import numpy as np

from PIL import Image
from scipy import ndimage
from scipy.cluster import vq
from scipy.spatial.distance import cdist

from umap import UMAP

from itertools import product, combinations

import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass(frozen=True)
class PetRecord:
	"""
	Represents a single image entry in the dataset.

	Attributes
	----------
	class_id : int
		Numeric identifier of the class.

	class_name : str
		Human-readable class name
		(e.g., 'Persian Cat').

	filename : str
		Image filename stored in the CSV.

	path : Path
		Full path to the image file.
	"""	
	class_id: int
	class_name: str
	filename: str
	path: Path


@dataclass(frozen=True)
class FeatureBundle:
    """
    Container that stores all descriptors extracted
    from a single image.

    Keeping all descriptors together makes it easier
    to compare individual descriptors or combine them
    later during classification and retrieval tasks.
    """

    # Texture descriptor based on gray-level co-occurrence matrices
    haralick: np.ndarray

    # Local Binary Pattern histogram (radius = 1)
    lbp_r1: np.ndarray

    # Local Binary Pattern histogram (radius = 2)
    lbp_r2: np.ndarray

    # Histogram of Oriented Gradients
    hog: np.ndarray

    # Bag of Visual Words histogram
    bovw: np.ndarray

    # Mean, standard deviation and skewness
    # extracted from LAB channels
    color_moments: np.ndarray

    # Multi-scale texture responses obtained
    # using Gabor filters
    gabor: np.ndarray

    # HSV color histogram
    hsv_histogram: np.ndarray


def load_pet_records(csv_path: str | Path,
                     image_root: str | Path = "pets256") -> list[PetRecord]:
	"""
	Reads the metadata CSV and creates a list of
	PetRecord objects.

	Each row in the CSV corresponds to a single image.
	The resulting structure keeps both class information
	and the path required to load the image later.
	"""
	csv_path = Path(csv_path)
	image_root = Path(image_root)

	records: list[PetRecord] = []
	with csv_path.open(newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file, skipinitialspace=True)
		for row in reader:
			class_id = int(row["class_id"].strip())
			class_name = row["class_name"].strip()
			filename = row["filename"].strip()
			records.append(
				PetRecord(
					class_id=class_id,
					class_name=class_name,
					filename=filename,
					path=image_root / filename,
				)
			)

	return records


def split_pets_by_class(
	records: Iterable[PetRecord],
	test_ratio: float = 0.1,
	validation_ratio: float = 0.1,
	min_images: int = 3,
	seed: int = 42,
) -> tuple[list[PetRecord], list[PetRecord], list[PetRecord], list[str]]:
	"""
	Performs a stratified split.

	Images belonging to the same class are grouped
	together before splitting so that every class
	contributes samples to:

		- Training set
		- Validation set
		- Test set

	Classes with too few images are ignored because
	they cannot be reliably distributed across the
	three subsets.
	"""
	grouped: dict[str, list[PetRecord]] = defaultdict(list)
	for record in records:
		grouped[record.class_name].append(record)

	rng = random.Random(seed)
	train_records: list[PetRecord] = []
	test_records: list[PetRecord] = []
	validation_records: list[PetRecord] = []
	skipped_classes: list[str] = []

	for class_name in sorted(grouped):
		class_records = grouped[class_name]
		if len(class_records) < min_images:
			skipped_classes.append(class_name)
			continue

		shuffled_records = class_records[:]
		rng.shuffle(shuffled_records)

		test_size = max(1, round(len(shuffled_records) * test_ratio))
		validation_size = max(1, round(len(shuffled_records) * validation_ratio))
		train_size = len(shuffled_records) - test_size - validation_size
		if train_size < 1:
			train_size = 1
			required_reduction = test_size + validation_size + train_size - len(shuffled_records)
			while required_reduction > 0 and (test_size > 1 or validation_size > 1):
				if test_size >= validation_size and test_size > 1:
					test_size -= 1
				elif validation_size > 1:
					validation_size -= 1
				required_reduction -= 1
			train_size = len(shuffled_records) - test_size - validation_size

		test_end = test_size
		validation_end = test_end + validation_size

		test_records.extend(shuffled_records[:test_end])
		validation_records.extend(shuffled_records[test_end:validation_end])
		train_records.extend(shuffled_records[validation_end:])

	return train_records, test_records, validation_records, skipped_classes


def load_rgb_image(path: Path) -> np.ndarray:
	try:
		with Image.open(path) as image:
			return np.asarray(image.convert("RGB"))
	except OSError as exc:
		raise FileNotFoundError(f"Could not read image: {path}")


def to_gray(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image.astype(np.uint8, copy=False)
	return np.asarray(Image.fromarray(image).convert("L"))


def haralick_features(image: np.ndarray,
                      levels: int = 32) -> np.ndarray:
	"""
	Extracts Haralick texture descriptors.

	Workflow
	--------
	1. Convert image to grayscale.
	2. Quantize intensities into a smaller number
		of gray levels.
	3. Build Gray-Level Co-occurrence Matrices (GLCM)
		for multiple directions.
	4. Compute statistical texture measures.
	5. Average descriptors across directions.

	Returned Features
	-----------------
	max_p      : Maximum probability
	corr       : Correlation
	contr      : Contrast
	energ      : Energy
	homog      : Homogeneity
	entropy    : Entropy
	"""	
	gray = to_gray(image)
	step = max(1, 256 // levels)
	quantized = np.clip(gray // step, 0, levels - 1).astype(np.int32)

	offsets = ((0, 1), (1, 0), (1, 1), (-1, 1))
	features: list[np.ndarray] = []
	row_idx = np.arange(levels, dtype=np.float64)[:, None]
	col_idx = np.arange(levels, dtype=np.float64)[None, :]

	for row_offset, col_offset in offsets:
		if row_offset >= 0:
			src = quantized[: quantized.shape[0] - row_offset, :]
			dst = quantized[row_offset:, :]
		else:
			src = quantized[-row_offset:, :]
			dst = quantized[: quantized.shape[0] + row_offset, :]

		if col_offset >= 0:
			src = src[:, : src.shape[1] - col_offset]
			dst = dst[:, col_offset:]
		else:
			src = src[:, -col_offset:]
			dst = dst[:, : dst.shape[1] + col_offset]

		cooccurrence = np.zeros((levels, levels), dtype=np.float64)
		np.add.at(cooccurrence, (src.ravel(), dst.ravel()), 1.0)
		cooccurrence += cooccurrence.T
		total = cooccurrence.sum()
		if total > 0:
			cooccurrence /= total

		max_p = float(cooccurrence.max())
		px = cooccurrence.sum(axis=1)
		py = cooccurrence.sum(axis=0)
		mean_x = float((np.arange(levels, dtype=np.float64) * px).sum())
		mean_y = float((np.arange(levels, dtype=np.float64) * py).sum())
		sigma_x = float(np.sqrt((((np.arange(levels, dtype=np.float64) - mean_x) ** 2) * px).sum()))
		sigma_y = float(np.sqrt((((np.arange(levels, dtype=np.float64) - mean_y) ** 2) * py).sum()))
		if sigma_x > 0 and sigma_y > 0:
			corr = float((((row_idx - mean_x) * (col_idx - mean_y) * cooccurrence).sum()) / (sigma_x * sigma_y))
		else:
			corr = 0.0
		contr = float((((row_idx - col_idx) ** 2) * cooccurrence).sum())
		energ = float(np.square(cooccurrence).sum())
		homog = float((cooccurrence / (1.0 + np.abs(row_idx - col_idx))).sum())
		non_zero = cooccurrence[cooccurrence > 0]
		entropy = float(-(non_zero * np.log2(non_zero)).sum()) if non_zero.size else 0.0
		features.append(np.array([max_p, corr, contr, energ, homog, entropy], dtype=np.float64))

	return np.mean(features, axis=0)


def lbp_features(image: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
	"""
	Computes Local Binary Pattern (LBP) features.

	For each pixel, neighboring pixels are compared
	against the center pixel.

	Neighbor >= Center -> bit = 1
	Neighbor <  Center -> bit = 0

	The resulting binary pattern is converted into
	an integer code. A histogram of all codes forms
	the descriptor.

	LBP is widely used for texture recognition.
	"""
	gray = to_gray(image).astype(np.float32)
	if gray.shape[0] <= 2 * radius or gray.shape[1] <= 2 * radius:
		return np.zeros(2**points, dtype=np.float64)

	center = gray[radius:-radius, radius:-radius]
	rows = np.arange(radius, gray.shape[0] - radius, dtype=np.float32)
	cols = np.arange(radius, gray.shape[1] - radius, dtype=np.float32)
	grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
	code = np.zeros_like(center, dtype=np.uint32)

	for bit in range(points):
		angle = 2.0 * np.pi * bit / points
		sample_rows = grid_rows - radius * np.sin(angle)
		sample_cols = grid_cols + radius * np.cos(angle)
		sampled = ndimage.map_coordinates(gray, [sample_rows, sample_cols], order=1, mode="reflect")
		code |= (sampled >= center).astype(np.uint32) << bit

	histogram = np.bincount(code.ravel(), minlength=2**points).astype(np.float64)
	total = histogram.sum()
	if total > 0:
		histogram /= total
	return histogram


def color_moments_features(image: np.ndarray) -> np.ndarray:
	"""
	Extracts Color Moments from the image.

	Color Moments summarize the distribution of pixel
	intensities in each color channel using three statistics:

	1. Mean      -> average color intensity
	2. Standard Deviation -> color variation
	3. Skewness  -> asymmetry of the distribution

	The image is first converted to the LAB color space,
	which separates luminance (L) from chromatic information
	(A and B channels).

	For each LAB channel, the descriptor computes:
		[mean, std, skewness]

	Result:
		3 channels × 3 statistics = 9 features
	"""
	lab = np.asarray(Image.fromarray(image).convert("LAB"), dtype=np.float32)
	channels = [lab[:, :, index].reshape(-1) for index in range(3)]
	features: list[float] = []
	for channel in channels:
		mean = float(channel.mean())
		std = float(channel.std())
		centered = channel - mean
		if std > 0:
			skew = float(np.mean((centered / std) ** 3))
		else:
			skew = 0.0
		features.extend([mean, std, skew])
	return np.asarray(features, dtype=np.float64)


def hog_features(image: np.ndarray, bins: int = 8) -> np.ndarray:
	"""
	Computes a simplified Histogram of Oriented Gradients.

	Steps
	-----
	1. Estimate image gradients using Sobel filters.
	2. Compute gradient magnitude and orientation.
	3. Accumulate magnitudes into orientation bins.
	4. Normalize the histogram.

	Captures edge structure and object shape.
	"""
	gray = to_gray(image).astype(np.float32)
	grad_x = ndimage.sobel(gray, axis=1, mode="reflect")
	grad_y = ndimage.sobel(gray, axis=0, mode="reflect")
	magnitude = np.hypot(grad_x, grad_y)
	angle = (np.degrees(np.arctan2(grad_y, grad_x)) + 360.0) % 360.0
	histogram, _ = np.histogram(angle, bins=bins, range=(0.0, 360.0), weights=magnitude)
	histogram = histogram.astype(np.float64)
	total = histogram.sum()
	if total > 0:
		histogram /= total
	return histogram


def gabor_features(
	image: np.ndarray,
	frequencies: tuple[float, ...] = (0.08, 0.16),
	orientations: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
	sigma: float = 4.0,
	gamma: float = 0.5,
) -> np.ndarray:
	"""
	Extracts texture descriptors using a bank of
	Gabor filters.

	Gabor filters act like orientation- and frequency-
	selective detectors, similar to the receptive fields
	found in the human visual cortex.

	Each filter responds strongly to texture patterns
	with a specific:

		- Orientation (direction)
		- Frequency (scale)

	For every filter response, two statistics are stored:

		1. Mean response magnitude
		2. Standard deviation of response magnitude

	The final descriptor concatenates all responses from
	the filter bank.
	"""
	gray = to_gray(image).astype(np.float32) / 255.0
	features: list[float] = []
	kernel_radius = max(4, int(round(3 * sigma)))
	axis = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
	x, y = np.meshgrid(axis, axis)

	for orientation in orientations:
		cos_theta = np.cos(orientation)
		sin_theta = np.sin(orientation)
		x_theta = x * cos_theta + y * sin_theta
		y_theta = -x * sin_theta + y * cos_theta
		for frequency in frequencies:
			envelope = np.exp(-((x_theta**2) + (gamma**2) * (y_theta**2)) / (2.0 * sigma**2))
			carrier = np.exp(1j * (2.0 * np.pi * frequency * x_theta))
			kernel = envelope * carrier
			response_real = ndimage.convolve(gray, np.real(kernel), mode="reflect")
			response_imag = ndimage.convolve(gray, np.imag(kernel), mode="reflect")
			magnitude = np.hypot(response_real, response_imag)
			features.extend([float(magnitude.mean()), float(magnitude.std())])

	return np.asarray(features, dtype=np.float64)


def hsv_histogram_features(image: np.ndarray, bins: int = 6) -> np.ndarray:
	"""
	Computes a color histogram in HSV space.

	HSV separates color information into:

		H -> Hue (color type)
		S -> Saturation (color purity)
		V -> Value (brightness)

	A histogram is computed independently for
	each channel and then concatenated.

	The final histogram is normalized so that
	it becomes independent of image size.

	Result:
		3 channels × bins
	"""
	hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.float32)
	channel_histograms: list[np.ndarray] = []
	for channel_index in range(3):
		channel = hsv[:, :, channel_index].ravel()
		histogram, _ = np.histogram(channel, bins=bins, range=(0.0, 255.0))
		channel_histograms.append(histogram.astype(np.float64))

	features = np.concatenate(channel_histograms)
	total = features.sum()
	if total > 0:
		features /= total
	return features


def region_lbp_features(image: np.ndarray, region_size: int = 32, points: int = 8, radius: int = 1) -> np.ndarray:
	"""
	Extracts LBP descriptors from multiple image regions.

	Instead of describing the entire image with a
	single LBP histogram, the image is divided into
	small patches.

	Each patch generates its own LBP descriptor.

	This preserves some spatial information and is
	particularly useful for Bag of Visual Words (BoVW).
	"""
	height, width = image.shape[:2]
	regions: list[np.ndarray] = []
	for row_start in range(0, height, region_size):
		for col_start in range(0, width, region_size):
			region = image[row_start : row_start + region_size, col_start : col_start + region_size]
			if region.size == 0:
				continue
			regions.append(lbp_features(region, points=points, radius=radius))
	if not regions:
		return np.zeros((0, 2**points), dtype=np.float64)
	return np.asarray(regions, dtype=np.float64)


def build_bovw_codebook(
	records: Iterable[PetRecord],
	region_size: int = 32,
	clusters: int = 8,
	points: int = 8,
	radius: int = 1,
) -> np.ndarray:
	"""
	Creates the visual vocabulary used by
	the Bag of Visual Words model.

	Process
	-------
	1. Divide images into small regions.
	2. Extract an LBP descriptor from each region.
	3. Collect descriptors from all training images.
	4. Cluster descriptors using k-means.
	5. Use cluster centers as visual words.

	The resulting codebook acts like a dictionary
	of common local texture patterns.
	"""
	region_features: list[np.ndarray] = []
	for record in records:
		image = load_rgb_image(record.path)
		region_features.extend(region_lbp_features(image, region_size, points, radius))

	if not region_features:
		return np.zeros((0, 2**points), dtype=np.float32)

	feature_matrix = np.asarray(region_features, dtype=np.float32)
	cluster_count = min(clusters, len(feature_matrix))
	centers, _labels = vq.kmeans2(feature_matrix, cluster_count, minit="points", iter=20)
	return centers.astype(np.float32)


def bovw_features(
	image: np.ndarray,
	codebook: np.ndarray,
	region_size: int = 32,
	points: int = 8,
	radius: int = 1,
) -> np.ndarray:
	"""
	Computes the Bag of Visual Words (BoVW) representation
	for a single image.

	Workflow
	--------
	1. Divide the image into small regions.
	2. Extract an LBP descriptor from each region.
	3. Compare each region descriptor against all visual
	words in the codebook.
	4. Assign the region to its nearest visual word.
	5. Count how many times each visual word appears.
	6. Normalize the resulting histogram.

	The final descriptor represents the image as a
	distribution of visual words, analogous to how a
	text document can be represented by word frequencies.
	"""

	# If no visual vocabulary exists,
	# return an empty descriptor.
	if codebook.size == 0:
		return np.zeros(0, dtype=np.float64)

	regions = region_lbp_features(image, region_size, points, radius)
	if regions.size == 0:
		return np.zeros(len(codebook), dtype=np.float64)

	distances = np.linalg.norm(regions[:, None, :] - codebook[None, :, :], axis=2)
	assignments = np.argmin(distances, axis=1)
	histogram = np.bincount(assignments, minlength=len(codebook)).astype(np.float64)
	total = histogram.sum()
	if total > 0:
		histogram /= total
	return histogram


def extract_feature_bundle(image: np.ndarray, codebook: np.ndarray) -> FeatureBundle:
	"""
	Extracts all descriptors for a single image and
	stores them inside a FeatureBundle object.

	This function acts as a central feature extraction
	pipeline, ensuring that all descriptors are computed
	consistently for every image.

	The resulting FeatureBundle can later be used for:

		- Descriptor evaluation
		- Descriptor fusion
		- Classification
		- Image retrieval
	"""
	return FeatureBundle(
		haralick=haralick_features(image),
		lbp_r1=lbp_features(image, points=8, radius=1),
		lbp_r2=lbp_features(image, points=8, radius=2),
		hog=hog_features(image),
		bovw=bovw_features(image, codebook),
		color_moments=color_moments_features(image),
		gabor=gabor_features(image),
		hsv_histogram=hsv_histogram_features(image),
	)


def extract_feature_bundle(
	image: np.ndarray,
	codebook: np.ndarray,
	*,
	haralick_levels: int = 32,
	lbp_points: int = 8,
	lbp_radius_r1: int = 1,
	lbp_radius_r2: int = 2,
	hog_bins: int = 8,
	bovw_region_size: int = 32,
	bovw_points: int = 8,
	bovw_radius: int = 1,
	gabor_frequencies: tuple[float, ...] = (0.08, 0.16),
	gabor_orientations: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
	gabor_sigma: float = 4.0,
	gabor_gamma: float = 0.5,
	hsv_bins: int = 6,
) -> FeatureBundle:
	"""
	Extracts all image descriptors using configurable
	parameters.

	Unlike the simplified version, this implementation
	allows descriptor hyperparameters to be adjusted,
	making it useful for experimentation and performance
	optimization.

	All extracted descriptors are grouped into a single
	FeatureBundle object.
	"""
	return FeatureBundle(
		haralick=haralick_features(image, levels=haralick_levels),
		lbp_r1=lbp_features(image, points=lbp_points, radius=lbp_radius_r1),
		lbp_r2=lbp_features(image, points=lbp_points, radius=lbp_radius_r2),
		hog=hog_features(image, bins=hog_bins),
		bovw=bovw_features(image, codebook, region_size=bovw_region_size, points=bovw_points, radius=bovw_radius),
		color_moments=color_moments_features(image),
		gabor=gabor_features(image, frequencies=gabor_frequencies, orientations=gabor_orientations, sigma=gabor_sigma, gamma=gabor_gamma),
		hsv_histogram=hsv_histogram_features(image, bins=hsv_bins),
	)


def extract_bundles(
	records: list[PetRecord],
	codebook: np.ndarray,
	*,
	haralick_levels: int = 32,
	lbp_points: int = 8,
	lbp_radius_r1: int = 1,
	lbp_radius_r2: int = 2,
	hog_bins: int = 8,
	bovw_region_size: int = 32,
	bovw_points: int = 8,
	bovw_radius: int = 1,
	gabor_frequencies: tuple[float, ...] = (0.08, 0.16),
	gabor_orientations: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
	gabor_sigma: float = 4.0,
	gabor_gamma: float = 0.5,
	hsv_bins: int = 6,
) -> tuple[list[str], list[FeatureBundle]]:
	"""
	Extracts descriptor bundles for an entire dataset.

	For each image:
		1. Load image from disk.
		2. Extract all descriptors.
		3. Store descriptors in a FeatureBundle.
		4. Save the corresponding class label.

	Returns
	-------
	labels:
		Ground-truth class labels.

	bundles:
		FeatureBundle objects containing all extracted
		descriptors for each image.

	The resulting data can be converted into feature
	matrices for classification or retrieval tasks.
	"""
	labels: list[str] = []
	bundles: list[FeatureBundle] = []
	for record in records:
		image = load_rgb_image(record.path)
		labels.append(record.class_name)
		bundles.append(
			extract_feature_bundle(
				image,
				codebook,
				haralick_levels=haralick_levels,
				lbp_points=lbp_points,
				lbp_radius_r1=lbp_radius_r1,
				lbp_radius_r2=lbp_radius_r2,
				hog_bins=hog_bins,
				bovw_region_size=bovw_region_size,
				bovw_points=bovw_points,
				bovw_radius=bovw_radius,
				gabor_frequencies=gabor_frequencies,
				gabor_orientations=gabor_orientations,
				gabor_sigma=gabor_sigma,
				gabor_gamma=gabor_gamma,
				hsv_bins=hsv_bins,
			),
		)
	return labels, bundles


def matrix_from_bundles(bundles: list[FeatureBundle], name: str) -> np.ndarray:
	"""
	Converts a descriptor stored inside multiple
	FeatureBundle objects into a feature matrix.

	Example:

		bundles[0].hog
		bundles[1].hog
		bundles[2].hog

	becomes:

		[
			hog_1,
			hog_2,
			hog_3
		]

	Each row corresponds to one image.
	"""
	return np.vstack([getattr(bundle, name) for bundle in bundles])


def standardize_features(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Applies z-score normalization.

	For each feature dimension:

		x' = (x - mean) / std

	This places all descriptor dimensions on a
	comparable scale, preventing features with
	larger numeric ranges from dominating the
	Euclidean distance calculation.
	"""
	mean = matrix.mean(axis=0)
	std = matrix.std(axis=0)
	std[std == 0] = 1.0
	return (matrix - mean) / std, mean, std


def apply_standardization(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	"""
	Applies previously computed normalization
	parameters to new data.

	Important:
		Test and validation sets must use the
		training-set statistics to avoid
		data leakage.
	"""
	return (matrix - mean) / std


def train_nearest_centroid(features: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
	"""
	Trains a Nearest Centroid classifier.

	For each class:
		centroid = mean(feature vectors)

	Classification is performed by assigning a sample
	to the class whose centroid is closest in Euclidean
	space.
	"""
	label_array = np.asarray(labels)
	centroids: dict[str, np.ndarray] = {}
	for class_name in sorted(set(labels)):
		class_features = features[label_array == class_name]
		centroids[class_name] = class_features.mean(axis=0)
	return centroids


def predict_nearest_centroid(features: np.ndarray, centroids: dict[str, np.ndarray]) -> list[str]:
	"""
	Classifies samples using the Nearest Centroid rule.

	For each feature vector:

		1. Compute Euclidean distance to all class centroids.
		2. Select the closest centroid.
		3. Assign the corresponding class.

	This is one of the simplest classification methods
	and works well when classes form compact clusters
	in feature space.
	"""
	class_names = list(centroids)
	centroid_matrix = np.vstack([centroids[name] for name in class_names])
	predictions: list[str] = []
	for row in features:
		distances = np.linalg.norm(centroid_matrix - row, axis=1)
		predictions.append(class_names[int(np.argmin(distances))])
	return predictions


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
	if not y_true:
		return 0.0
	return sum(true == pred for true, pred in zip(y_true, y_pred)) / len(y_true)


def evaluate_feature_set(
	name: str,
	train_matrix: np.ndarray,
	train_labels: list[str],
	validation_matrix: np.ndarray,
	validation_labels: list[str],
	test_matrix: np.ndarray,
	test_labels: list[str],
) -> tuple[float, float]:
	"""
	Evaluates a descriptor independently.

	Pipeline
	--------
	1. Standardize training features.
	2. Apply the same normalization to validation
		and test sets.
	3. Train a Nearest Centroid classifier.
	4. Compute validation accuracy.
	5. Compute test accuracy.

	Returns
	-------
	validation_score : float
	test_score : float
	"""
	train_scaled, mean, std = standardize_features(train_matrix)
	validation_scaled = apply_standardization(validation_matrix, mean, std)
	test_scaled = apply_standardization(test_matrix, mean, std)
	centroids = train_nearest_centroid(train_scaled, train_labels)
	validation_score = accuracy_score(validation_labels, predict_nearest_centroid(validation_scaled, centroids))
	test_score = accuracy_score(test_labels, predict_nearest_centroid(test_scaled, centroids))
	print(f"{name}: validation={validation_score:.3f} test={test_score:.3f}")
	return validation_score, test_score


def combine_feature_blocks(blocks: list[np.ndarray]) -> np.ndarray:
	return np.concatenate(blocks, axis=1)


# ==================================================
# VISUALIZATION UTILITIES
# ==================================================

########## GLOBAL ##########

def sample_images(records, num_samples=8):
    # Show a small grid of training images from different classes.
    sample_records = []
    seen_classes = set()
    for record in records:
        if record.class_name not in seen_classes:
            sample_records.append(record)
            seen_classes.add(record.class_name)
        if len(sample_records) == num_samples:
            break

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, record in zip(axes.flat, sample_records):
        image = load_rgb_image(record.path)
        ax.imshow(image)
        ax.set_title(record.class_name, fontsize=9)
        ax.axis('off')

    for ax in axes.flat[len(sample_records):]:
        ax.axis('off')

    fig.suptitle('Example images', y=1.02)
    fig.tight_layout()


def plot_classes_distribution(train_records):
    # Visualize how many samples each class contributes to the training split.
    train_counts = Counter(record.class_name for record in train_records)
    classes = sorted(train_counts)
    counts = [train_counts[class_name] for class_name in classes]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(classes, counts, color='#2a9d8f')
    ax.set_title('Training samples per class')
    ax.set_ylabel('Images')
    ax.tick_params(axis='x', rotation=90)
    fig.tight_layout()

########## CLASSIFICATION PIPELINE ##########
def iter_bovw_patches(image, region_size, points=8, radius=1):
    height, width = image.shape[:2]
    for row_start in range(0, height, region_size):
        for col_start in range(0, width, region_size):
            patch = image[row_start:row_start + region_size, col_start:col_start + region_size]
            if patch.size == 0:
                continue
            yield patch, lbp_features(patch, points=points, radius=radius)

def plot_bovw_representative_patches(bovw_codebook, train_records, region_size=32, points=8, radius=1):
    best_words = [None] * len(bovw_codebook)
    for record in train_records:
        image = load_rgb_image(record.path)
        for patch, feature in iter_bovw_patches(image, region_size, points, radius):
            if feature.size != bovw_codebook.shape[1]:
                continue
            distances = np.linalg.norm(bovw_codebook - feature, axis=1)
            word_index = int(np.argmin(distances))
            score = float(distances[word_index])
            current = best_words[word_index]
            if current is None or score < current[0]:
                best_words[word_index] = (score, patch, record.class_name)

    columns = min(4, len(best_words))
    rows = int(np.ceil(len(best_words) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index, ax in enumerate(axes.flat):
        if index >= len(best_words) or best_words[index] is None:
            ax.axis('off')
            continue
        score, patch, class_name = best_words[index]
        ax.imshow(patch)
        ax.set_title(f'Word {index} | {class_name} | {score:.2f}', fontsize=8)
        ax.axis('off')

    fig.suptitle('Representative patches for BoVW words', y=1.02)
    fig.tight_layout()

def plot_descriptor_comparison(labels, validation_scores, test_scores):
    # Compare validation and test accuracy for each descriptor.
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, validation_scores, width, label='Validation', color='#457b9d')
    ax.bar(x + width / 2, test_scores, width, label='Test', color='#e76f51')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Descriptor comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.legend()
    fig.tight_layout()


def plot_final_comparison(labels, scores):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, scores, color=['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#457b9d', '#e76f51', '#6a994e', '#8d99ae', '#ffb703'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Final comparison on the test split')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()


def plot_sample_predictions(records, true_labels, predicted_labels):
    # Select one sample from each class
    selected_indices = []

    for class_name in sorted(np.unique(true_labels)):
        class_indices = np.where(np.array(true_labels) == class_name)[0]

        rng = np.random.default_rng(42)
        selected_indices.append(rng.choice(class_indices))

    # Plot configuration
    cols = 4
    rows = int(np.ceil(len(selected_indices) / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, selected_indices):

        record = records[idx]

        image = load_rgb_image(record.path)

        true_label = true_labels[idx]
        predicted_label = predicted_labels[idx]

        correct = true_label == predicted_label

        ax.imshow(image)

        title_color = "green" if correct else "red"

        ax.set_title(
            f"GT: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"{'✓ Correct' if correct else '✗ Wrong'}",
            fontsize=9,
            color=title_color
        )

        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(selected_indices):]:
        ax.axis("off")

    fig.suptitle(
        "One Test Example Per Class",
        fontsize=14
    )

    fig.tight_layout()
    plt.show()

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print(f"Test Accuracy: {accuracy:.3f}")

########## SEARCH PIPELINE ##########
def plot_umaps(
    all_features,
    labels,
    descriptor_names,
    descriptor_labels,
    n_neighbors
):
    # Convert labels to NumPy array for indexing
    labels = np.asarray(labels)

    # Create a 2x4 grid of subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Generate one UMAP projection for each descriptor
    for ax, descriptor_name in zip(
        axes,
        descriptor_names
    ):
        # Feature vectors and corresponding labels
        X = all_features[descriptor_name]
        y = labels

        # Reduce feature dimensionality to 2D for visualization
        embedding = UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
        ).fit_transform(X)

        # Plot samples grouped by class
        for label in np.unique(y):
            mask = y == label

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.7,
                label=str(label)
            )

        # Configure subplot appearance
        ax.set_title(descriptor_labels[descriptor_name])
        ax.set_xticks([])
        ax.set_yticks([])

    # Create a single legend for all subplots
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        borderaxespad=0
    )

    # Adjust layout and display figure
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    plt.show()

def retrieve_ranking(query_idx, feature_matrix, top_k=10):
    # Feature vector of the query image
    query = feature_matrix[query_idx]

    # Compute Euclidean distance from the query to all images
    distances = cdist(
        query.reshape(1, -1),
        feature_matrix,
        metric='euclidean'
    ).flatten()

    # Sort images from most similar (smallest distance)
    # to least similar (largest distance)
    ranking = np.argsort(distances)

    # Remove the query image itself from the ranking
    ranking = ranking[ranking != query_idx]

    # Return the top-k retrieved images and all distances
    return ranking[:top_k], distances


def precision_at_k(query_idx, ranking, labels):
    # Class of the query image
    query_label = labels[query_idx]

    # Classes of the retrieved images
    retrieved_labels = labels[ranking]

    # Fraction of retrieved images belonging to the same class
    return np.mean(retrieved_labels == query_label)


def mean_precision_at_k(
    feature_matrix,
    labels,
    k=10
):
    # Store precision values for all queries
    precisions = []

    # Use each image as a query once
    for query_idx in range(len(labels)):

        # Retrieve the top-k most similar images
        ranking, _ = retrieve_ranking(
            query_idx,
            feature_matrix,
            k
        )

        # Compute Precision@k for the current query
        precisions.append(
            precision_at_k(
                query_idx,
                ranking,
                labels
            )
        )

    # Average Precision@k over all queries
    return np.mean(precisions)

def show_random_queries(
    feature_matrix,
    records,
    labels,
    n_queries=5,
    top_k=10,
):
    # Randomly select query images from the dataset
    query_indices = random.sample(
        range(len(labels)),
        n_queries
    )

    # Display retrieval results for each selected query
    for query_idx in query_indices:

        query_label = labels[query_idx]

        # Retrieve the top-k most similar images
        ranking, distances = retrieve_ranking(
            query_idx,
            feature_matrix,
            top_k=top_k,
        )

        # Create a row of plots:
        # 1 query image + top-k retrieved images
        fig, axes = plt.subplots(
            1,
            top_k + 1,
            figsize=(3 * (top_k + 1), 3)
        )

        # Display the query image
        axes[0].imshow(
            load_rgb_image(records[query_idx].path)
        )
        axes[0].set_title(
            f"Query\n{query_label}"
        )
        axes[0].axis("off")

        # Display retrieved images in ranking order
        for i, idx in enumerate(ranking):

            retrieved_label = labels[idx]

            axes[i + 1].imshow(
                load_rgb_image(records[idx].path)
            )

            # Green if class matches the query, red otherwise
            title_color = (
                "green"
                if retrieved_label == query_label
                else "red"
            )

            axes[i + 1].set_title(
                f"{retrieved_label}\n{distances[idx]:.2f}",
                color=title_color
            )

            axes[i + 1].axis("off")

        # Adjust spacing and show the result
        plt.tight_layout()
        plt.show()

def combine_features(
    feature_dict,
    feature_names,
):
    # Concatenate multiple feature matrices into a single representation
    return np.hstack([
        feature_dict[name]
        for name in feature_names
    ])


def evaluate_feature_combinations(
    feature_dict,
    labels,
    max_features=None,
    k=10,
):

    # List of available descriptors
    names = list(feature_dict.keys())

    # By default, test combinations using all descriptors
    if max_features is None:
        max_features = len(names)

    # Store evaluation results
    results = []

    # Test combinations of increasing size
    for size in tqdm(range(1, max_features + 1)):

        # Generate all descriptor combinations of the current size
        for combo in tqdm(combinations(names, size), leave=False):

            # Build the combined feature matrix
            X = combine_features(
                feature_dict,
                combo,
            )

            # Evaluate retrieval performance
            score = mean_precision_at_k(
                X,
                labels,
                k=k,
            )

            # Save combination and corresponding score
            results.append({
                "features": combo,
                "n_features": size,
                "precision": score,
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Return combinations sorted by Precision@k
    return df.sort_values(
        "precision",
        ascending=False
    )

def retrieve_weighted_fusion(
    query_idx,
    feature_dict,
    feature_names,
    weights,
    top_k=10,
):
    # Accumulate weighted distances from multiple descriptors
    total_distance = None

    for feature_name, weight in zip(feature_names, weights):

        # Feature matrix of the current descriptor
        X = feature_dict[feature_name]

        # Compute Euclidean distances from the query image
        d = cdist(
            X[query_idx].reshape(1, -1),
            X,
            metric="euclidean"
        ).flatten()

        # Add the weighted contribution of this descriptor
        if total_distance is None:
            total_distance = weight * d
        else:
            total_distance += weight * d

    # Sort images by combined distance
    ranking = np.argsort(total_distance)

    # Remove the query image itself
    ranking = ranking[ranking != query_idx]

    # Return the top-k retrieved images
    return ranking[:top_k]


def mean_precision_at_k_weighted(
    feature_dict,
    feature_names,
    weights,
    labels,
    k=10,
):
    # Store Precision@k values for all queries
    scores = []

    labels = np.asarray(labels)

    # Use each image as a query
    for query_idx in range(len(labels)):

        # Retrieve images using weighted descriptor fusion
        ranking = retrieve_weighted_fusion(
            query_idx,
            feature_dict,
            feature_names,
            weights,
            top_k=k,
        )

        # Compute Precision@k for the current query
        precision = np.mean(
            labels[ranking] == labels[query_idx]
        )

        scores.append(precision)

    # Average Precision@k over all queries
    return np.mean(scores)

def search_best_weights(
    feature_dict,
    feature_names,
    labels,
    candidate_weights=(0.25, 0.5, 1.0, 2.0, 4.0),
    k=10,
):
    # Store the evaluation results for each weight combination
    results = []

    # Test all possible combinations of candidate weights
    for weights in tqdm(
        product(candidate_weights, repeat=len(feature_names)),
        total=len(candidate_weights) ** len(feature_names),
        desc="Weight search",
    ):

        # Evaluate retrieval performance using the current weights
        score = mean_precision_at_k_weighted(
            feature_dict,
            feature_names,
            weights,
            labels,
            k=k,
        )

        # Save weights and corresponding Precision@k
        results.append({
            "weights": weights,
            "precision": score,
        })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Return weight combinations sorted by performance
    return df.sort_values(
        "precision",
        ascending=False,
    )


def plot_umap(
    embedding,
    labels,
    features,
):
    # Convert labels to NumPy array for indexing
    labels = np.asarray(labels)

    # Create the visualization figure
    plt.figure(figsize=(10, 8))

    # Plot one group of points per class
    for label in np.unique(labels):

        mask = labels == label

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=15,          # Marker size
            alpha=0.7,     # Transparency
            label=label,
        )

    # Display the descriptor(s) used to generate the embedding
    plt.title(
        f"UMAP - {' + '.join(features) if type(features) is list else features}"
    )

    # Place legend outside the plot area
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # Adjust layout and display figure
    plt.tight_layout()
    plt.show()