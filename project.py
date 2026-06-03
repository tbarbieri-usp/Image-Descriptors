from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.cluster import vq


@dataclass(frozen=True)
class PetRecord:
	class_id: int
	class_name: str
	filename: str
	path: Path


@dataclass(frozen=True)
class FeatureBundle:
	haralick: np.ndarray
	lbp_r1: np.ndarray
	lbp_r2: np.ndarray
	hog: np.ndarray
	bovw: np.ndarray
	color_moments: np.ndarray
	gabor: np.ndarray
	hsv_histogram: np.ndarray


def load_pet_records(csv_path: str | Path, image_root: str | Path = "pets256") -> list[PetRecord]:
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


def haralick_features(image: np.ndarray, levels: int = 32) -> np.ndarray:
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
	return np.vstack([getattr(bundle, name) for bundle in bundles])


def standardize_features(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	mean = matrix.mean(axis=0)
	std = matrix.std(axis=0)
	std[std == 0] = 1.0
	return (matrix - mean) / std, mean, std


def apply_standardization(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	return (matrix - mean) / std


def train_nearest_centroid(features: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
	label_array = np.asarray(labels)
	centroids: dict[str, np.ndarray] = {}
	for class_name in sorted(set(labels)):
		class_features = features[label_array == class_name]
		centroids[class_name] = class_features.mean(axis=0)
	return centroids


def predict_nearest_centroid(features: np.ndarray, centroids: dict[str, np.ndarray]) -> list[str]:
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