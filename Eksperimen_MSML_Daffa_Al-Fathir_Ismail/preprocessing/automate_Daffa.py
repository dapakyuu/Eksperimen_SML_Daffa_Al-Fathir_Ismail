import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RAW_DATA_DIRNAME = "Vegetables_raw"


@dataclass(frozen=True)
class DatasetPaths:
	root: str
	train_dir: str
	test_dir: str
	val_dir: str


def _is_image_file(filename: str) -> bool:
	return filename.lower().endswith(IMAGE_EXTENSIONS)


def _list_subdirs(path: str) -> List[str]:
	if not os.path.isdir(path):
		return []
	return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def resolve_dataset_paths(dataset_dir: str) -> DatasetPaths:
	"""Resolve dataset root folder.

	Supports 2 common layouts:
	1) dataset_dir/Vegetables_raw/{train,test,validation}/<class>/image.jpg
	2) dataset_dir/{train,test,validation}/<class>/image.jpg
	"""
	dataset_dir = os.path.abspath(dataset_dir)

	direct_train = os.path.join(dataset_dir, "train")
	direct_test = os.path.join(dataset_dir, "test")
	direct_val = os.path.join(dataset_dir, "validation")
	if all(os.path.isdir(p) for p in (direct_train, direct_test, direct_val)):
		return DatasetPaths(
			root=dataset_dir,
			train_dir=direct_train,
			test_dir=direct_test,
			val_dir=direct_val,
		)

	nested_root = os.path.join(dataset_dir, RAW_DATA_DIRNAME)
	nested_train = os.path.join(nested_root, "train")
	nested_test = os.path.join(nested_root, "test")
	nested_val = os.path.join(nested_root, "validation")
	if all(os.path.isdir(p) for p in (nested_train, nested_test, nested_val)):
		return DatasetPaths(
			root=nested_root,
			train_dir=nested_train,
			test_dir=nested_test,
			val_dir=nested_val,
		)

	raise FileNotFoundError(
		"Tidak menemukan struktur dataset yang valid. "
		"Pastikan salah satu struktur berikut ada:\n"
		"1) <dataset_dir>/Vegetables_raw/train|test|validation\n"
		"2) <dataset_dir>/train|test|validation\n"
		f"dataset_dir yang diberikan: {dataset_dir}"
	)


def create_dataframe_from_directory(base_dir: str, categories: Iterable[str]) -> pd.DataFrame:
	"""Create a dataframe with image_path + label from a split directory."""
	rows: List[Dict[str, str]] = []
	for category in categories:
		category_path = os.path.join(base_dir, category)
		if not os.path.isdir(category_path):
			continue
		for filename in os.listdir(category_path):
			if _is_image_file(filename):
				rows.append(
					{
						"image_path": os.path.abspath(os.path.join(category_path, filename)),
						"label": category,
					}
				)
	return pd.DataFrame(rows)


def compute_split_counts(split_dir: str, categories: List[str]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for category in categories:
		category_path = os.path.join(split_dir, category)
		if not os.path.isdir(category_path):
			counts[category] = 0
			continue
		counts[category] = len([f for f in os.listdir(category_path) if _is_image_file(f)])
	return counts


def save_label_mapping(label_encoder: LabelEncoder, output_dir: str) -> str:
	os.makedirs(output_dir, exist_ok=True)
	mapping_path = os.path.join(output_dir, "label_mapping.json")
	label_mapping = {int(i): label for i, label in enumerate(label_encoder.classes_)}
	with open(mapping_path, "w", encoding="utf-8") as f:
		json.dump(label_mapping, f, indent=4, ensure_ascii=False)
	return mapping_path


def preprocess_vegetable_images(
	dataset_dir: str,
	output_dir: str,
	*,
	write_summary: bool = True,
) -> Tuple[str, str, str, str]:
	"""Automate preprocessing for Vegetable Image Dataset.

	Output is ready-to-train metadata (paths + encoded labels) in CSV files.
	Image resizing/normalization is expected to be done on-the-fly at training time.

	Returns:
		(train_csv_path, test_csv_path, val_csv_path, mapping_path)
	"""
	paths = resolve_dataset_paths(dataset_dir)

	categories = _list_subdirs(paths.train_dir)
	if not categories:
		raise FileNotFoundError(
			f"Tidak ada kategori (folder kelas) di train_dir: {paths.train_dir}"
		)

	df_train = create_dataframe_from_directory(paths.train_dir, categories)
	df_test = create_dataframe_from_directory(paths.test_dir, categories)
	df_val = create_dataframe_from_directory(paths.val_dir, categories)

	if df_train.empty:
		raise FileNotFoundError(
			f"Train dataframe kosong. Pastikan ada gambar di: {paths.train_dir}"
		)

	label_encoder = LabelEncoder()
	label_encoder.fit(categories)

	df_train["label_encoded"] = label_encoder.transform(df_train["label"])
	df_test["label_encoded"] = label_encoder.transform(df_test["label"])
	df_val["label_encoded"] = label_encoder.transform(df_val["label"])

	os.makedirs(output_dir, exist_ok=True)
	train_csv_path = os.path.join(output_dir, "train_ready.csv")
	test_csv_path = os.path.join(output_dir, "test_ready.csv")
	val_csv_path = os.path.join(output_dir, "val_ready.csv")

	df_train.to_csv(train_csv_path, index=False)
	df_test.to_csv(test_csv_path, index=False)
	df_val.to_csv(val_csv_path, index=False)

	mapping_path = save_label_mapping(label_encoder, output_dir)

	if write_summary:
		train_counts = compute_split_counts(paths.train_dir, categories)
		test_counts = compute_split_counts(paths.test_dir, categories)
		val_counts = compute_split_counts(paths.val_dir, categories)
		summary = {
			"dataset_root": paths.root,
			"num_categories": len(categories),
			"categories": categories,
			"splits": {
				"train": {"total": int(df_train.shape[0]), "per_category": train_counts},
				"test": {"total": int(df_test.shape[0]), "per_category": test_counts},
				"validation": {"total": int(df_val.shape[0]), "per_category": val_counts},
			},
			"outputs": {
				"train_csv": os.path.abspath(train_csv_path),
				"test_csv": os.path.abspath(test_csv_path),
				"val_csv": os.path.abspath(val_csv_path),
				"label_mapping": os.path.abspath(mapping_path),
			},
		}

		summary_path = os.path.join(output_dir, "preprocessing_summary.json")
		with open(summary_path, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2, ensure_ascii=False)

	return train_csv_path, test_csv_path, val_csv_path, mapping_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Automated preprocessing for Vegetable Image Dataset. "
			"Outputs CSV metadata (paths + labels) ready for training."
		)
	)
	parser.add_argument(
		"--dataset_dir",
		type=str,
		default=f"../{RAW_DATA_DIRNAME}",
		help=(
			"Folder dataset. Bisa menunjuk ke folder yang berisi 'Vegetables_raw', "
			"atau langsung ke folder yang berisi train/test/validation. "
			"Default: ../Vegetables_raw"
		),
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="Vegetables_preprocessing",
		help="Folder output untuk file CSV & label mapping. Default: Vegetables_preprocessing",
	)
	parser.add_argument(
		"--no_summary",
		action="store_true",
		help="Jika di-set, tidak menulis preprocessing_summary.json",
	)
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()

	train_csv, test_csv, val_csv, mapping_path = preprocess_vegetable_images(
		dataset_dir=args.dataset_dir,
		output_dir=args.output_dir,
		write_summary=not args.no_summary,
	)

	print("✅ Preprocessing selesai")
	print(f"- Train CSV      : {train_csv}")
	print(f"- Test CSV       : {test_csv}")
	print(f"- Validation CSV : {val_csv}")
	print(f"- Label mapping  : {mapping_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
