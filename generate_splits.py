import numpy as np
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd
from numpy.array_api import int64


IMAGE_DIR_PATH = r"\\192.168.155.240\Robotica\dataset_iqa\HRIQ\512x384"
OUTPUT_ROOT_DIR = "datasets/HRIQ/splits"
NUM_SPLITS_TO_GENERATE = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
METADATA_FILE_PATH = None
DIST_IMG_COLUMN = None

def pad_lists(list_of_lists, pad_value=-1):
    """Aplica padding a uma lista de listas para que todas tenham o mesmo comprimento."""
    max_len = max(len(sublist) for sublist in list_of_lists)
    padded_lists = []
    for sublist in list_of_lists:
        padded_list = sublist + [pad_value] * (max_len - len(sublist))
        padded_lists.append(padded_list)
    return padded_lists


def generate_final_splits_indices(metadata_file, image_dir, output_root, num_splits, train_ratio, val_ratio, dist_col,
                                  ref_col):
    """
    Gera 3 arquivos .npy (train, val, test) contendo os ÍNDICES das imagens para 10 splits.
    """
    all_distorted_images = []

    if metadata_file:
        print("Modo: Dataset Sintético (baseado em imagens de referência)")
        df = pd.read_csv(metadata_file)

        # Cria a lista mestre de todas as imagens na ordem do arquivo
        all_distorted_images = df[dist_col].tolist()

        ref_to_dist_map = defaultdict(list)
        for _, row in df.iterrows():
            ref_to_dist_map[row[ref_col]].append(row[dist_col])

        base_items = list(ref_to_dist_map.keys())
        print(f"Encontradas {len(base_items)} imagens de referência únicas.")

    elif image_dir:
        print("Modo: Dataset Autêntico (baseado em imagens individuais)")
        image_dir = Path(image_dir)
        # A ordem dos arquivos é importante, então vamos ordená-los para consistência
        base_items = sorted([f.name for f in image_dir.glob('*.*')])
        all_distorted_images = base_items  # Neste caso, a lista base é a lista final
        print(f"Encontradas {len(base_items)} imagens únicas.")
    else:
        raise ValueError("Forneça ou METADATA_FILE_PATH ou IMAGE_DIR_PATH.")

    # --- NOVO: Cria um mapa de nome de arquivo para seu índice original ---
    filename_to_idx = {filename: i for i, filename in enumerate(all_distorted_images)}

    num_items = len(base_items)
    train_count = int(num_items * train_ratio)
    val_count = int(num_items * val_ratio)
    print(
        f"Divisão base: {train_count} (Treino), {val_count} (Validação), {num_items - train_count - val_count} (Teste)")

    all_train_splits = []
    all_val_splits = []
    all_test_splits = []

    for i in range(num_splits):
        print(f"--- Gerando dados para o Split Index {i} ---")
        random.seed(i)
        random.shuffle(base_items)

        train_base = base_items[:train_count]
        val_base = base_items[train_count: train_count + val_count]
        test_base = base_items[train_count + val_count:]

        if metadata_file:
            train_files = [dist for ref in train_base for dist in ref_to_dist_map[ref]]
            val_files = [dist for ref in val_base for dist in ref_to_dist_map[ref]]
            test_files = [dist for ref in test_base for dist in ref_to_dist_map[ref]]
        else:  # Datasets autênticos
            train_files, val_files, test_files = train_base, val_base, test_base

        # --- MODIFICAÇÃO PRINCIPAL: Converte nomes de arquivos para índices ---
        train_indices = [filename_to_idx[fname] for fname in train_files]
        val_indices = [filename_to_idx[fname] for fname in val_files]
        test_indices = [filename_to_idx[fname] for fname in test_files]

        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        all_train_splits.append(train_indices)
        all_val_splits.append(val_indices)
        all_test_splits.append(test_indices)

    padded_train_splits = pad_lists(all_train_splits)
    padded_val_splits = pad_lists(all_val_splits)
    padded_test_splits = pad_lists(all_test_splits)

    output_splits_dir = Path(output_root)
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_splits_dir / "train.npy", np.array(padded_train_splits, dtype=int64))
    np.save(output_splits_dir / "val.npy", np.array(padded_val_splits, dtype=int64))
    np.save(output_splits_dir / "test.npy", np.array(padded_test_splits, dtype=int64))

    print(f"\nConcluído! 3 arquivos de split com ÍNDICES foram salvos em: {output_splits_dir}")


# Bloco principal
if __name__ == '__main__':
    generate_final_splits_indices(
        metadata_file=METADATA_FILE_PATH,
        image_dir=IMAGE_DIR_PATH,
        output_root=OUTPUT_ROOT_DIR,
        num_splits=NUM_SPLITS_TO_GENERATE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        dist_col=DIST_IMG_COLUMN,
        ref_col=None
    )