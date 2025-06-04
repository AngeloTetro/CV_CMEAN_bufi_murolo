import pandas as pd
import os
import numpy as np
from PIL import Image
import clip
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Fix conflitto OpenMP ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Percorsi ===
df = pd.read_parquet("C:\\Users\\edoar\\Downloads\\artgraph_metadata.parquet")
image_dir = "C:\\Users\\edoar\\Desktop\\images_small"

# === Filtra immagini esistenti e titoli validi ===
available_files = set(os.listdir(image_dir))
df = df[df["FileName"].isin(available_files)].reset_index(drop=True)
df = df[df["ArtworkTitle"].notna()].reset_index(drop=True)

# === Caricamento modello CLIP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === Embedding e similarità ===
image_features_list = []
text_features_list = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processamento"):
    try:
        # Carica immagine
        image_path = os.path.join(image_dir, row["FileName"])
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Prepara testo
        text = clip.tokenize(str(row["ArtworkTitle"])).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)

        image_features_list.append(img_feat.cpu().numpy().flatten())
        text_features_list.append(txt_feat.cpu().numpy().flatten())

    except Exception as e:
        print(f"Errore con {row['FileName']}: {e}")

# === Calcola similarità coseno ===
image_arr = np.vstack(image_features_list)
text_arr = np.vstack(text_features_list)

# Normalizzazione L2
image_arr_norm = image_arr / np.linalg.norm(image_arr, axis=1, keepdims=True)
text_arr_norm = text_arr / np.linalg.norm(text_arr, axis=1, keepdims=True)

# Similarità elemento per elemento
similarities = np.sum(image_arr_norm * text_arr_norm, axis=1)

# === Visualizzazione distribuzione ===
plt.hist(similarities, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribuzione Similarità immagine-testo")
plt.xlabel("Similarità (coseno)")
plt.ylabel("Frequenza")
plt.grid(True)
plt.show()

# === Statistiche ===
print(f"Media: {np.mean(similarities):.3f}")
print(f"Mediana: {np.median(similarities):.3f}")
print(f"Deviazione standard: {np.std(similarities):.3f}")

# === Visualizza esempi migliori e peggiori ===
def show_examples(indices, title):
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        try:
            image_path = os.path.join(image_dir, df.iloc[idx]["FileName"])
            image = Image.open(image_path)

            plt.subplot(1, len(indices), i + 1)
            plt.imshow(image)
            plt.axis("off")
            title_str = df.iloc[idx]['ArtworkTitle']
            sim_score = similarities[idx]
            plt.title(f"{title_str[:25]}...\nSim: {sim_score:.2f}", fontsize=8)
        except Exception as e:
            print(f"Errore nel mostrare {df.iloc[idx]['FileName']}: {e}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Ordina per similarità
sorted_indices = np.argsort(similarities)

# Visualizza peggiori e migliori match
show_examples(sorted_indices[:5], "🔻 Peggiori match immagine-testo")
show_examples(sorted_indices[-5:], "🔺 Migliori match immagine-testo")
