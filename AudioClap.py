import os
import pandas as pd
import librosa # Mantenuto per compatibilità, anche se msclap carica direttamente
import torch
import numpy as np # Mantenuto per final numpy operations (e.g., statistics)
from sklearn.metrics.pairwise import cosine_similarity # Mantenuto, ma useremo clap_model.compute_similarity
from msclap import CLAP
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm # Per la barra di progresso

# --- 1. Initial Configuration ---
# Determine the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to necessary files and folders.
# *** IMPORTANT: MAKE SURE THESE PATHS ARE CORRECT ON YOUR SYSTEM! ***

PATHS = {
    "metadata": r"C:\Users\edoar\Desktop\computer vision\proj\tracks.csv",
    "audio_folder": r"C:\Users\edoar\Desktop\fma_small", # <-- CORRECT PATH
    "clap_weights": r"C:\Users\edoar\Downloads\CLAP_weights_2023.pth"
}

# --- 2. Load CLAP Model ---
# Attempt to load the CLAP model. If an error occurs, the program exits.
try:
    clap_model = CLAP(
        version='2023',
        model_fp=PATHS["clap_weights"],
        use_cuda=True if device == "cuda" else False
    )
    print("✅ CLAP model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading CLAP model: {e}")
    print("Make sure the CLAP weights path is correct and the file exists.")
    sys.exit(1) # Exit the script in case of a critical error

# --- 3. Function to Get Audio Path ---
# This function constructs the full path of an audio file
# based on the track ID and the FMA folder structure.
def get_audio_path(track_id):
    try:
        # Convert the track ID to an integer and format it with leading zeros
        tid = int(track_id)
        # Construct the path: audio_folder/XXX/XXXXXX.mp3
        return os.path.join(
            PATHS["audio_folder"],
            f"{tid:06d}"[:3],  # Take the first 3 digits of the formatted ID
            f"{tid:06d}.mp3"   # The full formatted ID with .mp3 extension
        )
    except ValueError:
        # Handle cases where track_id is not convertible to an integer
        return None
    except Exception as e:
        # Handle other unexpected errors during path construction
        return None

# --- 4. Load and Filter Metadata ---
# Load the CSV metadata file and filter only tracks from the 'small' subset.
try:
    metadata = pd.read_csv(PATHS["metadata"], index_col=0, header=[0, 1])
    metadata = metadata[metadata[('set', 'subset')] == 'small']
    print(f"✅ Metadata loaded and filtered: {len(metadata)} tracks in the 'small' subset.")
except FileNotFoundError:
    print(f"❌ Error: Metadata file not found at path: {PATHS['metadata']}")
    print("Make sure the path to the tracks.csv file is correct.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading or filtering metadata: {e}")
    sys.exit(1)

# --- 5. Search for Valid Audio Tracks and Prepare Data ---
# Search for valid audio tracks and prepare textual descriptions (genre).
# We want to process enough tracks to have meaningful statistics.
# We can limit the number of tracks processed to avoid excessive execution times.
max_tracks_to_check = 8000 # Maximum number of tracks from metadata to check
max_tracks_to_process = 8000 # Number of valid tracks to process for analysis

valid_tracks_info = [] # List of dictionaries { 'tid': ..., 'audio_path': ..., 'text_desc': ... }
print("\nSearching and preparing audio tracks and descriptions:")

# Iterate through metadata to find existing tracks and prepare textual descriptions
for i, tid in enumerate(tqdm(metadata.index, desc="Data Preparation")):
    # Limit the search to a certain number of IDs from metadata to avoid excessive times
    if i >= max_tracks_to_check and len(valid_tracks_info) < max_tracks_to_process:
        print(f"Search limit ({max_tracks_to_check} tracks from metadata) reached without finding enough valid tracks. Continuing search...")
        pass
    
    # Stop preparation if we have already found enough tracks to process
    if len(valid_tracks_info) >= max_tracks_to_process:
        print(f"Limit of {max_tracks_to_process} tracks to process reached. Stopping preparation.")
        break

    audio_path = get_audio_path(tid)
    if audio_path:
        # Check for existence of the audio file
        if os.path.exists(audio_path):
            # Get the top genre as a textual description
            genre = metadata.loc[tid, ('track', 'genre_top')]
            if pd.isna(genre):
                genre_text = "unknown genre" # Placeholder if genre is not available
            else:
                genre_text = str(genre)

            valid_tracks_info.append({
                'tid': tid,
                'audio_path': audio_path,
                'text_desc': genre_text
            })
        # else:
            # print(f"  Skipped track {tid}: File '{os.path.basename(audio_path)}' not found in '{os.path.dirname(audio_path)}'.")
    # else:
        # print(f"  Skipped track {tid}: Unable to generate a valid path.")

# Check if enough valid tracks were found for analysis
if len(valid_tracks_info) < 2:
    print("\n❌ Error: Insufficient valid audio tracks for comparison. Make sure the dataset is complete and paths are correct.")
    sys.exit(1)

print(f"✅ Prepared {len(valid_tracks_info)} tracks for analysis.")

# === 6. Extract Audio and Text Embeddings ===
audio_embeddings_list = []
text_embeddings_list = []
processed_tracks_info = [] # To keep track only of successfully processed tracks

print("\nExtracting audio and text embeddings:")
for info in tqdm(valid_tracks_info, desc="Embedding Extraction"):
    tid = info['tid']
    audio_file_path = info['audio_path']
    text_desc = info['text_desc']

    try:
        # Extract audio embedding from the file path
        audio_emb = clap_model.get_audio_embeddings([audio_file_path], resample=True)

        # Extract text embedding from the genre description
        text_emb = clap_model.get_text_embeddings([text_desc])

        # If both extractions are successful, add to lists
        audio_embeddings_list.append(audio_emb)
        text_embeddings_list.append(text_emb)
        processed_tracks_info.append(info) # Add only if extraction is successful

    except Exception as e:
        # Print the error and continue with the next track
        print(f"❌ Error during extraction for track {tid}: {e}")
        print(f"Error details: {e}")
        # Do not terminate the script, but skip this problematic track

# Convert lists of embeddings into a single PyTorch tensor
# Ensure there are at least two valid embeddings to proceed
if not audio_embeddings_list or not text_embeddings_list or len(audio_embeddings_list) < 2:
    print("\n❌ Error: No embeddings (or fewer than 2) were successfully extracted. Cannot compute similarity.")
    sys.exit(1)

# squeeze(0) removes the single dimension (batch size of 1) to allow vstack to concatenate correctly
audio_embeddings_arr = torch.vstack([emb.squeeze(0) for emb in audio_embeddings_list])
text_embeddings_arr = torch.vstack([emb.squeeze(0) for emb in text_embeddings_list])

print(f"✅ Extracted {len(audio_embeddings_arr)} audio embeddings and {len(text_embeddings_arr)} text embeddings.")


# === 7. Calcolo della Similarità Cosina (Pura) ===
# Normalizza gli embedding per ottenere la similarità coseno standard tra -1 e 1
audio_embeddings_norm = torch.nn.functional.normalize(audio_embeddings_arr, p=2, dim=-1)
text_embeddings_norm = torch.nn.functional.normalize(text_embeddings_arr, p=2, dim=-1)

# Calcola il prodotto scalare (dot product) tra gli embedding normalizzati
# Questo è equivalente alla similarità coseno
# @ è l'operatore di moltiplicazione di matrici in PyTorch (come np.dot)
# .sum(dim=-1) per sommare lungo l'ultima dimensione (la dimensione dell'embedding)
similarities = (audio_embeddings_norm * text_embeddings_norm).sum(dim=-1)

# Converti i risultati in NumPy per le operazioni statistiche e plotting
similarities = similarities.cpu().detach().numpy()

print(f"✅ Calcolate {len(similarities)} similarità audio-testo (coseno puro).")

# === 8. Visualizzazione Distribuzione Similarità ===
plt.figure(figsize=(10, 6))
plt.hist(similarities, bins=30, color='lightcoral', edgecolor='black')
plt.title("Distribuzione Similarità Audio-Testo (Genere - Coseno Puro)")
plt.xlabel("Similarità (coseno)")
plt.ylabel("Frequenza")
plt.grid(True)
plt.show()

# === 9. Statistiche ===
print("\n=== Statistiche Similarità Audio-Testo (Coseno Puro) ===")
print(f"Media: {np.mean(similarities):.3f}")
print(f"Mediana: {np.median(similarities):.3f}")
print(f"Deviazione standard: {np.std(similarities):.3f}")

# === 10. Visualizza Esempi Migliori e Peggiori ===
def print_examples(indices, title, data_info, similarities_scores, num_examples=5):
    """
    Stampa i dettagli degli esempi con le similarità più alte o più basse.
    data_info deve essere la lista `processed_tracks_info` che contiene solo le tracce valide.
    """
    print(f"\n=== {title} ===")
    for i, idx in enumerate(indices):
        if i >= num_examples:
            break
        # Usa data_info (che è processed_tracks_info) perché i suoi indici corrispondono a similarities_scores
        track_id = data_info[idx]['tid']
        text_desc = data_info[idx]['text_desc']
        sim_score = similarities_scores[idx]
        print(f"Track ID: {track_id}, Genere: '{text_desc}', Similarità: {sim_score:.4f}")

# Ordina gli indici per similarità (dal più basso al più alto)
sorted_indices = np.argsort(similarities)

# Visualizza i 5 peggiori match (similarità più basse)
print_examples(sorted_indices[:5], "🔻 Peggiori match Audio-Testo (Genere - Coseno Puro)", processed_tracks_info, similarities)

# Visualizza i 5 migliori match (similarità più alte)
# Usiamo [::-1] per invertire l'ordine e avere i migliori esempi in cima alla lista stampata
print_examples(sorted_indices[-5:][::-1], "🔺 Migliori match Audio-Testo (Genere - Coseno Puro)", processed_tracks_info, similarities)
