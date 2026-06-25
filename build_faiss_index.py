import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CSV_FILE = "main_df.csv"
FAISS_DIR = "quran_faiss_index"


def load_quran_data(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    required_columns = ["Surah", "Name", "Ayat", "Translation1", "Tafaseer1"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["text"] = df.apply(
        lambda row: (
            f"Surah {row['Surah']} ({row['Name']}), "
            f"Ayah {row['Ayat']}: {row['Translation1']} "
            f"Tafsir: {row['Tafaseer1']}"
        ),
        axis=1
    )

    return df


def build_and_save_faiss_index():
    print("Step 1: Loading CSV...")
    df = load_quran_data(CSV_FILE)
    print(f"Loaded {len(df)} rows")

    print("Step 2: Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Step 3: Creating FAISS index...")
    vector_store = FAISS.from_texts(
        texts=df["text"].tolist(),
        embedding=embedding_model
    )

    print(f"Step 4: Saving FAISS index to '{FAISS_DIR}'...")
    vector_store.save_local(FAISS_DIR)

    print("Done.")
    print(f"Saved in folder: {FAISS_DIR}")


if __name__ == "__main__":
    print("build_faiss_index.py started...")
    build_and_save_faiss_index()