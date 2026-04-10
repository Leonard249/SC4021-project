import json
import logging
from pathlib import Path
from tqdm import tqdm

# Adjust these imports if your folder structure is slightly different.
# This assumes test_classifier.py is run from the parent directory of 'pragmatics/'
from pragmatics.aspect_extractor import AspectExtractor
from pragmatics.sarcasm_detector import SarcasmDetector
from pragmatics.ensemble import PolarityEnsemble

# Suppress overly verbose logs from the individual modules so tqdm displays cleanly
logging.getLogger("pragmatics.aspect_extractor").setLevel(logging.WARNING)
logging.getLogger("pragmatics.sarcasm_detector").setLevel(logging.WARNING)
logging.getLogger("pragmatics.ensemble").setLevel(logging.WARNING)

def load_json(path: Path) -> list[dict]:
    """Load the JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def save_json(data: list[dict], path: Path) -> None:
    """Save the output JSON dataset, creating directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Pipeline complete. Results saved to: {path}")

def run_pragmatics_pipeline(input_file: str, output_file: str, s7_chunk_size: int = 128):
    """
    Runs Stages 6, 7, and 8 of the SC4021 NLP pipeline with tqdm progress tracking.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    print(f"Loading Stage 5 data from {input_path}...")
    records = load_json(input_path)
    total_records = len(records)
    print(f"Loaded {total_records} records.\n")

    # ---------------------------------------------------------
    # STAGE 6: Aspect Extraction
    # ---------------------------------------------------------
    print("Initializing Stage 6 (AspectExtractor)...")
    extractor = AspectExtractor()
    
    # Process record-by-record to allow a smooth 1-by-1 tqdm progress bar
    for record in tqdm(records, desc="Stage 6: Aspect Extraction", unit="rec"):
        extractor.extract_record(record)

    # ---------------------------------------------------------
    # STAGE 7: Sarcasm Detection
    # ---------------------------------------------------------
    print("\nInitializing Stage 7 (SarcasmDetector)...")
    # SarcasmDetector needs batching for Transformer efficiency. 
    # We chunk the records to get both batched inference AND a tqdm progress bar.
    detector = SarcasmDetector()
    
    # Calculate chunks
    chunks = [records[i:i + s7_chunk_size] for i in range(0, total_records, s7_chunk_size)]
    
    for chunk in tqdm(chunks, desc="Stage 7: Sarcasm Detection", unit="batch"):
        # detect_corpus handles the in-place modification and batching optimally
        detector.detect_corpus(chunk)

    # ---------------------------------------------------------
    # STAGE 8: Polarity Ensemble (Length-Aware Routing)
    # ---------------------------------------------------------
    print("\nInitializing Stage 8 (PolarityEnsemble)...")
    ensemble = PolarityEnsemble()

    for record in tqdm(records, desc="Stage 8: Polarity Ensemble", unit="rec"):
        ensemble.classify_record(record)

    # ---------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------
    save_json(records, output_path)

if __name__ == "__main__":
    # Define your input (Stage 5 output) and final output paths
    INPUT_FILE = "../../data/my_test/pipeline_output.json" # Change this to your actual input file
    OUTPUT_FILE = "../../data/results/draft_output.json"
    
    # Note: s7_chunk_size determines how many records are sent to the Transformer at once. 
    # Adjust based on your available RAM/VRAM. 128 is a good default for text records.
    run_pragmatics_pipeline(
        input_file=INPUT_FILE, 
        output_file=OUTPUT_FILE,
        s7_chunk_size=128 
    )