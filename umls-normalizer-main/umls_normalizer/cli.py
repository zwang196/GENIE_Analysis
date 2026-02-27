import argparse
from pathlib import Path

from .normalizer import normalize_file, normalize_folder
from .retriever import UMLSIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch normalize entity terms to UMLS (CUI + preferred term)."
    )
    parser.add_argument("--input-dir", help="Folder with CSV files to normalize.")
    parser.add_argument("--output-dir", required=True, help="Folder to write enriched CSV files.")
    parser.add_argument("--entity-column", required=True, help="Column name that contains raw entity strings.")
    parser.add_argument("--dictionary", required=True, help="Path to UMLS dictionary (CUI||term||semantic_type).")
    parser.add_argument("--model-name", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", help="Encoder model name or path.")
    parser.add_argument("--cache-dir", default="./cache", help="Where to cache dictionary embeddings.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for encoding queries and dictionary.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id when using a single GPU FAISS index.")
    parser.add_argument("--use-all-gpus", action="store_true", help="Spread FAISS index across all visible GPUs.")
    parser.add_argument("--no-faiss-gpu", action="store_true", help="Force FAISS to run on CPU even if CUDA is available.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for encoder forward pass (CUDA only).")
    parser.add_argument("--top-k", type=int, default=1, help="How many normalized terms to keep per mention.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold; below this candidates are filtered, but top-1 is kept with warning.")
    parser.add_argument("--dict-batch-size", type=int, default=256, help="Batch size for encoding the dictionary.")
    parser.add_argument("--query-batch-size", type=int, default=256, help="Batch size for encoding query terms.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input files when using folder mode.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose normalized output already exists.")
    parser.add_argument("--use-ivf", action="store_true", help="Enable IVF Flat index for large dictionaries.")
    parser.add_argument("--ivf-threshold", type=int, default=50_000, help="Minimum vocab size to trigger IVF if enabled.")
    parser.add_argument("--mode", choices=["folder", "file"], default="folder", help="Process a folder or a single file.")
    parser.add_argument("--input-file", help="Single CSV file to normalize when mode=file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING).")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force transformers to load SapBERT from local files only (no network calls).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "folder" and not args.input_dir:
        raise ValueError("--input-dir is required in folder mode")
    if args.mode == "file" and not args.input_file:
        raise ValueError("--input-file is required in file mode")

    use_faiss_gpu = not args.no_faiss_gpu
    index = UMLSIndex(
        model_name=args.model_name,
        dictionary_path=args.dictionary,
        cache_dir=args.cache_dir,
        device=args.device,
        fp16=args.fp16,
        use_faiss_gpu=use_faiss_gpu,
        use_all_gpus=args.use_all_gpus,
        gpu_id=args.gpu_id,
        use_ivf=args.use_ivf,
        ivf_threshold=args.ivf_threshold,
        local_files_only=args.offline,
    )
    index.logger.setLevel(args.log_level)
    index.load(batch_size=args.dict_batch_size)

    if args.mode == "file":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / (Path(args.input_file).stem + ".normalized.csv")
        normalize_file(
            input_path=args.input_file,
            output_path=str(outfile),
            retriever=index,
            entity_column=args.entity_column,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.query_batch_size,
        )
    else:
        normalize_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            retriever=index,
            entity_column=args.entity_column,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.query_batch_size,
            pattern=args.pattern,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
