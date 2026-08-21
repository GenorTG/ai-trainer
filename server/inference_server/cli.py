"""CLI entry point for the inference server.

WHAT THIS FILE DOES
==================
This is the script that runs when you type `inference-server` in your terminal.
It uses Typer (a modern CLI library) to parse command-line arguments and
launch the server with the right configuration.

KEY CONCEPTS
============
- Typer: a Python CLI library built on Click. Lets you define commands
  as functions with type annotations.
- Config file: a YAML file with server settings (port, model path, etc.)
- Hot reload: restarting the server when code changes (for development)
- Background mode: running the server as a daemon (not blocking the terminal)
"""

"""CLI for portable inference server."""
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="inference-server", description="Portable Inference Server with RAG"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--host", help="Override host")
    parser.add_argument("--port", type=int, help="Override port")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--ingest", help="Ingest documents from directory")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")

    args = parser.parse_args()

    # Load config
    from .config import load_config

    config = load_config(args.config)

    # Apply overrides
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.model:
        config.model.path = args.model
    if args.no_rag:
        config.rag.enabled = False

    # Ingest mode
    if args.ingest:
        from .rag import DocumentIngestor, RAGStore

        store = RAGStore(config.rag.store_path)
        ingestor = DocumentIngestor(store, config.rag.chunk_size, config.rag.chunk_overlap)
        result = ingestor.ingest_directory(args.ingest, embedding_model=config.rag.embedding_model)
        print(f"Ingested {result['files_ingested']} files, {result['chunks_added']} chunks")
        if result["errors"]:
            for e in result["errors"]:
                print(f"  Error: {e['file']}: {e['error']}")
        return

    # Start server
    import uvicorn

    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


if __name__ == "__main__":
    main()
