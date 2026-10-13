def main() -> None:
    """Entry point for the `ragstream` command."""
    # Imported here so that `import ragstream` stays light
    from ragstream.cli import main as cli_main
    cli_main()
