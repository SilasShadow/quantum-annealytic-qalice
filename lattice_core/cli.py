import typer

app = typer.Typer(help="Lattice — baseline CLI for the segmentation pipeline.")


@app.command()
def hello(name: str = "world"):
    """Smoke test command."""
    typer.echo(f"Hello, {name}! This is Lattice.")


if __name__ == "__main__":
    app()
