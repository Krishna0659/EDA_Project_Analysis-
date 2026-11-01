"""Execute the portfolio notebook from a fresh kernel, failing on any cell error."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "notebooks/credit_profile_analysis.ipynb")
    args = parser.parse_args()
    notebook = nbformat.read(root / "notebooks/credit_profile_analysis.ipynb", as_version=4)
    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {args.output}")


if __name__ == "__main__":
    main()
