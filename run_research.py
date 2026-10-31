"""Execute a fresh research notebook, excluding its manual loading alternatives."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='Write the fresh-run notebook without executing it.')
    parser.add_argument('--include-listening-tests', action='store_true', help='Also generate the four listening-test banks after model evaluation.')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    notebook = nbformat.read(project / 'FeatureSelectionforClustering.ipynb', as_version=4)
    # Exclude loading alternatives, the optional benchmark, listening-test design, and the historical summary.
    notebook.cells = [cell for i, cell in enumerate(notebook.cells)
                      if i not in (8, 49, 60) and i < (90 if args.include_listening_tests else 83)]
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = project / 'artifacts' / 'executed_notebooks' / f'research_{stamp}.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    print(f'Notebook: {destination}', flush=True)
    if args.prepare_only:
        return

    def report_output(message, **kwargs):
        content = message.get('content', {})
        if message.get('msg_type') == 'stream':
            print(content.get('text', ''), end='', flush=True)

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

    client = NotebookClient(
        notebook, timeout=None, kernel_name='python3',
        resources={'metadata': {'path': str(project)}},
        on_cell_start=report_cell, on_cell_executed=lambda **kwargs: nbformat.write(notebook, destination),
    )
    # Stream progress to PowerShell while retaining all rich output in the notebook.
    original_process_message = client.process_message

    def process_message(msg, cell, cell_index):
        report_output(msg)
        return original_process_message(msg, cell, cell_index)

    client.process_message = process_message
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
        print(f'Execution notebook saved: {destination}', flush=True)


if __name__ == '__main__':
    main()
