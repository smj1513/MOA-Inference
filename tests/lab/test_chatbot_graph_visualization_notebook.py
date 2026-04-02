from __future__ import annotations

import json
from pathlib import Path


def test_chatbot_graph_visualization_notebook_exists_and_references_app_graph() -> None:
    notebook_path = Path("lab/chatbot_graph_visualization.ipynb")

    assert notebook_path.exists(), "Visualization notebook must exist under lab/"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            sources.append("".join(source))
        else:
            sources.append(str(source))
    notebook_text = "\n".join(sources)

    assert "build_chatbot_graph" in notebook_text
    assert "ChatbotToolBundle.empty" in notebook_text
    assert "draw_mermaid" in notebook_text
    assert "draw_mermaid_png" in notebook_text
    assert "get_graph(xray=True)" in notebook_text
    assert "sys.path" in notebook_text
    assert "single-agent" in notebook_text.casefold()
    assert "sql team" not in notebook_text.casefold()
    assert "coaching team" not in notebook_text.casefold()
