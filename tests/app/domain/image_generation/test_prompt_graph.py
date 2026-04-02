import anyio

from app.core.config import Settings
from app.domain.image_generation.agents.graph import build_image_generation_graph
from app.domain.image_generation.agents.prompts.image_generation_prompt import (
    build_image_generation_prompt,
)


def test_build_image_generation_prompt_expands_user_edit_goal() -> None:
    prompt = build_image_generation_prompt(
        "Turn the reference image into a polished cyberpunk banana poster."
    )

    assert "cyberpunk banana poster" in prompt
    assert "reference image" in prompt
    assert "512x512" in prompt
    assert "single final image" in prompt
    assert "- grids" in prompt
    assert "speech bubbles" in prompt
    assert "UI overlays" in prompt
    assert prompt != "Turn the reference image into a polished cyberpunk banana poster."


def test_build_image_generation_prompt_keeps_structured_input_as_raw_goal_request() -> None:
    prompt = build_image_generation_prompt(
        "[Title] 노트북 구매 [description] 삼성 노트북 신상 구매"
    )

    assert "Goal request: [Title] 노트북 구매 [description] 삼성 노트북 신상 구매" in prompt
    assert "Title metadata:" not in prompt


def test_image_generation_graph_builds_prompt_and_delegates_generation() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    upstream_calls: list[dict[str, str]] = []

    async def fake_image_generator(
        *,
        prompt: str,
        model: str,
        image_base64: str,
        mime_type: str,
    ) -> dict[str, str]:
        upstream_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "image_base64": image_base64,
                "mime_type": mime_type,
            }
        )
        return {
            "generated_image_base64": "base64-image",
            "mime_type": "image/png",
            "model": model,
        }

    graph = build_image_generation_graph(
        settings=settings,
        image_generator=fake_image_generator,
    )

    async def run_graph() -> dict[str, str]:
        return await graph.ainvoke(
            {
                "user_prompt": "Turn the reference image into a polished cyberpunk banana poster.",
                "input_image_base64": "reference-image",
                "input_mime_type": "image/jpeg",
            }
        )

    result = anyio.run(run_graph)

    assert result["generated_image_base64"] == "base64-image"
    assert result["output_mime_type"] == "image/png"
    assert result["model"] == "gemini-2.5-flash-image"
    assert "cyberpunk banana poster" in result["final_prompt"]
    assert upstream_calls == [
        {
            "prompt": result["final_prompt"],
            "model": "gemini-2.5-flash-image",
            "image_base64": "reference-image",
            "mime_type": "image/jpeg",
        }
    ]
