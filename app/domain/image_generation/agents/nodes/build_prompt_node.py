from app.domain.image_generation.agents.prompts.image_generation_prompt import (
    build_image_generation_prompt,
)
from app.domain.image_generation.agents.states.state import ImageGenerationState


def build_prompt_node(state: ImageGenerationState) -> ImageGenerationState:
    return {
        "final_prompt": build_image_generation_prompt(state["user_prompt"]),
    }
