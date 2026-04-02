from collections.abc import Awaitable, Callable

from app.core.config import Settings
from app.domain.image_generation.agents.states.state import ImageGenerationState

ImageGenerator = Callable[..., Awaitable[dict[str, str]]]


def build_generate_image_node(
    *,
    settings: Settings,
    image_generator: ImageGenerator,
):
    async def generate_image_node(state: ImageGenerationState) -> ImageGenerationState:
        image_result = await image_generator(
            prompt=state["final_prompt"],
            model=settings.image_generation_model,
            image_base64=state["input_image_base64"],
            mime_type=state["input_mime_type"],
        )
        return {
            "generated_image_base64": image_result["generated_image_base64"],
            "output_mime_type": image_result["mime_type"],
            "model": image_result["model"],
        }

    return generate_image_node
