from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.domain.image_generation.agents.nodes.build_prompt_node import build_prompt_node
from app.domain.image_generation.agents.nodes.generate_image_node import (
    ImageGenerator,
    build_generate_image_node,
)
from app.domain.image_generation.agents.states.state import ImageGenerationState


def build_image_generation_graph(
    *,
    settings: Settings,
    image_generator: ImageGenerator,
):
    graph = StateGraph(ImageGenerationState)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node(
        "generate_image",
        build_generate_image_node(
            settings=settings,
            image_generator=image_generator,
        ),
    )
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "generate_image")
    graph.add_edge("generate_image", END)
    return graph.compile()
