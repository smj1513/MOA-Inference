from typing import TypedDict


class ImageGenerationState(TypedDict, total=False):
    user_prompt: str
    input_image_base64: str
    input_mime_type: str
    final_prompt: str
    generated_image_base64: str
    output_mime_type: str
    model: str
