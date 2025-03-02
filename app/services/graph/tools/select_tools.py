from pydantic import BaseModel, Field


class ToolOptions(BaseModel):
    """Optional tools that can be used by the user

    Attributes:
    -----------

    use_hand_to_draw (bool) : Detect if the user wants to draw the image, set to True
    use_google_to_search (bool) : Detect if the user wants to search on Google, set to True
    use_local_file (bool) : If the user provides a local file, set to True
    no_action (bool) : Detect if the user wants to take no action, set to True
    """

    use_hand_to_draw: bool = Field(
        description="If the user wants to draw the image, set to True to draw the image",
        default=False,
    )
    use_google_to_search: bool = Field(
        description="If you don't know the answer and want to search on Google, set to True to search on Google",
        default=False,
    )
    use_local_file: bool = Field(
        description="If user provides a local file, set to True to use the local file",
        default=False,
    )
    no_action: bool = Field(
        description="If you know the answer, you don't need use_google_to_search set to True to take no action",
        default=False,
    )
