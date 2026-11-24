"""Stand-in so scb imports without the openai package; the analysis never calls a model."""


class OpenAI:  # noqa: D101
    pass
