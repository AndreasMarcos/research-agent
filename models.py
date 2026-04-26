from dataclasses import dataclass, field


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    answer: str
    steps: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)