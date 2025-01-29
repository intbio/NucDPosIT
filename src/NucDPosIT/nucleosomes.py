from dataclasses import dataclass, field


@dataclass
class Nucleosome:
    dyad: int
    start: int
    end: int
    id: str
    tlen: int = field(init=False)
    center: int = field(init=False)

    def __post_init__(self):
        self.start = self.dyad - 73 - self.start
        self.end = self.dyad + 73 + self.end
        self.center = (self.start + self.end) / 2
        self.tlen = self.end - self.start