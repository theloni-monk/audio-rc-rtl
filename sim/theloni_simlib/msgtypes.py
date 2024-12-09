from dataclasses import dataclass
from enum import Enum
import numpy as np

class ContentType(str, Enum):
    SINGLE = "single"
    BURST = "burst"

# TODO: custom scoreboard to quiet classinstance errors

class AXISMessage:
    def __init__(self, data, msg_name = "_untitled_msg_", msg_type=ContentType.SINGLE,  msg_id = -1, last=0, strb = 0):
        self.data = data
        self.msg_name = msg_name
        self.msg_type = msg_type
        self.msg_id = msg_id
        self.last = last
        self.strb = strb

    def __repr__(self):
        return f"""MSG {self.msg_id} : {self.msg_name} => {self.data} ; strb {self.strb}; last {self.last}"""

    def __eq___(self, other):
        if isinstance(other, int):
            if len(self.data) > 1:
                return False
            return int(self.data[0]) == other
        if isinstance(other, list):
            return all(a == b for a, b in zip(self.data, other))
        if not isinstance(other, self.__class__):
            return all(a == b for a, b in zip(self.data, other.data))

    @classmethod
    def nullsingle(cls):
        return cls(msg_type=ContentType.SINGLE, data = [0])

    @classmethod
    def nullburst(cls):
        return cls(msg_type=ContentType.BURST, data = [0 for _ in range(256)])

    @classmethod
    def randmsg(cls):
        return cls(msg_type=ContentType.SINGLE, data = [random.randint(-10000, 10000)])

    @classmethod
    def randburst(cls):
        return cls(msg_type=ContentType.BURST, data = list(random.randint(-10000, 10000) for _ in range(256)))

@dataclass
class BspkVectorMsg:
    data: np.array
    tid: str
