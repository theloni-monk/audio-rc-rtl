from cocotb_bus.scoreboard import Scoreboard
from .msgtypes import *


class EpsScoreboard(Scoreboard):
    def __init__(self, *args, epsilon=0.01, debug = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.eps = epsilon

    def compare(self, got, exp, log, strict_type=True):
        if isinstance(got, AXISMessage):
            got = got.data[0]
        if isinstance(got, BspkVectorMsg):
            got = got.data

        err = float('inf')
        tol = self.eps
        if isinstance(got, np.ndarray):
            err = np.sum(np.abs(got-exp))
            tol = self.eps * len(got)
        else:
            raise NotImplementedError
        if err <= tol:
            if self.debug:
                try:
                    log.info(f"SCOREBOARD Received expected transaction {len(got)} bytes : {got}")
                except Exception:
                    pass
        else:
            self.errors += 1
            # Try our best to print out something useful
            if self.debug:
                strgot, strexp = str(got), str(exp)
                log.error(f"Received transaction {got} differed from expected output {strexp}")
                if self._imm:
                    assert False, (
                    "Received transaction differed from expected "
                    "transaction"
                    )

class DummyScorebaord(Scoreboard):
    def __init__(self, dut, reorder_depth=0, fail_immediately=True):
        super().__init__(dut, reorder_depth, fail_immediately)

    def compare(self, got, exp, log, strict_type=True):
        pass