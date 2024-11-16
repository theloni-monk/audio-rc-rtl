from cocotb_bus.scoreboard import Scoreboard

class EpsScoreboard(Scoreboard):
    def __init__(self, *args, epsilon=1, debug = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.eps = epsilon

    def compare(self, got, exp, log, strict_type=True):
        if abs(got.data[0]-exp) <= self.eps: #change to whatever you want for the problem at hand.
        # Don't want to fail the test
        # if we're passed something without __len__
            if self.debug:
                try:
                    log.debug("Received expected transaction %d bytes" %
                            (len(got.data)))
                    log.debug(repr(got))
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