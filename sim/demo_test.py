
import os
import cocotb
from bespoketb.bespoke import *
from bespoketb.boards import EpsScoreboard
from bespoketb.ctb_util import *

class DemoTester(BspkModuleTester):
    def __init__(self, dut, modelpath, in_dim, debug, **kwargs):
        board = EpsScoreboard(0.01, debug, dut)
        super().__init__(dut, modelpath, in_dim, board, False, debug, **kwargs)

    def _gen_indata(self):
        return 0.2*np.random.randn(self.in_dim).astype(dtype=np.float32)
CYCLES = 10
@cocotb.test()
async def demo_test(dut):
    #cocotb test for vwbmacc
    tlen = int(os.getenv("TLEN"))
    mdl_dim = int(os.getenv("MDIM"))
    mdl_path = os.getenv("MPATH")
    debug = os.getenv("DEBUG") == "True"
    if debug:
        print("DEBUGGING", os.environ)
    test = DemoTester(dut, mdl_path, mdl_dim, debug)
    await test.startup()

    await test.run_io_test(tlen)

    await ClockCycles(dut.clk_in, tlen * CYCLES)
    dut._log.info(
        f"Processed {test.inm.stats.received_transactions} transactions in and {test.outm.stats.received_transactions} out"
    )
    assert (
        test.inm.stats.received_transactions == test.outm.stats.received_transactions
    ), f"Transaction Count doesn't match! :/"
    dut._log.info(
        f"maximum float err between recieved and expected: {test.scoreboard.maxerr}"
    )
