import cocotb
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, ReadOnly,with_timeout, First, Join

async def generate_clock(clock_wire):
	while True: # repeat forever
		clock_wire.value = 0
		await Timer(5,units="ns")
		clock_wire.value = 1
		await Timer(5,units="ns")

async def reset(wire, clk, active = 0):
    wire.value = active
    await ClockCycles(clk, 5)
    wire.value = 1-active
