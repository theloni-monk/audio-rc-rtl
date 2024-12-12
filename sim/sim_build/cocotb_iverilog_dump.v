module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim_build/demo_dummy_tl.fst");
    $dumpvars(0, demo_dummy_tl);
end
endmodule
