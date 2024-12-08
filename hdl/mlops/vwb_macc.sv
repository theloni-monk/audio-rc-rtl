`timescale 1ns / 1ps
`default_nettype none // prevents system from inferring an undeclared logic (good practice)

module vwb_macc
#(  parameter InVecLength,
    parameter WorkingRegs,
    parameter NBits,
    parameter WeightFile,
    parameter BiasFile ) (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire signed [WorkingRegs-1:0][NBits-1:0] in_data,
    output logic signed [WorkingRegs-1:0][NBits-1:0] write_out_data,
    output logic req_chunk_in,
    output logic req_chunk_out,
    output logic out_vector_valid
);

typedef enum logic [1:0] {WAITING, LOADING, PROCESSING} bias_state;
bias_state state;
logic signed [WorkingRegs-1:0][NBits-1:0] bias_regs;
logic signed [WorkingRegs-1:0][NBits-1:0] weight_regs;
logic [$clog2(InVecLength/WorkingRegs)-1:0] weightbias_ptr;

// assumes single-cycle fifo
logic [$clog2(InVecLength):0] vec_in_idx;
logic [$clog2(InVecLength):0] vec_out_idx; // InVecLength = OutVecLength, one-to-one map
logic vec_op_complete;
assign vec_op_complete = vec_out_idx == 0;

xilinx_single_port_ram_read_first #(
  .RAM_WIDTH(WorkingRegs*NBits),
  .RAM_DEPTH(InVecLength/WorkingRegs),
  .RAM_PERFORMANCE("LOW_LATENCY"),
  .INIT_FILE(BiasFile)) bias_ram (
  .addra(weightbias_ptr),
  .dina(0),
  .clka(clk_in),
  .wea(1'd0),
  .ena(1'd1),
  .rsta(rst_in),
  .regcea(1'd1),
  .douta(bias_regs)
);

xilinx_single_port_ram_read_first #(
  .RAM_WIDTH(WorkingRegs*NBits),
  .RAM_DEPTH(InVecLength/WorkingRegs),
  .RAM_PERFORMANCE("LOW_LATENCY"),
  .INIT_FILE(WeightFile)) weight_ram (
  .addra(weightbias_ptr),
  .dina(0),
  .clka(clk_in),
  .wea(1'd0),
  .ena(1'd1),
  .rsta(rst_in),
  .regcea(1'd1),
  .douta(weight_regs)
);


genvar i
generate 
  for (i=0; i < WorkingRegs; i++) begin
    macc1d_fplib(
      .m($signed(weight_regs[WorkingRegs -1 -i])), 
      .x($signed(in_data[i])), 
      .b($signed(bias_regs[WorkingRegs -1 -i])), 
      .y($signed(write_out_data[i])))
  end
endgenerate

always_ff @(posedge clk_in) begin
  if(rst_in) begin
    vec_in_idx <= 0;
    vec_out_idx <= WorkingRegs;
    bias_ptr <= 0;
    state <= WAITING;
    req_chunk_in <= 0;
    req_chunk_out <= 0;
  end else begin
    if(state == WAITING) begin
        if(in_data_ready) begin
          vec_in_idx <= WorkingRegs >= InVecLength ? 0: WorkingRegs;
          vec_out_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
          req_chunk_in <= WorkingRegs < InVecLength;
          req_chunk_out <= 1;
          bias_ptr <= InVecLength/WorkingRegs > 1;
          state <= PROCESSING;
        end else begin
          vec_out_idx <= WorkingRegs;
          req_chunk_in <= 0;
          req_chunk_out <= 0;
        end
    end else if (state == PROCESSING) begin
      vec_in_idx <= vec_in_idx + WorkingRegs >= InVecLength ? 0 : vec_in_idx + WorkingRegs;
      vec_out_idx <= vec_out_idx + WorkingRegs >= InVecLength ? 0 : vec_out_idx + WorkingRegs;
      if(vec_op_complete) begin
        req_chunk_in <= in_data_ready;
        state <= in_data_ready ? PROCESSING : WAITING;
        vec_out_idx <= WorkingRegs;
        req_chunk_out <= in_data_ready;
      end else begin
        bias_ptr <= bias_ptr + 1;
        req_chunk_in <= 1;
        req_chunk_out <= 1;
      end
    end
  end
end

assign out_vector_valid = vec_out_idx == 0;

endmodule

`default_nettype wire