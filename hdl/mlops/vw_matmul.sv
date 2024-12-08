`timescale 1ns / 1ps
`default_nettype none // prevents system from inferring an undeclared logic (good practice)
module vw_matmul
#(  parameter InVecLength,
    parameter OutVecLength,
    parameter WorkingRegs,
    parameter NBits,
    parameter AccumBits,
    parameter WeightFile ) (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire signed [WorkingRegs-1:0][NBits-1:0] in_data,
    output logic signed [NBits-1:0] write_out_data,

    output logic req_chunk_in,
    output logic req_chunk_ptr_rst,
    output logic req_chunk_out,

    output logic out_vector_valid
);

typedef enum logic [2:0] {WAITING, ACCUMULATING, FLUSHING} mvprod_state;
mvprod_state state;
localparam WeightEls = InVecLength*OutVecLength/WorkingRegs;
localparam WeightDepth = $clog2(WeightEls);
logic [WeightDepth-1:0] weight_ptr;
logic signed [WorkingRegs-1:0][NBits-1:0] vector_regs;
(* use_dsp = "yes" *) logic signed [WorkingRegs-1:0][AccumBits-1:0] product_regs;
logic signed [NBits-1:0] dot;
logic signed [AccumBits-1:0] accumulator;
logic signed [WorkingRegs-1:0][NBits-1:0] weight_regs;
// assumes single-cycle fifo
logic [$clog2(InVecLength):0] vec_in_idx;
logic [$clog2(OutVecLength):0] vec_out_idx;
logic row_op_complete;
assign row_op_complete = vec_in_idx == 0;
logic all_op_complete;
assign all_op_complete = vec_out_idx == 0;
//assign out_vector_valid = all_op_complete;
xilinx_single_port_ram_read_first #(
  .RAM_WIDTH(WorkingRegs*NBits),
  .RAM_DEPTH(WeightEls),
  .RAM_PERFORMANCE("LOW_LATENCY"),
  .INIT_FILE(WeightFile)) weight_ram (
  .addra(weight_ptr),
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
      .m($signed(weight_regs[WorkingRegs - 1 -i])), 
      .x($signed(vector_regs[i])), 
      .b(0), 
      .y($signed(product_regs[i])))
  end
endgenerate

localparam DotWaitCycles = $clog2(WorkingRegs);
logic [DotWaitCycles:0] dot_cycles;
addertree #(.Elements(WorkingRegs), .NBitsIn(AccumBits), .NBitsOut(AccumBits)) atree (
  .clk_in(clk_in),
  .in(product_regs),
  .out(dot)
);

always_ff @(posedge clk_in) begin
  if(~rst_in) begin // RESET ACTIVE LOW
    vec_in_idx <= 0;
    vec_out_idx <= 1;
    weight_ptr <= 0;
    accumulator <= 0;
    out_vector_valid <= 0;
    state <= WAITING;
    req_chunk_in <= 0;
    req_chunk_out <= 0;
    req_chunk_ptr_rst <= 0;
    dot_cycles <= 0;
  end else if(state == WAITING) begin
    // state transitions
    if(in_data_ready) begin
      state <= ACCUMULATING;
      req_chunk_in <= InVecLength > WorkingRegs;
    end else begin
      req_chunk_in <= 0;
      for(int i = 0; i<WorkingRegs; i=i+1) vector_regs[i] = 0;
    end
    // starts at 1 idx so we can detect completion via zero idx
    vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
    vec_out_idx <= 1;
    req_chunk_out <= 0;
    req_chunk_ptr_rst <= 0;
    weight_ptr <= 0;
    dot_cycles <= 0;
    accumulator <= 0;
    out_vector_valid <= 0;

  end else if(state == ACCUMULATING) begin
    out_vector_valid <= 0;
    req_chunk_out <= 0;
    req_chunk_ptr_rst <= 0;
  
    for(integer i = 0; i< WorkingRegs; i= i+1) vector_regs[i] <= in_data[i]; // injest
    
    // update weight ptr with wraparound if needed
    weight_ptr <= weight_ptr + 1 >= InVecLength*OutVecLength/WorkingRegs ? 0 : weight_ptr + 1;
  
    if(row_op_complete) begin // row-completed -> write to out buffer
      req_chunk_in <= ~((~all_op_complete) & WorkingRegs < InVecLength) & WorkingRegs < InVecLength;
      req_chunk_ptr_rst <= (~all_op_complete) & WorkingRegs < InVecLength;
      dot_cycles <= 1;
      state <= FLUSHING;
    end else begin
      req_chunk_in <= vec_in_idx < InVecLength - WorkingRegs;
      req_chunk_ptr_rst <= 0;
      vec_in_idx <= vec_in_idx + WorkingRegs >= InVecLength ? 0 : vec_in_idx + WorkingRegs;
      dot_cycles <= 0;
    end
    accumulator <= accumulator + dot;

  end else if(state==FLUSHING) begin
    // dot product pipelined adder tree takes DotWaitCycles to complete a sum
    if(dot_cycles == DotWaitCycles + 1) begin
      dot_cycles <= 0;
      req_chunk_out <= 1;
      write_out_data <= accumulator[NBits-1:0];

      // signal full valid output if complete
      if(all_op_complete) begin
        out_vector_valid <= 1;
        if(in_data_ready)begin
          vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
          vec_out_idx <= 1;
          req_chunk_in <= WorkingRegs < InVecLength;
          vector_regs <= 0;
          weight_ptr <= 0;
          state <= ACCUMULATING;
        end else state <= WAITING;
      end
      // return to accumulation state on the next row
      else begin
        state <= ACCUMULATING;
        accumulator <= 0;
        req_chunk_in <= WorkingRegs < InVecLength;
        vec_out_idx <= vec_out_idx + 1 >= OutVecLength ? 0 : vec_out_idx + 1;
        vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
      end
    end else begin
      // wait until dot product adder tree is done
      req_chunk_in <= 0;
      vector_regs <= 0;
      req_chunk_ptr_rst <= 0;
      accumulator <= accumulator + dot;
      dot_cycles <= dot_cycles + 1;
    end
  end
end
endmodule;


`default_nettype wire