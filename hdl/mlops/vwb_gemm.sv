`timescale 1ps/1ps
`default_nettype none
 // prevents system from inferring an undeclared logic (good practice)
 module clip_signed
(
  input wire signed [23:0]      in,         // input
  output wire signed [11:0]    out        // clipped output
);
  localparam inw = 24;
  localparam outw = 12;
  // selects the bits that are to be checked for clipping (without the sign bit)
  wire [inw-2:outw-1] msbs = in[inw-2:outw-1];

  // the sign bit
  wire signbit = in[inw-1];

  // check if there was a positive or a negative clip
  wire positiveclip =  (|(msbs)) && !signbit;
  wire negativeclip = !(&(msbs)) && signbit;

  // full scale positive and negative value
  wire [outw-1:0] maxval = {1'b0, (outw-1)'('1)};  // 0111111...
  wire [outw-1:0] minval = {1'b1, (outw-1)'('0)};  // 1000000...

  // clipped value
  assign out = positiveclip ? maxval :
               negativeclip ? minval : in[outw-1:0];
endmodule

module fxp24_to_12(input wire signed [23:0] in,
                    output logic signed [11:0] out);
                    localparam iniw  = 8;
                    localparam inqw  = 16;
                    localparam inw   = 24;
                    localparam outiw = 4;
                    localparam outqw = 8;
                    localparam outw  = 12;
                    localparam tmp1w  = iniw+outqw;
                    localparam tmp2w  = outiw+inqw;
                    logic signed [tmp1w-1:0] tmp1;
                    //FIXME: clipping is a bit suspicious
                    // first handle the franctional bits by truncating LSBs or padding zeros
                    assign tmp1 = $signed(in[inw-1-:tmp1w]);
                    // then handle the integer bits by clipping / discarding MSBs (may causing wrapping!), or sign extending MSBs
                    clip_signed u_clip (.in(tmp1), .out(out));
endmodule

module fxp12_to_24(input wire signed [11:0] in,
  output logic signed [23:0] out);
  logic [7:0] intpad;
  assign intpad = $signed(in[11:8]);
  logic [15:0] fracpad;
  assign fracpad = {in[7:0], 8'b0};
  assign out = {intpad, fracpad};
endmodule

 module vwb_gemm
#(  parameter InVecLength,
    parameter OutVecLength,
    parameter WorkingRegs,
    parameter NBits,
    parameter WeightFile,
    parameter BiasFile) (
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
//NOT DRY
typedef enum logic [2:0] {WAITING, ACCUMULATING, FLUSHING} mvprod_state;
mvprod_state state;
localparam WeightChunks = InVecLength*OutVecLength/WorkingRegs;
localparam WeightDepth = $clog2(WeightChunks);
localparam AccumBits = 2*NBits;
logic [WeightDepth-1:0] weight_ptr;
logic signed [WorkingRegs-1:0][NBits-1:0] vector_regs;
(* use_dsp = "yes" *) logic signed [WorkingRegs-1:0][AccumBits-1:0] product_regs;
logic signed [2*NBits-1:0] dot;
logic signed [AccumBits-1:0] accumulator;
logic signed [WorkingRegs-1:0][NBits-1:0] weight_regs;
// assumes single-cycle fifo
logic [$clog2(InVecLength)+1:0] vec_in_idx;
logic [$clog2(OutVecLength)+1:0] vec_out_idx;
logic row_op_complete;
assign row_op_complete = vec_in_idx == 0;
logic all_op_complete;
assign all_op_complete = vec_out_idx == 0;
logic ovvalid;
//assign out_vector_valid = all_op_complete;
xilinx_single_port_ram_read_first #(
  .RAM_WIDTH(WorkingRegs*NBits),
  .RAM_DEPTH(WeightChunks),
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
genvar i;
generate
  for (i=0; i < WorkingRegs; i++) begin
    mac1d #(.IW_Y(8), .QW_Y(16)) mac (
      .m($signed(weight_regs[i])),
      .x($signed(vector_regs[i])),
      .b($signed(12'b0)),
      .y(product_regs[i]));
  end
endgenerate

localparam DotWaitCycles = $clog2(WorkingRegs) + 1;
logic [DotWaitCycles:0] dot_cycles;
addertree #(.Elements(WorkingRegs), .NBits(NBits)) atree (
  .clk_in(clk_in),
  .in(product_regs),
  .out(dot)
);

logic signed [NBits-1:0] bias_reg; // ###########  DIFFERENCE FROM vw_matmul #############
logic signed [AccumBits-1:0] bias_expanded;
logic [$clog2(OutVecLength):0] bias_ptr;
assign bias_ptr = vec_out_idx ? vec_out_idx-1'b1: OutVecLength-1;
fxp12_to_24 upcvrt (.in(bias_reg), 
                    .out(bias_expanded));
xilinx_single_port_ram_read_first #( // ###########  DIFFERENCE FROM vw_matmul #############
  .RAM_WIDTH(NBits),
  .RAM_DEPTH(OutVecLength),
  .RAM_PERFORMANCE("LOW_LATENCY"),
  .INIT_FILE(BiasFile)) gemm_bias_ram (
  .addra(bias_ptr), // using the out_idx as the ptr for the bias bram
  .dina(0),
  .clka(clk_in),
  .wea(1'd0),
  .ena(1'd1),
  .rsta(rst_in),
  .regcea(1'd1),
  .douta(bias_reg)
);


logic [23:0] fxp24out;
fxp24_to_12 downcvrt (.in(fxp24out),
                      .out(write_out_data));

always_ff @(posedge clk_in) begin
  if(~rst_in) begin // RESET ACTIVE LOW
    vec_in_idx <= 0;
    vec_out_idx <= 1;
    weight_ptr <= 0;
    accumulator <= 0;
    out_vector_valid <= 0;
    ovvalid <= 0;
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
      for(int i = 0; i<WorkingRegs; i=i+1) vector_regs[i] <= 0;
    end
    // starts at 1 idx so we can detect completion via zero idx
    vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
    vec_out_idx <= 1;
    req_chunk_out <= 0;
    req_chunk_ptr_rst <= 0;
    weight_ptr <= 0;
    dot_cycles <= 0;
    accumulator <= 0;
    out_vector_valid <= ovvalid;
    ovvalid <= 0;

  end else if(state == ACCUMULATING) begin
    out_vector_valid <= ovvalid;
    ovvalid <= 0;
    req_chunk_out <= 0;
    req_chunk_ptr_rst <= 0;
    for(integer i = 0; i< WorkingRegs; i= i+1) vector_regs[i] <= in_data[i]; // injest
    
    // update weight ptr with wraparound if needed
    weight_ptr <= weight_ptr + 1 > WeightChunks -1 ? 0 : weight_ptr + 1;
    
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
    if(dot_cycles == 0) begin
      dot_cycles <= 0;
      req_chunk_out <= 1;
      fxp24out <= accumulator + bias_expanded; // ###########  DIFFERENCE FROM vw_matmul ############# HACK

      // signal full valid output if complete
      if(all_op_complete) begin
        ovvalid <= 1;
        if(in_data_ready)begin
          vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
          vec_out_idx <= 1;
          req_chunk_in <= WorkingRegs < InVecLength;
          vector_regs <= in_data;
          weight_ptr <= 0;
          state <= ACCUMULATING;
        end else state <= WAITING;
      end else begin// return to accumulation state on the next row
        state <= ACCUMULATING;
        accumulator <= 0;
        req_chunk_in <= WorkingRegs < InVecLength;
        vec_out_idx <= vec_out_idx + 1 == OutVecLength ? 0 : vec_out_idx + 1;
        vec_in_idx <= WorkingRegs >= InVecLength ? 0 : WorkingRegs;
      end
    end else begin
      // wait until dot product adder tree is done
      req_chunk_in <= 0;
      vector_regs <= -1;
      req_chunk_ptr_rst <= 0;
      accumulator <= accumulator + dot;
      dot_cycles <= dot_cycles + 1 > DotWaitCycles ? 0 : dot_cycles + 1;
    end
  end
end
endmodule;



`default_nettype wire