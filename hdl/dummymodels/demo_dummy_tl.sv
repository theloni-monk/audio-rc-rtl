`timescale 1ps/1ps
`default_nettype none
module demo_dummy_tl (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire [9:0][11:0] in_data_top,
    input wire rd_out_top,
    output logic out_data_valid,
    output logic module_ready,
    output logic [2:0][11:0] out_data_top
);
logic ovalid;
always_ff @(posedge clk_in) begin
if(~rst_in) begin
    module_ready <= 1;
    out_data_valid <= 0;
end else begin
    if(in_data_ready) module_ready <= 0;
    else if(ovalid) module_ready <= 1;
    out_data_valid <= ovalid;
end
end


logic [9:0][11:0] rd_data_0;

logic  rd_en_0;
logic  v_fifo_0_out_vec_valid_0;
logic  ptr_rst_0;
v_fifo #(
        .VecElements(10),
        .ElementsPerRead(10),
        .ElementsPerWrite(10),
        .NBits(12),
        .Depth(1))
        v_fifo_in_top
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(in_data_ready),
            .wr_data(in_data_top),
            .rd_en(rd_en_0),
            .rd_data(rd_data_0),
            .ptr_rst(ptr_rst_0)
        );



logic [11:0] wr_data_5;

logic  wr_en_5;

logic  v_fifo_5_out_vec_valid_5;
logic  ptr_rst_5;
v_fifo #(
        .VecElements(3),
        .ElementsPerRead(3),
        .ElementsPerWrite(1),
        .NBits(12),
        .Depth(1))
        v_fifo_out_top
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_5),
            .wr_data(wr_data_5),
            .rd_en(rd_out_top),
            .rd_data(out_data_top),
            .ptr_rst(1'b0)
        );




logic [11:0] wr_data_1;

logic  wr_en_1;
logic  vwb_gemm_0_out_vec_valid_0;
vwb_gemm #(  .InVecLength(10),
            .OutVecLength(20),
            .WorkingRegs(10),
            .NBits(12),
            .WeightFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin1_Gemm_weight.mem"),
            .BiasFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin1_Gemm_bias.mem"))
        vwb_gemm_0
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .in_data_ready(in_data_ready),
            .in_data(rd_data_0),
            .write_out_data(wr_data_1),
            .req_chunk_in(rd_en_0),
            .req_chunk_out(wr_en_1),
            .out_vector_valid(vwb_gemm_0_out_vec_valid_0)
        );


logic [19:0][11:0] rd_data_1;

logic  rd_en_1;
logic  v_fifo_1_out_vec_valid_1;
logic  ptr_rst_1;
v_fifo #(
        .VecElements(20),
        .ElementsPerRead(20),
        .ElementsPerWrite(1),
        .NBits(12),
        .Depth(1))
        v_fifo_1
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_1),
            .wr_data(wr_data_1),
            .rd_en(rd_en_1),
            .rd_data(rd_data_1),
            .ptr_rst(1'b0)
        );


logic [19:0][11:0] wr_data_2;

logic  wr_en_2;
logic  v_leakyrelu_0_out_vec_valid_0;
v_leakyrelu #(
  .InVecLength(20),
  .NBits(12),
  .WorkingRegs(20)) v_leakyrelu_0 (
  .clk_in(clk_in),
  .rst_in(rst_in),
  .in_data_ready(vwb_gemm_0_out_vec_valid_0),
  .in_data(rd_data_1),
  .write_out_data(wr_data_2),
  .req_chunk_in(rd_en_1),
  .req_chunk_out(wr_en_2),
  .out_vector_valid(v_leakyrelu_0_out_vec_valid_0)
            );

logic [19:0][11:0] rd_data_2;

logic  rd_en_2;
logic  v_fifo_2_out_vec_valid_2;
logic  ptr_rst_2;
v_fifo #(
        .VecElements(20),
        .ElementsPerRead(20),
        .ElementsPerWrite(20),
        .NBits(12),
        .Depth(1))
        v_fifo_2
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_2),
            .wr_data(wr_data_2),
            .rd_en(rd_en_2),
            .rd_data(rd_data_2),
            .ptr_rst(ptr_rst_2)
        );


logic [11:0] wr_data_3;

logic  wr_en_3;
logic  vwb_gemm_1_out_vec_valid_1;
vwb_gemm #(  .InVecLength(20),
            .OutVecLength(40),
            .WorkingRegs(20),
            .NBits(12),
            .WeightFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin2_Gemm_weight.mem"),
            .BiasFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin2_Gemm_bias.mem"))
        vwb_gemm_1
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .in_data_ready(v_leakyrelu_0_out_vec_valid_0),
            .in_data(rd_data_2),
            .write_out_data(wr_data_3),
            .req_chunk_in(rd_en_2),
            .req_chunk_out(wr_en_3),
            .out_vector_valid(vwb_gemm_1_out_vec_valid_1)
        );


logic [39:0][11:0] rd_data_3;

logic  rd_en_3;
logic  v_fifo_3_out_vec_valid_3;
logic  ptr_rst_3;
v_fifo #(
        .VecElements(40),
        .ElementsPerRead(40),
        .ElementsPerWrite(1),
        .NBits(12),
        .Depth(1))
        v_fifo_3
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_3),
            .wr_data(wr_data_3),
            .rd_en(rd_en_3),
            .rd_data(rd_data_3),
            .ptr_rst(1'b0)
        );


logic [39:0][11:0] wr_data_4;

logic  wr_en_4;
logic  v_leakyrelu_1_out_vec_valid_1;
v_leakyrelu #(
  .InVecLength(40),
  .NBits(12),
  .WorkingRegs(40)) v_leakyrelu_1 (
  .clk_in(clk_in),
  .rst_in(rst_in),
  .in_data_ready(vwb_gemm_1_out_vec_valid_1),
  .in_data(rd_data_3),
  .write_out_data(wr_data_4),
  .req_chunk_in(rd_en_3),
  .req_chunk_out(wr_en_4),
  .out_vector_valid(v_leakyrelu_1_out_vec_valid_1)
            );

logic [39:0][11:0] rd_data_4;

logic  rd_en_4;
logic  v_fifo_4_out_vec_valid_4;
logic  ptr_rst_4;
v_fifo #(
        .VecElements(40),
        .ElementsPerRead(40),
        .ElementsPerWrite(40),
        .NBits(12),
        .Depth(1))
        v_fifo_4
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_4),
            .wr_data(wr_data_4),
            .rd_en(rd_en_4),
            .rd_data(rd_data_4),
            .ptr_rst(ptr_rst_4)
        );






vwb_gemm #(  .InVecLength(40),
            .OutVecLength(3),
            .WorkingRegs(40),
            .NBits(12),
            .WeightFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin3_Gemm_weight.mem"),
            .BiasFile("c:/Users/TheoA/Documents/MIT/SENIORSLIDE/6.S965 Digital Systems Lab II/Final Proj/audio-rc-rtl/sim/sim/data/_lin3_Gemm_bias.mem"))
        vwb_gemm_2
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .in_data_ready(v_leakyrelu_1_out_vec_valid_1),
            .in_data(rd_data_4),
            .write_out_data(wr_data_5),
            .req_chunk_in(rd_en_4),
            .req_chunk_out(wr_en_5),
            .out_vector_valid(ovalid)
        );


endmodule;
