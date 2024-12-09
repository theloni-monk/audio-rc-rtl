`timescale 1ps/1ps
`default_nettype none
module vwb_mac_dummy_tl (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire [10:0][8-1:0] in_data_top,
    input wire rd_out_top,
    output logic out_data_valid,
    output logic [10:0][8-1:0] out_data_top
);



logic [10:0][7:0] rd_data_0;

logic  rd_en_0;
logic  v_fifo_0_out_vec_valid_0;
logic  wrap_rd_0;
v_fifo #(
        .VecElements(11),
        .ElementsPerRead(11),
        .ElementsPerWrite(11),
        .NBits(8),
        .Depth(1))
        v_fifo_in_top
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(in_data_ready),
            .wr_data(in_data_top),
            .rd_en(rd_en_0),
            .rd_data(rd_data_0),
            .wrap_rd(0)
        );



logic [10:0][7:0] wr_data_1;

logic  wr_en_1;

logic  v_fifo_1_out_vec_valid_1;
logic  wrap_rd_1;
v_fifo #(
        .VecElements(11),
        .ElementsPerRead(11),
        .ElementsPerWrite(11),
        .NBits(8),
        .Depth(1))
        v_fifo_out_top
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_1),
            .wr_data(wr_data_1),
            .rd_en(rd_out_top),
            .rd_data(out_data_top),
            .wrap_rd(0)
        );








vwb_mac #(  .InVecLength(11),
            .WorkingRegs(11),
            .NBits(8),
            .WeightFile("../sim/data/onnx_bn_weight.mem"),
            .BiasFile("../sim/data/onnx_bn_bias.mem"))
        vwb_mac_0
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .in_data_ready(in_data_ready),
            .in_data(rd_data_0),
            .write_out_data(wr_data_1),
            .req_chunk_in(rd_en_0),
            .req_chunk_out(wr_en_1),
            .out_vector_valid(out_data_valid)
        );


endmodule;
