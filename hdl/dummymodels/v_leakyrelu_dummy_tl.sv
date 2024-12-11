`timescale 1ps/1ps
`default_nettype none
module v_leakyrelu_dummy_tl (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire [15:0][11:0] in_data_top,
    input wire rd_out_top,
    output logic out_data_valid,
    output logic module_ready,
    output logic [15:0][11:0] out_data_top
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


logic [15:0][11:0] rd_data_0;

logic  rd_en_0;
logic  v_fifo_0_out_vec_valid_0;
logic  wrap_rd_0;
v_fifo #(
        .VecElements(16),
        .ElementsPerRead(16),
        .ElementsPerWrite(16),
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
            .wrap_rd(1'b0)
        );



logic [15:0][11:0] wr_data_1;

logic  wr_en_1;

logic  v_fifo_1_out_vec_valid_1;
logic  wrap_rd_1;
v_fifo #(
        .VecElements(16),
        .ElementsPerRead(16),
        .ElementsPerWrite(16),
        .NBits(12),
        .Depth(1))
        v_fifo_out_top
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .wr_en(wr_en_1),
            .wr_data(wr_data_1),
            .rd_en(rd_out_top),
            .rd_data(out_data_top),
            .wrap_rd(1'b0)
        );








v_leakyrelu #(
  .InVecLength(16),
  .NBits(12),
  .WorkingRegs(16)) v_leakyrelu_0 (
  .clk_in(clk_in),
  .rst_in(rst_in),
  .in_data_ready(in_data_ready),
  .in_data(rd_data_0),
  .write_out_data(wr_data_1),
  .req_chunk_in(rd_en_0),
  .req_chunk_out(wr_en_1),
  .out_vector_valid(ovalid)
            );

endmodule;
