

logic [7:0] in_data_master;
logic [4:0][7:0] rd_data_0;
logic  wr_in_master;
logic  rd_en_0;
logic  v_fifo_0_out_vec_valid_0;
logic  wrap_rd_0;
v_fifo #(
        .VecElements(5),
        .ElementsPerRead(5),
        .ElementsPerWrite(1),
        .NBits(8)
        .Depth(1))
        v_fifo_0
        (
            .clk_in(clk_100mhz),
            .rst_in(sys_rst),
            .wr_en(wr_in_master),
            .wr_data(in_data_master),
            .rd_en(rd_en_0),
            .rd_data(rd_data_0),
            .wrap_rd(0)
        );


logic [7:0] wr_data_1;
logic [4:0][7:0] out_data_master;
logic  wr_en_1;
logic  rd_out_master;
logic  v_fifo_1_out_vec_valid_1;
logic  wrap_rd_1;
v_fifo #(
        .VecElements(5),
        .ElementsPerRead(5),
        .ElementsPerWrite(5),
        .NBits(8)
        .Depth(1))
        v_fifo_1
        (
            .clk_in(clk_100mhz),
            .rst_in(sys_rst),
            .wr_en(wr_en_1),
            .wr_data(wr_data_1),
            .rd_en(rd_out_master),
            .rd_data(out_data_master),
            .wrap_rd(0)
        );


module MLInference(
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    output logic out_data_ready
);

  





vwb_mac #(  .InVecLength(5),
            .WorkingRegs(5),
            .NBits(8),
            .WeightFile(onnx_bn_weight),
            .BiasFile("onnx_bn_bias"))
        vwb_mac_0
        (
            .clk_in(clk_in),
            .rst_in(rst_in),
            .in_data_ready(in_data_ready),
            .in_data(rd_data_0),
            .write_out_data(wr_data_1),
            .req_chunk_in(rd_en_0),
            .req_chunk_out(wr_en_1),
            .out_vector_valid(out_data_ready)
        );


endmodule;

logic ml_inf_valid;
MLInference ml_inf(
    .clk_in(clk_100mhz),
    .rst_in(sys_rst),
    .in_data_ready(in_data_ready_master),
    .out_data_ready(ml_inf_valid)
);
