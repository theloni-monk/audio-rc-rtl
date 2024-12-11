`default_nettype none // prevents system from inferring an undeclared logic (good practice)

// TODO: force bram syntehsis
// TODO: stale address tracking and overwrite flag
module v_fifo #(
  parameter  VecElements,
  parameter  ElementsPerWrite,
  parameter  ElementsPerRead,
  parameter  NBits,
  parameter  Depth )(
  input  wire clk_in,
  input wire rst_in,
  input wire wr_en,
  input wire [ElementsPerWrite-1:0][NBits-1:0] wr_data,
  input wire ptr_rst,
  input wire rd_en,
  output logic [ElementsPerRead-1:0][NBits-1:0] rd_data
);

localparam TotalBits = Depth*VecElements*NBits;
localparam VecLen = VecElements * NBits;
localparam WrAdv = NBits * ElementsPerWrite;
localparam RdAdv = NBits*ElementsPerRead;
logic [TotalBits-1:0] mem;
logic [$clog2(TotalBits):0] wr_ptr;
logic [$clog2(TotalBits):0] rd_ptr;

always_ff @(posedge clk_in) begin
    if (~rst_in) begin
      wr_ptr <= 0;
      rd_ptr <= 0;
      for(int i = 0; i<Depth*VecElements; i=i+1) mem[NBits*i +: NBits] <= 0;
    end else begin
      if(wr_en) begin
        if (wr_ptr + WrAdv >= TotalBits) wr_ptr <= 0;
        else wr_ptr <= wr_ptr + WrAdv;
        mem[wr_ptr +: NBits*ElementsPerWrite] <= wr_data;
      end
      if (ptr_rst) begin
        rd_ptr <= rd_ptr - TotalBits;
      end else if(rd_en) begin
        if (rd_ptr + RdAdv >= TotalBits) rd_ptr <= 0;
        else rd_ptr <= rd_ptr + RdAdv;
      end
    end

end

assign rd_data = mem[rd_ptr +: RdAdv];

endmodule;

`default_nettype wire