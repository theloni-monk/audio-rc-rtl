`default_nettype none

module cat2 #(
  parameter NBits,
  parameter VecElements0,
  parameter ElementsPerWrite0,
  parameter VecElements1,
  parameter ElementsPerWrite1,
  parameter ElementsPerRead
)(
input wire clk_in,
input wire rst_in,
input wire wr_en0,
input wire [ElementsPerWrite0-1:0][NBits-1:0] wr_data0,
input wire wr_en1,
input wire [ElementsPerWrite1-1:0][NBits-1:0] wr_data1,
input wire rd_en,
output logic [ElementsPerRead-1:0][NBits-1:0] rd_data,
output logic cat_valid,
output logic wr_collision
);

localparam TotalBits = (VecElement0 + VecElements1)*NBits;
localparam StartAddr1 = VecElement0 * NBits;
localparam MaxAddr0 = (VecElement0 - 1)*NBits;
localparam MaxAddr1 = (VecElement0 + VecElements1 - 1)*NBits;
localparam WrAdv0 = NBits * ElementsPerWrite0;
localparam WrAdv1 = NBits * ElementsPerWrite1;
localparam RdAdv = NBits * ElementsPerRead1;

logic [TotalBits-1:0] mem;
logic [$clog2(TotalBits):0] wr_ptr0;
logic [$clog2(TotalBits):0] wr_ptr1;
logic collision;
assign wr_collision = collision;
logic [$clog2(TotalBits):0] rd_ptr;

always_comb begin
  collision = wr_ptr0 > MaxAddr0 || (wr_ptr1 < MaxAddr0) || (wr_ptr1 > MaxAddr1);
  cat_valid = (wr_ptr0 == 0) & (wr_ptr1 == MaxAddr0);
end
always_ff @(posedge clk_in) begin
    if (~rst_in) begin
      wr_ptr0 <= 0;
      wr_ptr1 <= StartAddr1;
      rd_ptr <= 0;
      for(int i = 0; i<Depth*VecElements; i=i+1) mem[NBits*i +: NBits] <= 0;
    end else begin
      if (collision) begin
        wr_ptr0 <= -1;
        wr_ptr1 <= -1;
      end
      else begin
        if(wr_en0) begin
          if (wr_ptr0 + WrAdv0 >= MaxAddr0) wr_ptr0 <= 0;
          else wr_ptr0 <= wr_ptr0 + WrAdv0;
          mem[wr_ptr0 +: NBits*ElementsPerWrite] <= wr_data0;
        end
        if(wr_en1) begin
          if (wr_ptr1 + WrAdv0 >= MaxAddr1) wr_ptr1 <= StartAddr1;
          else wr_ptr1 <= wr_ptr1 + WrAdv1;
          mem[wr_ptr1 +: NBits*ElementsPerWrite] <= wr_data1;
        if (rd_rst) begin
          rd_ptr <= rd_ptr - TotalBits;
        end else if(rd_en) begin
          if (rd_ptr + RdAdv >= TotalBits) rd_ptr <= 0;
          else rd_ptr <= rd_ptr + RdAdv;
        end
        end
      end
  end
end

assign rd_data = mem[rd_ptr +: RdAdv];
endmodule

`default_nettype none
