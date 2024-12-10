`timescale 1ps/1ps
`default_nettype none
module addertree #(parameter Elements = 12, parameter NBits=12)(
  input wire clk_in,
  input wire [Elements-1:0][2*NBits-1:0] in,
  output logic signed [2*NBits-1:0] out
);
generate
  if(Elements == 1) begin
    always_ff @(posedge clk_in) out <= $signed(in);
  end else if(Elements == 2) begin
    always_ff @(posedge clk_in) out <= $signed(in[0]) + $signed(in[1]);
  end else begin
    
    logic [Elements-(Elements/2)-1:0][2*NBits-1:0] leftin;
    logic [(Elements/2)-1:0][2*NBits-1:0] rightin;
    logic [2*NBits-1:0] leftout;
    logic [2*NBits-1:0] rightout;

    always_ff @(posedge clk_in) begin
      leftin <= in[Elements-1:Elements/2];
      rightin <= in[Elements/2-1:0];
    end

    addertree #(.Elements(Elements - Elements/2)) ladder (
      .clk_in(clk_in),
      .in(leftin),
      .out(leftout)
    );
    addertree #(.Elements(Elements/2)) radder (
      .clk_in(clk_in),
      .in(rightin),
      .out(rightout)
    );
    assign out = $signed(leftout + rightout);
  end
endgenerate
endmodule

`default_nettype wire