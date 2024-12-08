`timescale 1ps/1ps
`default_nettype none
//TODO: validate and optimize to only use in_bits in at top level

module addertree_inner #(parameter Elements, parameter NBits)(
  input wire clk_in,
  input wire [Elements-1:0][NBits-1:0] in,
  output logic signed [NBits-1:0] out
);
generate
  if(Elements == 1) begin
    always_ff @(posedge clk_in) out <= $signed(in);
  end else if(Elements == 2) begin
    always_ff @(posedge clk_in) out <= $signed(in[0]) + $signed(in[1]);
  end else begin
    always_ff @(posedge clk_in) begin
      leftin <= in[Elements-1:Elements/2];
      rightin <= in[Elements/2-1:0];
    end
    logic [Elements-(Elements/2)-1:0][NBits-1:0] leftin;
    logic [(Elements/2)-1:0][NBits-1:0] rightin;
    logic [NBits-1:0] leftout;
    logic [NBits-1:0] rightout;
    PipeAdderTree #(.Elements(Elements - Elements/2)) ladder (
      .clk_in(clk_in),
      .in(leftin),
      .out(leftout)
    );
    PipeAdderTree #(.Elements(Elements/2)) radder (
      .clk_in(clk_in),
      .in(rightin),
      .out(rightout)
    );
    assign out = leftout + rightout;
  end
endgenerate
endmodule

module addertree #( parameter Elements,
                    parameter NBitsIn,
                    parameter NBitsOut)(
  input wire clk_in,
  input wire [Elements-1:0][NBitsIn-1:0] in,
  output logic signed [NBitsOut-1:0] out
);
generate
  if(Elements == 1) begin
    always_ff @(posedge clk_in) out <= $signed(in);
  end else if(Elements == 2) begin
    always_ff @(posedge clk_in) out <= $signed(in[0]) + $signed(in[1]);
  end else begin
    // promote to output size
    logic [Elements-1:0][NBitsOut-1:0] in_expanded;
    foreach(in[i]) assign in_expanded[i] = $signed(in[i]);   
    always_ff @(posedge clk_in) begin
      leftin <= in_expanded[Elements-1:Elements/2];
      rightin <= in_expanded[Elements/2-1:0];
    end
    logic [Elements-(Elements/2)-1:0][NBitsOut-1:0] leftin;
    logic [(Elements/2)-1:0][NBitsOut-1:0] rightin;
    logic [NBitsOut-1:0] leftout;
    logic [NBitsOut-1:0] rightout;
    addertree_inner #(.Elements(Elements - Elements/2), .NBits(NBitsOut)) ladder (
      .clk_in(clk_in),
      .in(leftin),
      .out(leftout)
    );
    addertree_inner #(.Elements(Elements/2), .NBits(NBitsOut)) radder (
      .clk_in(clk_in),
      .in(rightin),
      .out(rightout)
    );
    assign out = leftout + rightout;
  end
endgenerate
endmodule
`default_nettype wire