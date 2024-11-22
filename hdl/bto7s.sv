module bto7s(
        input wire [3:0]   x_in,
        output logic [6:0] s_out
        );
    logic [6:0] _out;
    always_comb begin
        case (x_in)
          0: _out = 7'b1111110;
          1: _out = 7'b0110000;
          2: _out = 7'b1101101;
          3: _out = 7'b1111001;
          4: _out = 7'b0110011;
          5: _out = 7'b1011011;
          6: _out = 7'b1011111;
          7: _out = 7'b1110000;
          8: _out = 7'b1111111;
          9: _out = 7'b1111011;
          10: _out = 7'b1110111;
          11: _out = 7'b0011111;
          12: _out = 7'b1001110;
          13: _out = 7'b0111101;
          14: _out = 7'b1001111;
          15: _out = 7'b1000111;
          default: _out = 7'b1010101;
        endcase
    end
    assign s_out = {_out[0],_out[1],_out[2],_out[3],_out[4],_out[5],_out[6]};
endmodule
