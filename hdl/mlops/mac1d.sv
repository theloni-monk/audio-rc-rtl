`timescale 1ps/1ps
`define fp_max(a,b)       (((a) > (b)) ? (a) : (b))

// Access iw, qw, and wl parameters of a ufp / sfp :
// Simulators cannot generally access interface parameters directly.
// However, $bits() gets special treatment and its output can be used in
// other constant expressions, so we encode parameters as the width of dummy
// signals inside the interface and call $bits() to get to the values indirectly
// For Xcelium, it might be required to pass the flag "-vlogcontrolrelax NOTDOT"
// This is likely to work on other simulators as well, but YMMV
// For Verilator, see https://github.com/verilator/verilator/issues/1593
`define fp_qw(fp) ($bits(fp.dummy_qw)-1)
`define fp_wl(fp) ($bits(fp.dummy_wl)-1)
`define fp_iw(fp) (`fp_wl(fp)-`fp_qw(fp)) // doing this instead of ($bits(fp.dummy_iw)-1) to allow negative iw

// calculate the # of bits needed to hold the full results of an add/sub/mult
`define add_iw(fp1, fp2) (`fp_max(`fp_iw(fp1), `fp_iw(fp2)) + 1)
`define add_qw(fp1, fp2) (`fp_max(`fp_qw(fp1), `fp_qw(fp2)))
`define sub_iw(fp1, fp2) `add_iw(fp1, fp2)
`define sub_qw(fp1, fp2) `add_qw(fp1, fp2)
`define mult_iw(fp1, fp2) (`fp_iw(fp1) + `fp_iw(fp2))
`define mult_qw(fp1, fp2) (`fp_qw(fp1) + `fp_qw(fp2))

// 'real' representing an IEEE float NaN
`define float_nan ($bitstoreal(64'hffffffffffffffff))

// convert a fp.val to a 'real' (any X/Z bits result in a float NaN)
`define fp_to_float(x, qw) ( $isunknown(x) ? `float_nan : real'(x) * (2.0 ** -real'(qw)) )


module mac1d #(
    parameter IW_M = 4, parameter QW_M = 8, // int/frac width of m
    parameter IW_X = 4, parameter QW_X = 8, // int/frac width of x
    parameter IW_B = 4, parameter QW_B = 8, // int/frac width of b
    parameter IW_Y = 4, parameter QW_Y = 8  // int/frac width of y
)(
    input  logic signed [IW_M + QW_M - 1:0] m,
    input  logic signed [IW_X + QW_X - 1:0] x,
    input  logic signed [IW_B + QW_B - 1:0] b,
    output logic signed [IW_Y + QW_Y - 1:0] y
);

    // determine FP format of prod = m * x and perform the mult
    localparam IW_PROD = IW_M + IW_X;
    localparam QW_PROD = QW_M + QW_X;
    wire signed [IW_PROD + QW_PROD - 1:0] prod = m * x;

    // determine FP format of sum = prod + b
    localparam IW_SUM = `fp_max(IW_PROD, IW_B) + 1;
    localparam QW_SUM = `fp_max(QW_PROD, QW_B);
    localparam WL_SUM = IW_SUM + QW_SUM;

    // align binary point of prod and b and perform sum = prod + b
    logic signed [WL_SUM - 1:0] b_aligned;
    logic signed [WL_SUM - 1:0] prod_aligned;

    if (QW_B >= QW_PROD) begin
        assign b_aligned = WL_SUM'(b);
        assign prod_aligned = WL_SUM'(prod) <<< (QW_B - QW_PROD);
    end else begin
        assign b_aligned = WL_SUM'(b) <<< (QW_PROD - QW_B);
        assign prod_aligned = WL_SUM'(prod);
    end
    wire signed [WL_SUM - 1:0] sum = prod_aligned + b_aligned;

    // match frac width of sum with y by truncating LSBs or adding zero LSBs
    logic signed [IW_SUM + QW_Y - 1:0] tmp;
    if (QW_SUM >= QW_Y) assign tmp = $signed(sum[(WL_SUM - 1)-:(IW_SUM + QW_Y)]);
    else assign tmp = $signed({sum, (QW_Y - QW_SUM)'('b0)});

    // then match the integer bits by discarding MSBs or sign extending
    if (IW_SUM >= IW_Y) assign y = $signed(tmp[(IW_Y + QW_Y - 1):0]);
    else assign y = (IW_Y + QW_Y)'(tmp);

endmodule
