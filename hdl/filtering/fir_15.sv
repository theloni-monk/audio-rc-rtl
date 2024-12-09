`default_nettype none // prevents system from inferring an undeclared logic (good practice)


module fir_15 #
  (
    parameter integer C_S00_AXIS_TDATA_WIDTH  = 32,
    parameter integer C_M00_AXIS_TDATA_WIDTH  = 32
  )
  (
  // Ports of Axi Slave Bus Interface S00_AXIS
  input wire  s00_axis_aclk, s00_axis_aresetn,
  input wire  s00_axis_tlast, s00_axis_tvalid,
  input wire [C_S00_AXIS_TDATA_WIDTH-1 : 0] s00_axis_tdata,
  input wire [(C_S00_AXIS_TDATA_WIDTH/8)-1: 0] s00_axis_tstrb,
  output logic  s00_axis_tready,

  // Ports of Axi Master Bus Interface M00_AXIS
  input wire  m00_axis_aclk, m00_axis_aresetn,
  input wire  m00_axis_tready,
  output logic  m00_axis_tvalid, m00_axis_tlast,
  output logic [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
  output logic [$clog2(C_M00_AXIS_TDATA_WIDTH)-1: 0] m00_axis_tstrb
  );
  // LOCAL INVARIANTS
  localparam NUM_COEFFS = 15;
  logic signed [7:0] coeffs [NUM_COEFFS-1 : 0];

  //initializing values
  initial begin //updated you coefficients
    coeffs[0] = -8'sd2;
    coeffs[1] = -8'sd3;
    coeffs[2] = -8'sd4;
    coeffs[3] = 0;
    coeffs[4] = 8'sd9;
    coeffs[5] = 8'sd21;
    coeffs[6] = 8'sd32;
    coeffs[7] = 8'sd36;
    coeffs[8] = 8'sd32;
    coeffs[9] = 8'sd21;
    coeffs[10] = 8'sd9;
    coeffs[11] = 0;
    coeffs[12] = -8'sd4;
    coeffs[13] = -8'sd3;
    coeffs[14] = -8'sd2;
    $display("Done initializing FIR sim");
  end

  // STATE
  logic tlast_pipe [NUM_COEFFS-1:0];
  logic tlast_in_pipe;
  logic signed [C_M00_AXIS_TDATA_WIDTH-1:0] taps [NUM_COEFFS-1:0];
  logic [$clog2(NUM_COEFFS):0] kernel_overlap;
  logic outvalid;


  //COMB LOGIC
  always_comb begin
    m00_axis_tvalid = outvalid; // once the last databucket is full our output becomes valid
    m00_axis_tstrb = 0; // no data strb b/c we don't use bursts in axis
    s00_axis_tready = m00_axis_tready; // module is ready for data only when outbus is ready for data

    tlast_in_pipe = 0;
    foreach(tlast_pipe[i]) tlast_in_pipe |= tlast_pipe[i]; // need to flush buffer if a tlast has come in, even if new data is invalid
    m00_axis_tlast = tlast_pipe[0];

    m00_axis_tdata = taps[0];
  end

  always_ff @(posedge s00_axis_aclk) begin
  if(~s00_axis_aresetn | (~m00_axis_aresetn)) begin // annoying active low reset
    // RESET
    outvalid <= 0;
    for(integer i = 0; i<NUM_COEFFS; ++i) begin
      taps[i] = 0;
      tlast_pipe[i] = 0;
    end
    kernel_overlap <= 0;
  end else begin
    // PROPAGATE FLUSH SIGNAL
    tlast_pipe[NUM_COEFFS-1] <= s00_axis_tlast;
    for(integer i = 0; i<NUM_COEFFS-1; ++i) begin
        tlast_pipe[i] <= tlast_pipe[i+1];
    end
    // PROPAGATE TAPS if new data or still flushing old data
    if((s00_axis_tvalid | tlast_in_pipe) & m00_axis_tready)begin
      taps[NUM_COEFFS-1] <= (coeffs[NUM_COEFFS-1] * $signed(s00_axis_tdata)) & {8{s00_axis_tvalid}};
      for(integer i = 0; i<NUM_COEFFS-1; ++i) begin
        taps[i] <= (taps[i+1] + (coeffs[i] * $signed(s00_axis_tdata)));
      end
      outvalid <= kernel_overlap > NUM_COEFFS-1;
      kernel_overlap <= tlast_pipe[0] ? 0 : (kernel_overlap > NUM_COEFFS ? kernel_overlap : kernel_overlap + 1);
      // kernel_overlap <= kernel_overlap > NUM_COEFFS ? kernel_overlap + 1 : kernel_overlap;
    end else outvalid <= 0;
  end
  end

endmodule


`default_nettype wire