// performs m * x + b, where y, m, x, and b are arbitary FP numbers
module macc1d_fplib (
  sfp.in    m, x, b,    // input sfp signals
  sfp.out   y           // output sfp signal
);
  // define prod with the right format to hold m * x
  sfp #(`mult_iw(m, x), `mult_qw(m, x)) prod();

  // perform m * x = prod
  sfp_mult_full mult (m, x, prod);

  // perform prod + b and resize to fit the format of y
  sfp_add add (prod, b, y);
endmodule