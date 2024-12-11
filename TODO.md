# TODOs

## MVP

What I am expected to complete

### Software

- ~~[*] model arch~~
- ~~[*] training~~
- ~~[*] export~~
- ~~[*] validation~~
- ~~[*] quantization~~

### HDL

Port to Fixed Point Q4.8:
- ~~[*] vwb macc~~
- ~~[*] vw matmul~~

Implement

- [ ] vector-vector Add (interleaved)
- ~~[*] Gemm~~
- [*] Clip (-1, 1)
- ~~[*] Bitshift~~
- ~~[*] LeakyReLU128~~
- [*] Concat
- [ ] Clone
- [ ] Split

### Testbench

- ~~[*] Validate vwb_macc~~
  - ~~[*] bespoke fifo driver~~
  - ~~[*] bespoke fifo monitor~~
  - ~~[*] sw model~~
- ~~[*] Validate vwb_gemm in cocotb~~
- [ ] Validate remaining models
  -~~[*] leakyrelu~~
  - [ ] bshiftmul
  - [ ] clip
  - [ ] vvadd
  - [*] concat
- [ ] 1step
- [ ] loopback
- [ ] integration (fake pqmf driver)

### Compiler

- ~~[*] gen fixed-point weight files~~
- ~~[*] BatchNorm as vwb_macc~~
- [*] nonlinear traversal
- [ ] pow2 mul as v_bshift

### Writeup

- [ ] Intro
- [ ] model justification / training process
- [ ] hdl implementations
- [ ] success?
- [ ] conclusion

## Stretch goals (unlikely)

What would be really cool

### Sigproc

- [ ] Averaging Filter
- [ ] Export quantized pqmf weights
- [ ] Symmetric FIR
- [ ] Polyphase decimation filter

### Recursive Compilation

- [ ] loopy traversal
- [ ] Axi-wrapping

### Hardware Deployment

- [ ] DMA debugging setup
- [ ] Mic access
- [ ] Mic gain tuning
- [ ] model deployment
- [ ] profit
