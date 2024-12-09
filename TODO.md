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

Port to Fixed Point Q2.6:
- ~~[*] vwb macc~~
- ~~[*] vw matmul~~

Implement

- [ ] vector-vector Add
- ~~[*] Gemm~~
- [ ] Clip (-1, 1)
- ~~[*] Bitshift~~
- ~~[*] LeakyReLU128~~
- [ ] Concat

### Testbench

- [*] Validate vw_matmul and vwb_gemm in cocotb
  - [*] bespoke fifo driver
  - [*] bespoke fifo monitor
  - [ ] sw model
  - [ ] debug
- [ ] 1step
- [ ] loopback
- [ ] integration (fake pqmf driver)

### Compiler

- ~~[*] gen fixed-point weight files~~
- ~~[*] BatchNorm as vwb_macc~~
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
