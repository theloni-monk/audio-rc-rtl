# TODOs

## MVP
### Software
- ~~[*] model arch~~
- ~~[*] training~~
- ~~[*] export~~
- ~~[*] validation~~
- ~~[*] quantization~~

### HDL

Port to Fixed Point Q4.8:
- ~~[*] Add~~
- ~~[*] Mul~~
- ~~[*] MatMul~~

Implement
- ~~[*] Gemm~~
- [ ] Clip (-1, 1)
- ~~[*] Bitshift ~~
- ~~[*] LeakyReLU128~~
- [ ] Concat

### Compiler
- [ ] gen fixed-point weight files
- [ ] BatchNorm as vwb_macc
- [ ] pow2 mul as v_bshift

### Testbench
- [*] Validate vw_matmul and vwb_gemm in cocotb
- [ ] 1step
- [ ] loopback
- [ ] integration (fake pqmf driver)

### Writeup
- [ ] Intro
- [ ] model justification / training process
- [ ] hdl implementations
- [ ] success?
- [ ] conclusion

## Stretch goals (unlikely)

#### Sigproc - skipping for now
- [ ] Averaging Filter
- [ ] Export quantized pqmf weights
- [ ] Symmetric FIR
- [ ] Polyphase decimation filter

### Compiler
- [ ] loopy traversal
- [ ] Axi-wrapping

### Hardware Deployment
- [ ] DMA debugging setup
- [ ] Mic access
- [ ] Mic gain tuning
- [ ] model deployment
- [ ] profit
