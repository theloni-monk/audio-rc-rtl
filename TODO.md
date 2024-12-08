# TODOs

## MVP
### Software
- ~~[*] model arch~~
- ~~[*] training~~
- ~~[*] export~~
- ~~[*] validation~~
- ~~[*] quantization~~

### HDL

#### MLOps
Port to Fixed Point:
- [ ] Add
- [ ] Mul
- [ ] Gemm

Implement
- [ ] BatchNorm
- [ ] Clip (-1, 1)
- [ ] LeakyReLU
- [ ] Concat
- [ ] Pipeline balancer


### Testbench
- [ ] Validate MVProd in cocotb
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
