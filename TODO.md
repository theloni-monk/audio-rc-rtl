# TODOs:

## Software
- ~~[*] model arch~~
- ~~[*] training~~
- ~~[*] export~~
- ~~[*] validation~~
- ~~[*] quatnziation~~

## HDL
### Sigproc
- [*] Averaging Filter
- [ ] Export quantized pqmf weights
- [ ] Symmetric FIR
- [ ] Polyphase decimation filter

### MLOps
Float domain:
- [*] QuantizeLinear
- [ ] DequantizeLinear
- [ ] BatchNormalization
- [ ] QLinearLeakyRelu (see if i can take advantage of >>7)
- [ ] QLinearConcat

Mixed domain:
- [ ] QLinearAdd
- [ ] QLinearMul
- [ ] QGemm

## Compiler
- [ ] loopy traversal
- [ ] Axi-wrapping

## Testbench
- [ ] QDQ
- [ ] 1step
- [ ] loopback
- [ ] integration (fake driver)

## Hardware
- [ ] DMA debugging setup
- [ ] Mic access
- [ ] Mic gain tuning
- [ ] model deployment
- [ ] profit

## Writeup
- [ ] Intro
- [ ] model justification / training process
- [ ] hdl implementations
- [ ] success?
- [ ] conclusion