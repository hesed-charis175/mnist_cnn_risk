# mnist_cnn_risk
 
A small convolutional neural network for MNIST digit recognition, written from scratch in Risk.
- **Train** on the 60,000 MNIST images and save the weights to a 24 KB binary file.
- **Infer** interactively: draw a digit with your mouse in an X11 window and see the model's prediction and per-class probabilities.
## Repository layout
 
| File | Purpose |
|---|---|
| `download_mnist.sh` | Downloads the MNIST IDX files into `mnist_data/` |
| `mnist_cnn_train.risk` | Loads data, trains the CNN with mini-batch SGD, evaluates on the test set, writes `cnn_weights.bin` |
| `mnist_cnn_infer.risk` | Loads `cnn_weights.bin`, opens an X11 canvas, preprocesses your drawing and predicts the digit |
 
## Architecture
 
```
Input        1 x 28 x 28
Conv1        8 filters, 5x5, stride 1, no padding  ->  8 x 24 x 24   + ReLU
MaxPool      2x2                                   ->  8 x 12 x 12
Conv2        16 filters, 5x5 (over 8 channels)     -> 16 x  8 x  8   + ReLU
MaxPool      2x2                                   -> 16 x  4 x  4
Flatten                                            -> 256
Dense        256 -> 10
Softmax + cross-entropy loss
```
 
**Parameters:** 5,994 in total (`W1` 200, `b1` 8, `W2` 3,200, `b2` 16, `W3` 2,560, `b3` 10), i.e. 23,976 bytes as `f32`.
 
### Implementation notes
 
- **Convolution via `im2col` + GEMM.** Each convolution is turned into one `cblas_sgemm` call by unrolling input patches into a column matrix (`im2col_1`, `im2col_2`).
- **Batched layout.** Activations are stored as `[filters, batch * H * W]`, so a whole mini-batch goes through a single GEMM per layer.
- **Manual backprop.** Gradients flow back through softmax/cross-entropy, the dense layer, max-pool (via stored argmax indices), ReLU (via stored masks), and convolutions (two GEMMs each: one for weight gradients, one for `dcol`, followed by a `col2im`-style scatter).
- **Initialisation:** Glorot/Xavier uniform. **Loss:** numerically-stable softmax (max subtraction) with a clamped log.
- **Optimiser:** plain mini-batch SGD.
### Training hyperparameters
 
| Setting | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 0.1 |
| Epochs | 3 |
| Pixel scaling | `/ 255` |
 
## How to run

### Prerequisites

- The `riskc` Risk compiler
- A BLAS library providing `cblas_sgemm` (e.g. OpenBLAS)
- X11 (needed only for the interactive inference window)

### 1. Download MNIST

```bash
./download_mnist.sh
```

### 2. Compile

```bash
riskc mnist_cnn_train.risk -o train -O3 -march=native
riskc mnist_cnn_infer.risk -o infer -O3 -march=native
```

### 3. Train

```bash
./train
```

Training runs for 3 epochs and prints the loss every 200 steps, plus the test accuracy after each epoch. You should see test accuracy of about:

| Epoch | Test accuracy |
|---|---|
| 1 | ~97.6% |
| 2 | ~98.0% |
| 3 | ~98.2% |

When it finishes, the weights are saved to `cnn_weights.bin`.

### 4. Run inference

```bash
./infer
```

An X11 window opens. Draw a digit with the mouse and the model shows its prediction along with the per-class probabilities.

### Quick start

```bash
./download_mnist.sh
riskc mnist_cnn_train.risk -o train -O3 -march=native
riskc mnist_cnn_infer.risk -o infer -O3 -march=native
./train && ./infer
```
