# Gradients, Jacobians &amp; Backprop

Deck 09 of the [Linear Algebra for AI / ML](https://github.com/BrendanJamesLynskey/LLM_Hub_Linear_Algebra) series.

**Live presentation:** https://brendanjameslynskey.github.io/Linear_Algebra_AI_09_Gradients_Jacobians_Backprop/

A derivative is the best linear approximation of a function. The Jacobian is what that linear approximation looks like as a matrix. The chain rule is matrix multiplication. Backprop is just multiplying these matrices in the cheap order. Includes an interactive backprop walker that animates a 3-layer MLP forward and backward step by step.

## What's inside

- Derivative as the best linear approximation: $f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0) + J\mathbf{h} + o(\|\mathbf{h}\|)$
- The Jacobian matrix; gradient as transpose of a row Jacobian
- Chain rule = matrix multiplication; product of singular values across depth (vanishing/exploding gradients)
- JVP (forward mode) vs VJP (reverse mode); cost rule of thumb
- Why backprop is reverse mode &mdash; the $P\times$ saving for loss-of-many-parameters
- Activation memory and the tricks to reduce it (gradient checkpointing, FlashAttention, mixed precision, ZeRO)
- Common layers' Jacobians: linear, bias, elementwise activation, softmax, sum
- Backward through a linear layer: $\nabla_\mathbf{x} = W^\top \nabla_\mathbf{y}$, $\nabla_W = \nabla_\mathbf{y}\mathbf{x}^\top$ &mdash; the 3&times; training cost rule
- Softmax + cross-entropy: the gradient is $\mathbf{p} - \mathbf{y}^\star$, with all the complexity cancelled
- Interactive backprop walker (3-layer MLP, watch activations and gradients propagate step by step)

Single-page HTML, KaTeX-rendered maths, no build step. Open `index.html` directly.
