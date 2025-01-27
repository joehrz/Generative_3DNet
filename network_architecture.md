# BI‐Net Overview

**BI‐Net** is a **bidirectional network** that unifies:

1. An **auto‐encoder (AE) direction**: real point clouds $X \to z \to \widehat{X}$.  
2. A **GAN direction**: random noise $z \to \widetilde{X}$, plus a discriminator to distinguish real vs. fake.

The **decoder** in both directions is the same **TreeGCN** module. The **encoder** and **discriminator** can share parameters or be closely related modules.

---

## 1. Forward Direction (Auto‐Encoder)

Given a real point cloud $X \in \mathbb{R}^{N \times 3}$, the AE direction attempts to **reconstruct** $X$. This forms the **auto‐encoder** path.

### Encoder

We define an encoder $E(\cdot)$:
$$
z = E(X) \quad \in \mathbb{R}^d,
$$
where $d$ is the latent dimension. This might be done by point‐cloud convolutions or an MLP.

### Decoder

Next, a decoder $\text{De}(\cdot)$ reconstructs the point cloud:
$$
\widehat{X} = \text{De}(z) \quad \in \mathbb{R}^{N \times 3}.
$$
This is often a **TreeGCN** that expands $z$ from a small node set to $N$ final 3D points.

### Reconstruction Loss

We measure the distance between the real $X$ and the reconstructed $\widehat{X}$. For example, **Chamfer Distance**:

$$
d_{\mathrm{Chamfer}}(X, \widehat{X})
=
\sum_{x \in X}
  \min_{\hat{x} \in \widehat{X}}
  \|x - \hat{x}\|_2
\;+\;
\sum_{\hat{x} \in \widehat{X}}
  \min_{x \in X}
  \|\hat{x} - x\|_2,
$$

or **EMD**. The **AE loss** is then:
$$
L_{\mathrm{AE}}
=
\mathbb{E}_{X}
\Bigl[
  d\bigl(X, \widehat{X}\bigr)
\Bigr].
$$

---

## 2. Reverse Direction (GAN)

In the reverse direction, BI‐Net uses the **same decoder** as a **generator**, plus a **discriminator** that can share parameters with the encoder.

### Generator

A random latent vector $z \sim \mathcal{N}(0,I)$ is mapped to fake point clouds:
$$
\widetilde{X} = G(z).
$$
The module $G(\cdot)$ can be the same **TreeGCN** used by the decoder.

### Discriminator

For any point cloud (real or fake), a discriminator $D(\cdot)$ outputs a scalar score. In BI‐Net, this might share layers with the encoder.

---

### WGAN‐GP Objective

We use a **Wasserstein GAN** with Gradient Penalty:

- **Discriminator loss**:

  $$
  L_{D}
  =
  \mathbb{E}\bigl[D(\widetilde{X})\bigr]
  -
  \mathbb{E}\bigl[D(X)\bigr]
  +
  \lambda_{\mathrm{gp}}\;L_{\mathrm{gp}},
  $$
  
  where
  
  $$
  L_{\mathrm{gp}}
  =
  \mathbb{E}_{\hat{x} \in \mathrm{interp}}
  \Bigl[
    \bigl(\|\nabla_{\hat{x}}\,D(\hat{x})\|_2 - 1\bigr)^2
  \Bigr].
  $$

- **Generator loss**:

  $$
  L_{G}
  =
  -\,\mathbb{E}\bigl[D(\widetilde{X})\bigr].
  $$

#### Optional Uniformity (NNME) Loss

We can add a **Nearest Neighbor Mutual Exclusion (NNME)** term to encourage uniform point placement. The generator’s total cost becomes:

$$
L_{G}^{\mathrm{total}}
=
L_{G}
+
\lambda_{\mathrm{NNME}}
\,L_{\mathrm{NNME}}(\widetilde{X}).
$$

---

## 3. BI‐Net Training Strategy

Each iteration has **two phases**:

1. **AE direction**: encode real $X \to z$, decode to $\widehat{X}$, minimize the reconstruction distance $d(X,\widehat{X})$.  
2. **GAN direction**: generate $\widetilde{X} = G(z)$ from random noise, update $\max D, \min G$ via the WGAN‐GP objective (plus optional NNME).

Because the encoder and discriminator can share parameters, the forward (AE) training helps the network learn better geometric features, while the reverse (GAN) training forces the generator to produce realistic shapes. This **collaborative** approach stabilizes learning and often yields higher‐quality point clouds.

---

### TreeGCN Pipeline

Given an input embedding $X^{(l)} \in \mathbb{R}^{B \times N \times d_{\mathrm{in}}}$, a single TreeGCN layer outputs $X^{(l+1)} \in \mathbb{R}^{B \times (N \cdot \mathrm{degree}) \times d_{\mathrm{out}}}$. 

Let:
- $W_{\mathrm{root}} \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$,
- $W_{\mathrm{branch}} \in \mathbb{R}^{d_{\mathrm{in}} \times (d_{\mathrm{in}} \cdot \mathrm{degree})}$,
- $W_{\mathrm{loop}}:$ an MLP from $d_{\mathrm{in}}$ to $d_{\mathrm{out}}$,
- $b \in \mathbb{R}^{\mathrm{degree} \times d_{\mathrm{out}}}$.

Then for each node $x_{b,n} \in \mathbb{R}^{d_{\mathrm{in}}}$ (batch index $b$, node index $n$):

1. **Root transform**:

   $$
   r_{b,n}
   =
   x_{b,n}\,W_{\mathrm{root}}
   \quad\in\quad
   \mathbb{R}^{d_{\mathrm{out}}}.
   $$

2. **Branch transform**:

   $$
   z_{b,n}
   =
   x_{b,n}\,W_{\mathrm{branch}}
   \quad\in\quad
   \mathbb{R}^{(d_{\mathrm{in}} \cdot \mathrm{degree})}.
   $$
   Reshape to $z_{b,n}^{(k)} \in \mathbb{R}^{d_{\mathrm{in}}}$ for $k = 1,\dots,\mathrm{degree}$.

3. **Loop transform** each branch:

   $$
   \hat{z}_{b,n}^{(k)}
   =
   W_{\mathrm{loop}}\!\bigl(z_{b,n}^{(k)}\bigr)
   \quad\in\quad
   \mathbb{R}^{d_{\mathrm{out}}}.
   $$

4. **Combine root + branch**:

   $$
   y_{b,n}^{(k)}
   =
   r_{b,n}
   +
   \hat{z}_{b,n}^{(k)}
   +
   b^{(k)},
   $$
   then apply an activation (e.g., LeakyReLU).

Hence each parent node $(b,n)$ spawns $\mathrm{degree}$ children in $y_{b,n}^{(k)}$. Stacking layers grows from 1 (or a few) nodes up to $N$ final points.

