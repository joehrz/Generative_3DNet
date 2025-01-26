# BI‐Net Overview

**BI‐Net** is a **bidirectional network** that unifies:

1. An **auto‐encoder (AE) direction**: real point clouds \(\mathbf{X} \to \mathbf{z} \to \widehat{\mathbf{X}}\).  
2. A **GAN direction**: random noise \(\mathbf{z} \to \widetilde{\mathbf{X}}\), plus a discriminator to distinguish real vs. fake.

The **decoder** in both directions is the same **TreeGCN** module. The **encoder** and **discriminator** can share parameters or be closely related modules.

---

## 1. Forward Direction (Auto‐Encoder)

Given a real point cloud \(\mathbf{X} \in \mathbb{R}^{N \times 3}\), the AE direction attempts to **reconstruct** \(\mathbf{X}\). This forms an **auto‐encoder** path.

### Encoder

We define an encoder \(E(\cdot)\):

\[
\mathbf{z} = E(\mathbf{X}) 
\quad 
\in 
\mathbb{R}^d,
\]

where \(d\) is the latent dimension. This might be implemented via point‐cloud convolutions or an MLP.

### Decoder

Next, a decoder \(\text{De}(\cdot)\) reconstructs the point cloud:

\[
\widehat{\mathbf{X}} = \text{De}(\mathbf{z})
\quad
\in
\mathbb{R}^{N \times 3}.
\]

Typically, this **TreeGCN** expands \(\mathbf{z}\) from a small node set to \(N\) final 3D points.

### Reconstruction Loss

We measure how close the reconstruction \(\widehat{\mathbf{X}}\) is to the real \(\mathbf{X}\). Common choices:

- **Chamfer Distance**:

  \[
  d_{\mathrm{Chamfer}}(\mathbf{X}, \widehat{\mathbf{X}})
  =
  \sum_{\mathbf{x}\,\in\,\mathbf{X}}
    \min_{\hat{\mathbf{x}}\,\in\,\widehat{\mathbf{X}}}
    \|\mathbf{x} - \hat{\mathbf{x}}\|_2
  \;+\;
  \sum_{\hat{\mathbf{x}}\,\in\,\widehat{\mathbf{X}}}
    \min_{\mathbf{x}\,\in\,\mathbf{X}}
    \|\hat{\mathbf{x}} - \mathbf{x}\|_2.
  \]

- **EMD** (Earth Mover’s Distance).  

Hence, the **AE loss** is:

\[
L_{\mathrm{AE}}
=
\mathbb{E}_{\mathbf{X}}\bigl[
  d(\mathbf{X}, \widehat{\mathbf{X}})
\bigr].
\]

where \(d(\cdot)\) could be **Chamfer** or **EMD**.

---

## 2. Reverse Direction (GAN)

In the **reverse** direction, BI‐Net uses the **same decoder** as a **generator**, plus a **discriminator** that can share parameters with the encoder.

### Generator

A random latent vector \(\mathbf{z} \sim \mathcal{N}(0,\mathbf{I})\) is mapped to fake point clouds:

\[
\widetilde{\mathbf{X}} = G(\mathbf{z}).
\]

The module \(G(\cdot)\) can be the same **TreeGCN** used by the decoder.

### Discriminator

For real or fake point clouds, a discriminator \(D(\cdot)\) outputs a scalar score. In BI‐Net, this might share layers with the encoder.

---

#### WGAN‐GP Objective

We employ **Wasserstein GAN** with Gradient Penalty:

- **Discriminator loss**:

  \[
  L_{D}
  =
  \mathbb{E}\bigl[D(\widetilde{\mathbf{X}})\bigr]
  \;-\;
  \mathbb{E}\bigl[D(\mathbf{X})\bigr]
  \;+\;
  \lambda_{\mathrm{gp}}\;L_{\mathrm{gp}},
  \]

  where

  \[
  L_{\mathrm{gp}}
  =
  \mathbb{E}_{\hat{\mathbf{x}}\in\mathrm{interp}}
  \Bigl[
    \bigl(\|\nabla_{\hat{\mathbf{x}}}\,D(\hat{\mathbf{x}})\|_2 - 1\bigr)^2
  \Bigr].
  \]

- **Generator loss**:

  \[
  L_{G} 
  = 
  -\,\mathbb{E}\bigl[D(\widetilde{\mathbf{X}})\bigr].
  \]

##### Optional Uniformity (NNME) Loss

We can add a **Nearest Neighbor Mutual Exclusion (NNME)** term to encourage uniform point placement:

\[
L_{\mathrm{NNME}}\bigl(\widetilde{\mathbf{X}}\bigr),
\]
which penalizes highly clustered points. Then the generator’s total cost is:

\[
L_{G}^{\mathrm{total}}
=
L_{G}
+
\lambda_{\mathrm{NNME}}
\,L_{\mathrm{NNME}}\bigl(\widetilde{\mathbf{X}}\bigr).
\]

---

## 3. BI‐Net Training Strategy

Each iteration has **two phases**:

1. **AE direction**  
   - Encode real \(\mathbf{X} \to \mathbf{z}\), then decode \(\widehat{\mathbf{X}}=\text{De}(\mathbf{z})\).  
   - Minimize \(d(\mathbf{X},\widehat{\mathbf{X}})\).

2. **GAN direction**  
   - Generate \(\widetilde{\mathbf{X}}=G(\mathbf{z})\) from random noise.  
   - Update \(\max D\), \(\min G\) via WGAN‐GP (+ optional NNME).  

Because the encoder and discriminator can share parameters, the **AE** training helps the network learn better geometric features, while the **GAN** training forces the generator to produce more realistic shapes. This **collaborative** approach stabilizes learning and often yields higher‐quality point clouds.

---

## TreeGCN Pipeline

Given \(\mathbf{X}^{(l)} \in \mathbb{R}^{B \times N \times d_{\mathrm{in}}}\), a **single** TreeGCN layer outputs \(\mathbf{X}^{(l+1)} \in \mathbb{R}^{B \times (N \cdot \mathrm{degree}) \times d_{\mathrm{out}}}\).  
Let:

- \(\mathbf{W}_{\mathrm{root}} \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}\),  
- \(\mathbf{W}_{\mathrm{branch}} \in \mathbb{R}^{d_{\mathrm{in}} \times (d_{\mathrm{in}} \cdot \mathrm{degree})}\),  
- \(\mathbf{W}_{\mathrm{loop}}\) (MLP from \(d_{\mathrm{in}}\) to \(d_{\mathrm{out}}\)),  
- \(\mathbf{b} \in \mathbb{R}^{\mathrm{degree} \times d_{\mathrm{out}}}\).

For each node \(\mathbf{x}_{b,n} \in \mathbb{R}^{d_{\mathrm{in}}}\):

1. **Root transform**  
   \[
   \mathbf{r}_{b,n}
   =
   \mathbf{x}_{b,n}\,\mathbf{W}_{\mathrm{root}}
   \;\;\in\;\;\mathbb{R}^{d_{\mathrm{out}}}.
   \]

2. **Branch transform**  
   \[
   \mathbf{z}_{b,n}
   =
   \mathbf{x}_{b,n}\,\mathbf{W}_{\mathrm{branch}}
   \;\;\in\;\;\mathbb{R}^{d_{\mathrm{in}}\cdot\mathrm{degree}}.
   \]
   Reshape into \(\mathbf{z}_{b,n}^{(k)} \in \mathbb{R}^{d_{\mathrm{in}}}\), \(k=1,\dots,\mathrm{degree}\).

3. **Loop transform** each child  
   \[
   \hat{\mathbf{z}}_{b,n}^{(k)}
   =
   \mathbf{W}_{\mathrm{loop}}\bigl(\mathbf{z}_{b,n}^{(k)}\bigr)
   \;\;\in\;\;\mathbb{R}^{d_{\mathrm{out}}}.
   \]

4. **Combine root + branch**  
   \[
   \mathbf{y}_{b,n}^{(k)}
   =
   \mathbf{r}_{b,n}
   +
   \hat{\mathbf{z}}_{b,n}^{(k)}
   +
   \mathbf{b}^{(k)}.
   \]
   Then apply an activation (e.g., LeakyReLU).

Hence each parent node \((b,n)\) spawns \(\mathrm{degree}\) children. Stacking multiple layers grows from 1 (or a few) nodes up to \(N\) final points.

---

