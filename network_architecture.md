# 1. Forward Direction (Auto-Encoder)

Given a real point cloud $\mathbf{X} \in \mathbb{R}^{N \times 3}$, our goal in the forward direction is to **reconstruct** $ \mathbf{X} $. This direction is basically an **auto-encoder**:

**Encoder** \( E(\cdot) \):
$$
\mathbf{z} = E(\mathbf{X}) \in \mathbb{R}^d,
$$

where $d$ is the latent (embedding) dimension, e.g., $d=96$. This is done by the **En–Di** module in “encoder” mode.

**Decoder** \( \text{De}(\cdot) \):
$$
\widehat{\mathbf{X}} = \text{De}(\mathbf{z}) \in \mathbb{R}^{N \times 3}.
$$

This is the **De–Ge** (TreeGCN) network in “decoder” mode, expanding the latent code $\mathbf{z}$ back to $N$ 3D points.

### Reconstruction Loss $L_{\text{AE}}$
We typically measure how close $\widehat{\mathbf{X}}$ is to $\mathbf{X}$. Common choices are **Chamfer Distance** or **Earth Mover’s Distance (EMD)**. For instance, if we use Chamfer Distance:

$$
d_{\text{Chamfer}}(\mathbf{X}, \widehat{\mathbf{X}}) \;=\; 
\sum_{\mathbf{x} \in \mathbf{X}} \min_{\hat{\mathbf{x}} \in \widehat{\mathbf{X}}} \|\mathbf{x} - \hat{\mathbf{x}}\|_2
\;+\;
\sum_{\hat{\mathbf{x}} \in \widehat{\mathbf{X}}} \min_{\mathbf{x} \in \mathbf{X}} \|\hat{\mathbf{x}} - \mathbf{x}\|_2,
$$

or if we use EMD:

$$
d_{\text{EMD}}(\mathbf{X}, \widehat{\mathbf{X}}) \;=\; 
\min_{\phi: \widehat{\mathbf{X}} \to \mathbf{X}}
\sum_{\hat{\mathbf{x}} \in \widehat{\mathbf{X}}} \|\hat{\mathbf{x}} - \phi(\hat{\mathbf{x}})\|_2.
$$

Then, the **auto-encoder loss** for the forward direction is:

$$
L_{\text{AE}} \;=\;
\mathbb{E}_{\mathbf{X} \in \text{data}}\big[d(\mathbf{X}, \widehat{\mathbf{X}})\big],
$$

where $d(\cdot)$ is our chosen distance (Chamfer or EMD).

---

# 2. Reverse Direction (Generative Adversarial Network)

In the reverse direction, we train the **same** modules as a **GAN**:

**Generator** \( G(\cdot) \):
$$
\widetilde{\mathbf{X}} = G(\mathbf{z}), \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}).
$$

Here, the **De–Ge** (TreeGCN) module runs in “generator” mode, mapping random latent vectors $\mathbf{z}$ to **fake** 3D point clouds $\widetilde{\mathbf{X}}$.

**Discriminator** \( D(\cdot) \):
For any point cloud (real or fake), the **En–Di** network in “discriminator” mode outputs a scalar:
$$
D(\mathbf{X}) \quad \text{or} \quad D(\widetilde{\mathbf{X}}) \quad \in \mathbb{R}.
$$

### Wasserstein GAN Loss (with Gradient Penalty)

We adopt a standard **WGAN-GP** objective. For a batch of real data $\mathbf{X}$ and fake data $\widetilde{\mathbf{X}}$:

- **Discriminator Loss**:
  
 $$
L_{D} \;=\;
\mathbb{E}\big[D(\widetilde{\mathbf{X}})\big]
\;-\;
\mathbb{E}\big[D(\mathbf{X})\big]
\;+\;
\lambda_{\text{gp}}\;L_{\text{gp}},
$$
where
$$
L_{\text{gp}}
\;=\;
\mathbb{E}_{\hat{\mathbf{x}} \in \text{interp}} \Big[
\big(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1\big)^2
\Big],
$$
is the gradient penalty term enforcing $D$ to be **1-Lipschitz**, and $\hat{\mathbf{x}}$ is a linear interpolation of real and fake points.

- **Generator Loss**:
  $$
  L_{G} \;=\; -\,\mathbb{E}\big[D(\widetilde{\mathbf{X}})\big].
  $$
  The generator attempts to **maximize** $D(\widetilde{\mathbf{X}})$.

#### Optional NNME Loss

Additionally, to encourage **uniform** point distributions, we can include a “Nearest Neighbor Mutual Exclusion” penalty, $L_{\text{NNME}}$. A simplified version is:

$$
L_{\text{NNME}}(\widetilde{\mathbf{X}})
  \;=\;
  \mathrm{Var}\!\Big(
    \min_{\mathbf{x}' \in \widetilde{\mathbf{X}}} \|\mathbf{x} - \mathbf{x}'\|
  \Big),
$$

across all pairs or subsets of pairs in $\widetilde{\mathbf{X}}$. Then the generator’s total objective might be:

$$
L_{G}^{\text{total}}
  \;=\;
  L_G
  \;+\;
  \lambda_{\text{NNME}}\,
  L_{\text{NNME}}(\widetilde{\mathbf{X}}).
$$

Hence, for the **GAN direction**, we have a **2-player minimax** problem:

$$
\min_{G} \max_{D} 
  \quad
  \mathbb{E}_{\mathbf{X}\in \text{data}}[D(\mathbf{X})]
  \;-\;
  \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))] 
  \;+\; 
  (\text{gradient penalty} + \text{NNME, etc.}).
$$

---

# 3. TreeGCN Expansion (Decoder/Generator Math)

Inside the **Decoder–Generator** (“De–Ge”) we have **TreeGCN** layers that progressively “branch” from a small number of latent nodes to a larger set of nodes. A simplified notation might be:

- Start with $\mathbf{z} \in \mathbb{R}^{1 \times d}$ (one root node per shape).  
- At layer $\ell$ (where $\ell = 1,\dots,L$), we have $\mathbf{P}^{(\ell-1)} \in \mathbb{R}^{n_{\ell-1} \times c_{\ell-1}}$ describing the current set of nodes. We apply a “branching + transform” step:

  $$
    \mathbf{P}^{(\ell)} = f_{\text{TreeGCN}}^{(\ell)}(\mathbf{P}^{(\ell-1)}),
  $$

  which **upsamples** from $n_{\ell-1}$ nodes to $n_{\ell} = n_{\ell-1} \times \text{degree}_{\ell}$ nodes.  
- Eventually, $\mathbf{P}^{(L)}$ yields $\mathbf{X} \in \mathbb{R}^{N \times 3}$ (the final point cloud).

Concretely, if each TreeGCN layer does something like:

$$
\mathbf{P}^{(\ell)}
  =
  \text{Branch}(\mathbf{P}^{(\ell-1)})
  \;+\;
  \text{Loop}(\mathbf{P}^{(\ell-1)})
  \;+\;
  \text{BiasTerm},
$$

and each module is a small MLP or linear transform (like $\mathbf{W}_{\text{root}}, \mathbf{W}_{\text{branch}}, \mathbf{W}_{\text{loop}}$), then you get a **hierarchical expansion** from a single latent node to a large set of 3D coordinates.

---

# 4. Putting It All Together: BI-Net Objective

BI-Net combines:

- **Auto-Encoder objective**:
  $$
    L_{\text{AE}}
      \;=\;
      \mathbb{E}_{\mathbf{X}}\big[
        d(\mathbf{X}, \text{De}( E(\mathbf{X}) ))
      \big],
  $$
  e.g. $d = \text{EMD} \;\text{or}\; \text{Chamfer}.$

- **GAN objective** (WGAN-GP + optional NNME):
  $$
    \min_{G} \max_{D} 
      \quad
      \mathbb{E}_{\mathbf{X}}[D(\mathbf{X})]
      \;-\; 
      \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))] 
      \;+\;
      \lambda_{\text{gp}}\;L_{\text{gp}}
      \;+\;
      \lambda_{\text{NNME}}\;L_{\text{NNME}}.
  $$

In practice, training proceeds in **two steps** each iteration:

1. **Forward direction (Auto-Encoder step)**:
   - Freeze or partially freeze the “discriminator” function.
   - Encode real $\mathbf{X}$ into $\mathbf{z}$, decode to $\widehat{\mathbf{X}}$.
   - Update parameters to minimize $L_{\text{AE}}$.

2. **Reverse direction (GAN step)**:
   - Freeze or partially freeze the “encoder” function.
   - Generate $\widetilde{\mathbf{X}} = G(\mathbf{z})$ from noise $\mathbf{z}$.
   - Discriminate real vs. fake.
   - Update parameters to minimize or maximize the WGAN objective accordingly.

Because the **encoder** shares parameters with the **discriminator** (they are the same module “En–Di” in different modes), the AE step “guides” part of the network to learn useful geometry features, while the GAN step forces the generator to create realistic shapes that fool the discriminator. This **collaborative training** often stabilizes learning and improves quality with relatively few real shapes.

**Summary of the Math**  

- **Forward (AE) direction**:  
  $$
    \mathbf{z} = E(\mathbf{X}), \quad \widehat{\mathbf{X}} = \text{De}(\mathbf{z}), \quad L_{\text{AE}} = d(\widehat{\mathbf{X}}, \mathbf{X}).
  $$

- **Reverse (GAN) direction**:  
  $$
    \widetilde{\mathbf{X}} = G(\mathbf{z}), \quad \text{GAN Loss} = L_D(\mathbf{X}, \widetilde{\mathbf{X}}) + L_G(\widetilde{\mathbf{X}}),
  $$
  with **WGAN-GP** (and optional **NNME**).

- **TreeGCN**: a **hierarchical expansion** from a small latent code (root node) to a large set of 3D points via repeated “branching” transforms.

This covers the **mathematical essence** of the **BI-Net** architecture, combining an **Auto-Encoder** and **GAN** within the **same** set of parameters for 3D point-cloud generation.

