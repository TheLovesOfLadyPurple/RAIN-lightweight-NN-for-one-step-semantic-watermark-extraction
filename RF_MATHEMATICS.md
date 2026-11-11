# Mathematical Foundations of Extended Rectified Flow and RAIN

> **Companion PDFs Included in This Repository:**
> - **Main Paper:** [RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction (`RAIN__Region_Aware_Inversion_Network_for_Semantic_Watermark_Extraction.pdf`)](RAIN__Region_Aware_Inversion_Network_for_Semantic_Watermark_Extraction.pdf)
> - **Mathematical Proof & Distillation Note:** [The Reason Why Some Distillation Models Could Not Use DDIMInverse (`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`)](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf) — *We also include this companion PDF directly in the repository for readers who want the complete, self-contained Test-Function deduction of the ground-truth diffusion / Rectified Flow velocity target (without requiring Fokker–Planck PDEs) and the proof of why distillation models (CM, DMD, LCM) cannot apply DDIM or DDIM Inverse.*
>
> This guide synthesizes both documents to explain the mathematical elegance behind **Extended Rectified Flow (RF)** and why a tiny, one-step regression network (**14.394 GFLOPs**, **47×** smaller than diffusion-scale extractors) suffices to extract Gaussian Shading watermarks without reconstructing a multi-step inverse trajectory:
> 1. **Why** does Rectified Flow predict the **conditional mean** of velocity — and why is that algebraically identical to predicting the conditional mean of noise $\varepsilon$ or clean latent $z_0$ ($X_0$)?
> 2. **How can that possibly work?** If a network only predicts *averages* over crossing trajectories, why does integrating that deterministic velocity field still transport the exact probability distribution? *(Proved below using the exact Test-Function + multivariable Taylor deduction from [`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf).)*
> 3. **Why does this unlock one-step watermark extraction** even though pure regression blurs one-step *image generation*?

---

## 1. From DDPM Corruption to Extended Flow Matching & Rectified Flow

### 1.1 Interpolation View of Diffusion

In Denoising Diffusion Probabilistic Models (DDPM), clean data $X_0 \sim p_{\text{data}}$ (often denoted $z_0$ in latent diffusion) is progressively corrupted through a Gaussian Markov chain $q(X_t \mid X_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\,X_{t-1}, \beta_t I)$. Closed-form Gaussian marginalization yields:

$$
X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I), \quad \bar{\alpha}_t = \prod_{i=1}^{t}(1 - \beta_i).
$$

Rather than viewing $X_t$ merely as stochastic corruption, we can interpret every noisy sample as a point along a continuous-time **interpolation trajectory** between an image-side endpoint $X_0$ and a Gaussian-noise endpoint $\varepsilon$.

More generally, let $(X_0, \varepsilon) \sim \pi_{0,\varepsilon}$ be **any coupling** (joint distribution) whose marginals are the clean latent distribution and standard normal noise:
- **Independent coupling (DDPM / RF-1):** $X_0 \sim p_{\text{data}}$, $\varepsilon \sim \mathcal{N}(0, I)$, with $X_0 \perp \varepsilon$.
- **Teacher-induced coupling (Reflow / RF-2):** $(X_0, \varepsilon)$ is deterministically or stochastically paired by running a pretrained diffusion sampler from $\varepsilon$ to $X_0$.

For differentiable schedules $a_t, b_t$ on $t \in [0, 1]$ (with clean-to-noise boundary convention $a_0 \approx 1, b_0 \approx 0$ and $a_1 \approx 0, b_1 \approx 1$), define the sample-wise interpolation:

$$
X_t = \varphi_t(X_0, \varepsilon) = a_t X_0 + b_t \varepsilon.
$$

- **Standard linear Rectified Flow:** $a_t = 1 - t,\; b_t = t$.
- **DDPM / Variance-Preserving (VP) schedule:** $a_t = \sqrt{\bar{\alpha}_t},\; b_t = \sqrt{1 - \bar{\alpha}_t}$.

*(Note: Reversing the time index so $t=0$ is noise and $t=1$ is clean data simply swaps $(a_t, b_t)$ and negates time derivatives; every identity below remains unchanged.)*

### 1.2 Why Sample Trajectories Are Non-Causal

Differentiating $\varphi_t(X_0, \varepsilon)$ with respect to $t$ gives the instantaneous velocity of a specific pair $(X_0, \varepsilon)$:

$$
U_t = \frac{\partial \varphi_t(X_0, \varepsilon)}{\partial t} = \dot{a}_t X_0 + \dot{b}_t \varepsilon,
$$

where $\dot{a}_t = \frac{da_t}{dt}$ and $\dot{b}_t = \frac{db_t}{dt}$.

During training, both endpoints $(X_0, \varepsilon)$ are known, so $(X_t, U_t)$ can be computed directly. **During inference, however, $U_t$ is non-causal:** we only observe the current state $X_t = x$ at time $t$, not the hidden pair $(X_0, \varepsilon)$ that produced it. Because Gaussian noise occupies the full ambient space $\mathbb{R}^{c \times h \times w}$ while encoded images concentrate near a much lower-dimensional manifold — and because non-optimal couplings allow distinct pairs $(X_0^{(1)}, \varepsilon^{(1)}) \neq (X_0^{(2)}, \varepsilon^{(2)})$ to cross at the same point $X_t = x$ — $U_t$ is not uniquely determined by $(X_t, t)$ alone. Thus, $\dot{X}_t = U_t$ is **not** a closed ODE that can be integrated at test time.

---

## 2. Why RF Predicts Conditional Means: Velocity, $\varepsilon$, and $z_0$ ($X_0$)

### 2.1 Optimal Regression Learns the Conditional Expectation

Flow Matching and Rectified Flow turn the non-causal velocity $U_t$ into a deterministic, observable vector field $v(x, t)$ by minimizing the squared $L_2$ regression loss:

$$
\min_{v} \; \mathbb{E}_{(X_0, \varepsilon) \sim \pi_{0,\varepsilon}} \left[ \left\| v(X_t, t) - U_t \right\|_2^2 \right].
$$

Conditioning on $X_t = x$ and adding/subtracting the conditional expectation $\mathbb{E}[U_t \mid X_t = x]$ yields the standard orthogonal decomposition:

$$
\mathbb{E}\!\left[ \| v(x, t) - U_t \|_2^2 \;\middle|\; X_t = x \right]
= \left\| v(x, t) - \mathbb{E}[U_t \mid X_t = x] \right\|_2^2
+ \mathbb{E}\!\left[ \left\| U_t - \mathbb{E}[U_t \mid X_t = x] \right\|_2^2 \;\middle|\; X_t = x \right],
$$

because the cross-term vanishes: $\mathbb{E}[U_t - \mathbb{E}[U_t \mid X_t = x] \mid X_t = x] = 0$. Since the second term (conditional variance) is independent of $v$, the unique population-optimal velocity field is the **conditional mean**:

$$
v^*(x, t) = \mathbb{E}\!\left[ U_t \mid X_t = x \right] = \dot{a}_t\,\mathbb{E}[X_0 \mid X_t = x] + \dot{b}_t\,\mathbb{E}[\varepsilon \mid X_t = x].
$$

### 2.2 Exact Equivalence of Predicting Mean Velocity, Mean $\varepsilon$, and Mean $z_0$ ($X_0$)

A remarkably clean consequence of linear interpolation is that **velocity prediction ($v$), noise prediction ($\varepsilon$), and clean-latent prediction ($z_0 / X_0$) all estimate the exact same conditional expectation**.

Taking the conditional expectation of the constraint $X_t = a_t X_0 + b_t \varepsilon$ given $X_t = x$ gives the deterministic link between the two endpoint means:

$$
x = a_t\,\mathbb{E}[X_0 \mid X_t = x] + b_t\,\mathbb{E}[\varepsilon \mid X_t = x].
$$

1. **Eliminating $X_0$ (Noise / $\varepsilon$-prediction form — Eq. 15 in the RAIN paper):**
   Substituting $X_0 = \frac{X_t - b_t \varepsilon}{a_t}$ (for $a_t \neq 0$) into $U_t = \dot{a}_t X_0 + \dot{b}_t \varepsilon$ gives:

   $$
   U_t = \frac{\dot{a}_t}{a_t} X_t + \left( \dot{b}_t - \frac{\dot{a}_t b_t}{a_t} \right) \varepsilon
   \quad\Longrightarrow\quad
   v^*(x, t) = \frac{\dot{a}_t}{a_t} x + \left( \dot{b}_t - \frac{\dot{a}_t b_t}{a_t} \right) \mathbb{E}[\varepsilon \mid X_t = x].
   $$

2. **Eliminating $\varepsilon$ (Clean-endpoint / $z_0$-prediction form):**
   Substituting $\varepsilon = \frac{X_t - a_t X_0}{b_t}$ (for $b_t \neq 0$) into $U_t = \dot{a}_t X_0 + \dot{b}_t \varepsilon$ gives:

   $$
   U_t = \frac{\dot{b}_t}{b_t} X_t + \left( \dot{a}_t - \frac{\dot{b}_t a_t}{b_t} \right) X_0
   \quad\Longrightarrow\quad
   v^*(x, t) = \frac{\dot{b}_t}{b_t} x + \left( \dot{a}_t - \frac{\dot{b}_t a_t}{b_t} \right) \mathbb{E}[X_0 \mid X_t = x].
   $$

3. **Concrete intuition on Standard Linear Rectified Flow ($a_t = 1 - t,\; b_t = t,\; \dot{a}_t = -1,\; \dot{b}_t = 1$):**

   $$
   v^*(x, t) = \mathbb{E}[\varepsilon - X_0 \mid X_t = x]
   = \frac{\mathbb{E}[\varepsilon \mid X_t = x] - x}{1 - t}
   = \frac{x - \mathbb{E}[X_0 \mid X_t = x]}{t}.
   $$

> [!IMPORTANT]
> **Unified Takeaway (and Connection to Tweedie's Formula):**
> At any observable state $(x, t)$, knowing **any one** of the three quantities — the mean velocity $v^*(x, t)$, the mean noise $\mathbb{E}[\varepsilon \mid X_t = x]$, or the mean clean latent $\mathbb{E}[X_0 \mid X_t = x]$ ($\mathbb{E}[z_0 \mid z_t]$) — algebraically determines the other two in closed form.
>
> Under the special case of independent Gaussian corruption ($X_0 \perp \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$), $X_t \mid X_0 \sim \mathcal{N}(a_t X_0, b_t^2 I)$, and **Tweedie's formula** further identifies this conditional mean with the score function:
> $\mathbb{E}[X_0 \mid X_t = x] = \frac{x + b_t^2 \nabla_x \log p_t(x)}{a_t}$ and $\mathbb{E}[\varepsilon \mid X_t = x] = -b_t \nabla_x \log p_t(x)$.
>
> Crucially, however, the Extended Rectified Flow identities above **do not require $X_0 \perp \varepsilon$** — they remain exact even when $(X_0, \varepsilon)$ is a deterministic teacher coupling (RF-2)!

---

## 3. How Can Predicting the *Mean* Work? (Test-Function Proof Without Fokker–Planck)

At first glance, replacing a sample's velocity with its conditional expectation seems paradoxical:
- Whenever multiple trajectories pass through the same state, averaging their velocities points in none of the individual directions!
- Worse still, averaging random vectors shrinks variance. Why doesn't integrating the deterministic causal velocity field collapse to a blurry mean instead of reproducing the target distribution?

Below we present the exact **Test-Function Method** deduction from [`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf) (building on Jianlin Su's elementary derivation), which requires no complex physics such as the Fokker–Planck formula — only treating functions of vectors as ordinary multivariable functions and applying the first-order Taylor formula.

### 3.1 Deduction via the Test Function Method

*(Notation matching [`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf): here $X_1$ denotes the target image and $X_0 \sim \mathcal{N}(0, I)$ denotes pure Gaussian noise, moving from noise at $t = 0$ to image at $t = 1$ with $\alpha_1 = 1, \beta_1 = 0$.)*

For an arbitrary interpolation between image $X_1$ and noise $X_0$, we have the non-causal training trajectory:

$$
X_t = \varphi_t(X_1, X_0) = \alpha_t X_1 + \beta_t X_0.
$$

Eliminating $X_0 = \frac{X_t - \alpha_t X_1}{\beta_t}$, the conditional velocity conditioned on image $X_1$ is:

$$
u^{\text{cond}}(X_t, X_1, t) = \frac{\partial \varphi_t}{\partial t} = \frac{-\dot{\beta}_t \alpha_t X_1 + \dot{\beta}_t X_t + \dot{\alpha}_t \beta_t X_1}{\beta_t},
$$

where $\dot{\alpha}_t$ and $\dot{\beta}_t$ are the time derivatives of $\alpha_t$ and $\beta_t$ with respect to $t$.

Although $u^{\text{cond}}(X_t, X_1, t)$ transports the distribution along $\varphi_t$, it is **non-causal** and can only be used in training because it requires the future target $X_1$. During inference, a neural network can only approximate the **ground-truth causal velocity field**:

$$
u^*(x_t, t) = \mathbb{E}_{x_1 \mid x_t}\!\left[u^{\text{cond}}(x_t, x_1, t)\right].
$$

Explicitly, during inference we evolve $Z_t$ via:

$$
dZ_t = \mathbb{E}_{x_1 \mid x_t}\!\left[u^{\text{cond}}(X_t, X_1, t)\right] dt,
$$

which transforms $p_t(X_t)$ to $q_{t + \Delta t}(Z_{t + \Delta t})$ via the discrete step:

$$
Z_{t + \Delta t} = X_t + \mathbb{E}_{x_1 \mid x_t}\!\left[u^{\text{cond}}(X_t, X_1, t)\right] \Delta t.
$$

*(Here $Z_t$ explicitly denotes the state produced by causal neural-network inference, whereas $X_t$ is produced by non-causal interpolation between $X_1$ and $X_0$.)*

**Theorem 1 (Test Function Method).** *If for any test function $\phi(x)$, the following equation holds, then $f(x) = g(x)$:*

$$
\int f(x)\phi(x)\,dx = \int g(x)\phi(x)\,dx.
$$

Thus, to prove $q_{t + \Delta t} = p_{t + \Delta t}$, we need to show that for every test function $\phi$:

$$
\int p_{t + \Delta t}(X_{t + \Delta t})\phi(X_{t + \Delta t})\,dX_{t + \Delta t} = \int q_{t + \Delta t}(Z_{t + \Delta t})\phi(Z_{t + \Delta t})\,dZ_{t + \Delta t}.
$$

Because the ground-truth velocity field $u^*(X_t, t)$ is **deterministic** for each given $X_t$, on the right-hand side we can avoid integrating over $Z_{t + \Delta t} \sim q_{t + \Delta t}(Z_{t + \Delta t})$ and instead integrate directly over $X_t \sim p_t(X_t)$:

$$
\int p_{t + \Delta t}(X_{t + \Delta t})\phi(X_{t + \Delta t})\,dX_{t + \Delta t} = \int p_t(X_t)\phi(Z_{t + \Delta t})\,dX_t,
$$

which rewrites directly as our target identity:

$$
\int p_{t + \Delta t}(x_{t + \Delta t})\phi(x_{t + \Delta t})\,dx_{t + \Delta t} = \int p_t(x_t)\phi\!\left(x_t + u^*(x_t, t)\Delta t\right) dx_t.
$$

**Proof:** Expanding $\mathbb{E}_{X_{t + \Delta t}}[\phi(X_{t + \Delta t})]$ to first order in $\Delta t$ via the multivariable Taylor formula:

$$
\begin{aligned}
\mathbb{E}_{X_{t + \Delta t}}\!\left[\phi(X_{t + \Delta t})\right]
&= \mathbb{E}_{X_0, X_1}\!\left[\phi(\varphi_{t + \Delta t})\right] \\
&= \mathbb{E}_{X_0, X_1}\!\left[\phi(\varphi_t) + \Delta t\,\frac{\partial \varphi_t}{\partial t} \cdot \nabla_{\varphi_t}\phi(\varphi_t)\right] \\
&= \mathbb{E}_{X_t}\!\left[\phi(X_t)\right] + \Delta t\,\mathbb{E}_{X_0, X_1}\!\left[\frac{\partial \varphi_t}{\partial t} \cdot \nabla_{X_t}\phi(X_t)\right],
\end{aligned}
$$

where $\varphi_{t + \Delta t}(X_1, X_0) \in \mathbb{R}^d$, `·` denotes the vector inner product, $X_t$ substitutes $\varphi_t$ once $(X_1, X_0)$ is given, and the gradient operator is:

$$
\nabla_{\varphi_t} = \left[\frac{\partial}{\partial X_t^0}, \frac{\partial}{\partial X_t^1}, \dots, \frac{\partial}{\partial X_t^d}\right]^T, \qquad
\nabla_{\varphi_t}\phi(\varphi_t) = \left[\frac{\partial \phi(\varphi_t)}{\partial X_t^0}, \frac{\partial \phi(\varphi_t)}{\partial X_t^1}, \dots, \frac{\partial \phi(\varphi_t)}{\partial X_t^d}\right]^T.
$$

When $X_1$ in $u^{\text{cond}}(X_t, X_1, t)$ can be deduced given $(X_0, X_t)$, conditioning on $X_t$ and re-applying the first-order Taylor formula yields:

$$
\begin{aligned}
\mathbb{E}_{X_{t + \Delta t}}\!\left[\phi(X_{t + \Delta t})\right]
&= \mathbb{E}_{X_t}\!\left[\phi(X_t)\right] + \Delta t\,\mathbb{E}_{X_0, X_t}\!\left[\frac{\partial \varphi_t}{\partial t} \cdot \nabla_{X_t}\phi(X_t)\right] \\
&= \mathbb{E}_{X_t}\!\left[\phi(X_t)\right] + \Delta t\,\mathbb{E}_{X_t}\!\left[\mathbb{E}_{X_1 \mid X_t}\!\left[u^{\text{cond}}(X_t, X_1, t)\right] \cdot \nabla_{X_t}\phi(X_t)\right] \\
&= \mathbb{E}_{X_t}\!\left[\phi\!\left(X_t + \Delta t\,\mathbb{E}_{X_1 \mid X_t}\!\left[u^{\text{cond}}(X_t, X_1, t)\right]\right)\right] \\
&= \mathbb{E}_{X_t}\!\left[\phi\!\left(X_t + \Delta t\,u^*(X_t, t)\right)\right].
\end{aligned}
$$

Therefore, for **every** test function $\phi(x)$, **arbitrary** endpoint distributions $p_0, p_1$, and **any** schedule $(\alpha_t, \beta_t)$ with $\alpha_1 = 1, \beta_1 = 0$ — **without any independence assumption on the coupling $p(X_1 \mid X_0)$** — adding $\Delta t\,u^*(X_t, t)$ reproduces the exact target distribution of the non-causal path $\varphi_t$ at every step starting from pure Gaussian noise $Z_0 \sim \mathcal{N}(0, I)$. $\blacksquare$

### 3.2 Geometric Intuition: Local Uncrossing vs. Single-Step Norm Shrinkage

Why does this work so cleanly, and what changes when we take **one step** instead of integrating the ODE?

1. **Continuous ODE integration locally "uncrosses" trajectories without losing mass:**
   In the Test-Function deduction above, conservation of probability mass at point $x_t$ only depends on the **net probability flux** through $x_t$, which equals density times the *conditional mean velocity* $u^*(x_t, t) = \mathbb{E}_{X_1 \mid X_t = x_t}[u^{\text{cond}}(x_t, X_1, t)]$. When two training paths cross at $(x_t, t)$, deterministic ODE trajectories cannot cross each other (by uniqueness of ODE solutions); instead, particles arriving at $x_t$ smoothly *re-route* along the average velocity vector, swapping partners so that the overall marginal distribution $p_t$ stays completely intact!

2. **Arbitrary couplings and Reflow (RF-1 $\to$ RF-2 $\to \cdots$):**
   Notice that the proof never assumed $X_0 \perp \varepsilon$. If we first train a diffusion/RF-1 model on independent pairs $X_0 \perp \varepsilon$, integrate its deterministic ODE to obtain coupled pairs $(X_0, \varepsilon) \sim \pi_{\text{teacher}}$, and train a second model (**RF-2**) on $X_t = a_t X_0 + b_t \varepsilon$, the new trajectories cross far less and become much straighter — yet still preserve the exact endpoint marginals!

3. **Why a *single giant jump* shrinks norms (and why DMD2 / LCM differ from DDIM):**
   Although *infinitesimal* integration along $v^*$ preserves marginals, evaluating the conditional expectation in **one single jump** from $X_0$ outputs $\hat{\varepsilon} \approx \mathbb{E}[\varepsilon \mid X_0]$ — a convex weighted average $\sum_i w_i \varepsilon^{(i)}$ with $\sum_i w_i = 1$. For unit-variance Gaussian vectors, preserving the $L_2$ norm requires $\sum_i w_i^2 = 1$, whereas any non-degenerate average has $\sum_i w_i^2 < 1$ and therefore **strictly smaller Euclidean norm**!
   - For **one-step image generation**, landing on a conditional mean blurs high-frequency details off the image manifold — which is why state-of-the-art one-step generators such as **DMD2** explicitly remove the regression loss in favor of distribution-matching objectives (seeking fully straightened "RF-$\infty$" transport), and why distilled models like **LCM / DMD** alter the coupling $p(X_0 \mid \varepsilon)$ so that intermediate ODE states no longer match independent Gaussian corruption (preventing direct use of standard DDIM or DDIM Inverse; see the companion proof note [`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf) for the full derivation).
   - For **semantic watermark extraction**, however, this exact conditional-statistic behavior turns out to be a decisive advantage rather than a flaw!

---

## 4. Why This Makes RAIN Possible: High-SNR RF-2 + Decision-Region Recovery

### 4.1 Gaussian Shading Needs Regions, Not Exact Sample Reconstruction

Gaussian Shading (GS) encrypts repeated watermark bits ($f_c$ channel copies, $f_{hw}^2$ spatial copies) into a uniform bitstream that selects equal-probability quantile intervals of $\mathcal{N}(0, 1)$:

$$
I_i = \left( \Phi^{-1}\!\left(\frac{i}{2^l}\right),\; \Phi^{-1}\!\left(\frac{i + 1}{2^l}\right) \right], \qquad i \in \{0, \dots, 2^l - 1\}.
$$

In the standard $l = 1$ configuration ($f_c = 1, f_{hw} = 8$), the two regions are simply $I_0 = (-\infty, 0]$ and $I_1 = (0, +\infty)$ — **the sign of each coordinate**, followed by stream-cipher decryption and $64\times$ majority voting.

Consequently, extraction **never** requires recovering the exact floating-point sample $\varepsilon$ or matching its $\mathcal{N}(0, I)$ norm; it only requires the predicted statistic $\hat{\varepsilon}$ to stay on the same side of the Gaussian decision boundaries:

$$
\operatorname{GSDec}_K(\hat{\varepsilon}) = \operatorname{GSDec}_K(\varepsilon) \qquad (\text{rather than } \hat{\varepsilon} = \varepsilon).
$$

### 4.2 High-SNR Endpoint Regression Is Dramatically Simpler Than Full Inversion

When training pairs $(X_0, \varepsilon)$ are generated by a diffusion teacher sampler (an **extended RF-2 coupling**), consider the **high-SNR regime** right next to the clean image endpoint ($X_t \approx X_0$):
- Almost the entire noise-to-image transport has already completed, so trajectories in the local neighborhood of $X_0$ have minimal dispersion.
- By Eq. (15), learning the high-SNR velocity $v^*(x, t)$ is equivalent to predicting the **conditional mean $\mathbb{E}[\varepsilon \mid X_0]$** (under $L_2$ loss) or the **conditional median** (under coordinate-wise $L_1$ loss, which RAIN adopts for sharper coordinate recovery) of the terminal noises capable of generating $X_0$.
- Even if multiple noises map to visually indistinguishable images (so even 50-step DDIM inversion returns a noise different from the exact initial sample), their conditional mean/median preserves the encoded Gaussian half-spaces with overwhelming probability!

> [!TIP]
> **Verify This Property Empirically in the Codebase:**
> You can directly observe conditional-mean norm shrinkage in [`coco2017_diffusers_unipc_gaussian_shading_watermark_norm_sdv21.py`](coco2017_diffusers_unipc_gaussian_shading_watermark_norm_sdv21.py). It compares $\|\hat{\varepsilon}\|_2$ recovered by **RAIN**, **FARI**, and **50-step DDIM inversion** against the exact expected Gaussian norm $\mathbb{E}[\|Z\|_2]$ ($Z \sim \mathcal{N}(0, I_{4 \times 64 \times 64})$). Both RAIN and FARI return conditional statistics with smaller $L_2$ norms than DDIM inversion, yet achieve **99.99% clean bit accuracy** and **100.00% detection/traceability**!

### 4.3 Anchor–Residual Parallel Decomposition in RAIN

Because high-SNR conditional statistic estimation does not need to model the full multi-step diffusion trajectory across all SNR regimes, RAIN discards the 865M+ diffusion U-Net, text encoder, and cross-attention altogether. Instead, given an observed (and potentially distorted) latent $\widetilde{X}_0 = D(X_0)$, RAIN decomposes endpoint regression across **two compact, prompt-free width-64 NAFNet branches** (`15.468M` parameters total, **14.394 GFLOPs**):

1. **Image-like Anchor Branch ($F_0$):** Predicts a clean teacher anchor $\widehat{X}_0 = F_0(\widetilde{X}_0)$ supervised against the diffusion teacher's clean prediction $X_0^*(t)$, staying near the low-dimensional image manifold.
2. **Intermediate Noisy Branch ($F_1$) + Lightweight Timestep Head ($F_2$):** Predicts an intermediate noisy state $\widehat{X}_t = F_1(\widetilde{X}_0)$ together with its effective noise level $\hat{t} = F_2(\widehat{X}_t)$, supervised on synthetic interpolations $X_t^* = \sqrt{\bar{\alpha}_t}\,X_0^*(0.9) + \sqrt{1 - \bar{\alpha}_t}\,\varepsilon$.
3. **Algebraic Schedule Recombination & Parallelism:** Combines $\widehat{X}_t$, $\operatorname{SG}(\widehat{X}_0)$, and $\bar{\alpha}_{\hat{t}}$ analytically via the diffusion schedule while running $F_0$ and $F_1$ **concurrently on separate CUDA streams or GPUs**, cutting extraction latency down to **1.392 ms/image** (vs. `37.344 ms` for FARI and `31.320 ms` for OSI under a matched VRAM budget).

---

## Further Reading

- **Full Paper:** [RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction (`RAIN__Region_Aware_Inversion_Network_for_Semantic_Watermark_Extraction.pdf`)](RAIN__Region_Aware_Inversion_Network_for_Semantic_Watermark_Extraction.pdf) — see **Section 2.1** (*From DDPM Corruption to Extended Flow Matching and Rectified Flow*), **Section 3.1–3.2** (*One-Step Recovery as Conditional Regression* & *High-SNR Endpoint Regression*), and **Section 6** (*Further Discussion* on RF-2 vs. DMD2).
- **Companion Mathematical Proof Note:** [The Reason Why Some Distillation Models Could Not Use DDIMInverse (`TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf`)](TheReasonWhySomeDistillationModelCouldNotUseDDIMInverse.pdf) — Zilai Li's self-contained essay presenting the complete Test-Function + first-order Taylor deduction of ground-truth diffusion/RF velocity fields without Fokker–Planck PDEs, why altering the coupling $p(X_1 \mid X_0)$ changes intermediate ODE marginals (RF-2 vs. Consistency Models vs. DMD as "RF-$\infty$"), and why LCM/DMD distillation models require the LCM solver rather than DDIM or DDIM Inverse.
- **Elementary Test-Function Derivation Reference:** Jianlin Su, *"Generating Diffusion Models (Part 17): Constructing ODEs for Diffusion Models — A General Approach"*, [spaces.ac.cn/archives/9497](https://spaces.ac.cn/archives/9497).
