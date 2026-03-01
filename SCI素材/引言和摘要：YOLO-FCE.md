# 1. Introduction

Aquatic environments harbor a growing accumulation of plastic debris, among which transparent and semi-transparent plastic bottles represent one of the most pervasive yet detection-resistant categories of waterborne pollutants. The proliferation of unmanned surface vehicles (USVs) has created urgent demand for real-time, onboard detection systems capable of identifying such targets under highly adverse conditions—including specular surface reflections, dynamic wave-induced deformations, and the near-total absence of texture contrast that renders transparent objects effectively invisible to conventional RGB-based detectors. Addressing this challenge has direct implications for autonomous environmental monitoring, coastal cleanup operations, and ecological preservation.

Despite substantial advances in general-purpose object detection, existing methods exhibit critical failure modes when deployed on open water surfaces. CNN-based detectors such as YOLOv5 [CITE], YOLOv8 [CITE], and YOLO11 [CITE] rely on locally-receptive convolutional operators that excel at extracting high-frequency texture features but are fundamentally ill-suited to reasoning about the global scene context necessary for suppressing water surface clutter. A floating bottle partially submerged beneath sun glare, for instance, may share nearly identical local texture statistics with surrounding specular highlights, making local-window feature matching unreliable. Transformer-based detectors [CITE: DETR, RT-DETR] can, in principle, capture long-range dependencies through self-attention, but their quadratic complexity $\mathcal{O}(N^2)$ with respect to token count renders them computationally prohibitive for high-resolution, real-time edge deployment on resource-constrained USV platforms.

A second, orthogonal limitation concerns the modality of input sensing. Virtually all existing waterborne debris detectors operate exclusively on RGB imagery, implicitly assuming that luminance-based texture is a reliable discriminative cue. This assumption breaks down precisely in the conditions most relevant to environmental monitoring: under direct sunlight, water surfaces exhibit intense specular reflection that saturates RGB channels and masks target boundaries; in turbid or shadowed water, the absence of contrast renders single-modality detection unreliable. Physical optics offers an alternative signal: water surface reflections are highly polarized due to Brewster-angle scattering, while floating debris—particularly plastics—exhibits substantially lower polarization due to diffuse reflection. This physical asymmetry provides an orthogonal, background-suppressive cue that RGB alone cannot encode. Yet, to date, no detection framework has systematically incorporated polarimetric physical priors alongside architectural innovations to jointly address both limitations.

A further methodological gap concerns the opacity of the architecture design process itself. When a detection model fails on a particular scene type, practitioners typically respond by empirically stacking additional layers, increasing channel widths, or substituting attention modules—without principled guidance on _where_ in the feature hierarchy the representational failure originates. This trial-and-error paradigm is inefficient and risks over-parameterization. What is needed is a diagnostic framework capable of localizing feature degradation to specific network layers, thereby enabling targeted, minimal-footprint architectural intervention.

To address all three of these limitations within a unified framework, this paper makes the following contributions:

**(1) A Physically-Informed Heterogeneous Data Strategy.** We move beyond single-modality RGB input and interpret raw polarimetric sensor data as $K=6$ heterogeneous physical views—RGB, Monochromatic (Mono), Degree of Linear Polarization (DoLP), Angle of Polarization (AoP), Stokes parameter $S_0$, and Pauli-decomposed pseudo-color—each sharing identical bounding-box annotations. During training, these views are treated as independent samples drawn from a unified distribution via a Dynamic Modality Dropout policy (Bernoulli sampling with $p=0.5$). This strategy forces the backbone to learn modality-invariant geometric features of bottle targets rather than modality-specific artifacts, leveraging the physical complementarity between polarization-based object-background separation and luminance-based morphological reasoning.

**(2) A Metric-Driven Diagnostic Framework: Hierarchical Feature Clustering Evaluation (FCE).** We propose FCE as an architecture-agnostic diagnostic probe that quantifies the quality of feature manifold separation at any network layer. Using RoIAlign-based feature extraction and PCA-projected Fisher Discriminant Scoring, FCE decomposes feature quality into interpretable components: inter-class separability (signal) and intra-class variance (noise). Applied to YOLO11n, FCE precisely localizes a severe semantic collapse at Layers 13 and 19 (the P4/P5 deep fusion stages), characterized by simultaneously high intra-class dispersion and low inter-class separation. This diagnosis provides a quantitative, layer-specific prescription for architectural intervention, replacing empirical trial-and-error with a principled _diagnose-then-repair_ paradigm.

**(3) C3k2-Mamba: A Metric-Guided Selective State Space Backbone.** Guided by FCE diagnosis, we perform targeted surgical replacement of the convolutional bottleneck modules at P4 and P5 with a novel C3k2-Mamba block. This block embeds cascaded Selective State Space Models (Mamba) [CITE] within a Cross-Stage Partial (CSP) dual-stream architecture, enabling $\mathcal{O}(N)$-complexity global context modeling at precisely the layers where long-range dependency is most needed. The dense concatenation of all intermediate Mamba states with the identity stream ensures multi-scale semantic indexing and lossless gradient propagation, directly counteracting the intra-class dispersion and background entanglement identified by FCE.

**(4) A Bottle-Centric Heterogeneous Benchmark.** We construct and release a unified single-class benchmark (28,591 images across train/val/test splits) by consolidating and re-annotating three complementary sources: the PoTATO stereo-polarimetric dataset [CITE] for transparent bottle optical fingerprinting, the FloW-Img USV dataset [CITE] for dynamic floating-pose adaptation, and a self-collected Wild-Bottle corpus of 2,400+ images targeting extreme specular glare and turbid water generalization.

Experimental results demonstrate that our method achieves a mAP@50 of **95.86%** on the Water-Bottle benchmark, representing an **+8.05%** absolute improvement over the YOLO11n baseline (87.81%), while simultaneously _reducing_ model parameters by 15.3% (2.22M vs. 2.62M) and sustaining real-time inference at **125.8 FPS** on an NVIDIA RTX 5090. Comprehensive ablation studies validate each contribution independently and confirm that FCE-guided layer placement achieves higher performance than any alternative placement strategy, establishing FCE as a reusable diagnostic instrument for CNN architecture refinement beyond the scope of this paper.

The remainder of this paper is organized as follows. Section 2 reviews related work on waterborne object detection, polarimetric imaging, and state space models. Section 3 presents the proposed framework in detail. Section 4 describes the benchmark, experimental protocol, and results. Section 5 concludes with a discussion of limitations and future directions.

# 2. Related Work

## 2.1 Waterborne Object Detection

The detection of floating objects on water surfaces has emerged as a focused research area driven by the proliferation of autonomous surface vehicles and environmental monitoring platforms. Early approaches relied on hand-crafted features and classical computer vision pipelines—background subtraction, Gaussian mixture models, and morphological filtering—which proved fragile under changing illumination and wave-induced motion [CITE]. The advent of deep convolutional neural networks dramatically improved detection robustness, with subsequent works adapting general-purpose detectors to the aquatic domain.

The FloW dataset [CITE] established an important benchmark for floating object detection captured from a USV perspective, revealing that standard detectors suffer significant performance degradation due to the low-altitude viewpoint, continuous surface motion, and scale variability inherent to USV deployment. Follow-up works explored temporal modeling [CITE] and domain adaptation [CITE] to address scene dynamics, but remained confined to RGB-only inputs. Concurrent efforts on marine debris detection [CITE] and river litter identification [CITE] confirmed the generality of these failure modes: in all cases, specular reflections and background-foreground texture similarity were identified as primary sources of false negatives, yet no solution addressed these from a physical sensing perspective. Our work departs from this trajectory by explicitly incorporating polarimetric physics into the detection pipeline, treating the sensing modality itself as a first-class design variable rather than a fixed constraint.

## 2.2 Polarimetric Imaging for Visual Recognition

Polarimetric imaging exploits the vectorial nature of light beyond scalar intensity, encoding surface orientation, material properties, and reflection type in the Stokes parameters $[S_0, S_1, S_2, S_3]$. The Degree of Linear Polarization (DoLP) and Angle of Polarization (AoP) derived from these parameters carry physical information that is orthogonal to RGB luminance and fundamentally inaccessible to standard cameras.

In the context of transparent and specular object perception, polarimetric cues have demonstrated unique advantages. Transparent objects exhibit characteristic polarization signatures arising from refraction and internal reflection that are absent from opaque backgrounds [CITE]. The PoTATO dataset [CITE] specifically addressed transparent container detection using stereo-polarimetric imagery, demonstrating that DoLP-based features substantially reduce false negatives for glass and plastic objects that are effectively invisible in RGB. In outdoor water-surface scenarios, the physical asymmetry is even more pronounced: water surface reflections conform closely to Brewster-angle specular reflection and are therefore highly polarized, whereas floating plastic debris—being diffusely reflective—exhibits substantially lower DoLP values [CITE]. This physical contrast directly encodes the foreground-background separation that CNN-based detectors struggle to learn from RGB statistics alone.

Despite these favorable properties, polarimetric sensing has seen limited adoption in real-time detection frameworks. Existing methods either require specialized multi-shot polarimetric cameras that preclude real-time operation [CITE], or process polarimetric channels via simple concatenation without adapting the network architecture to exploit cross-modal complementarity [CITE]. Our approach differs in two key respects: (1) we treat polarimetric views not as additional input channels but as independent training samples under a Dynamic Modality Dropout strategy, forcing the model to internalize modality-invariant representations; and (2) we pair this data-centric strategy with an architectural mechanism specifically designed to exploit the global context encoded in polarimetric statistics.

## 2.3 Real-Time Object Detection Architectures

The YOLO family [CITE: YOLOv1–YOLO11] has dominated real-time detection by coupling single-stage prediction with aggressively optimized convolutional backbones. Key architectural innovations include the Feature Pyramid Network (FPN) [CITE] and Path Aggregation Network (PAN) [CITE] for multi-scale feature fusion, depth-wise separable convolutions for parameter efficiency, and re-parameterized convolutions for inference acceleration. YOLO11 [CITE], the most recent iteration, introduces C3k2 bottleneck modules and the C2PSA attention block, achieving state-of-the-art accuracy-speed trade-offs on COCO.

However, these architectures share a fundamental structural constraint: every feature extraction operation is bounded by a local receptive field. Even with dilated convolutions or deep stacking, the effective receptive field grows only polynomially with depth, and the information aggregated at any given position is dominated by spatially proximate features. This locality bias is well-suited for detecting texturally distinctive objects against clean backgrounds but becomes a liability on dynamic water surfaces, where the discriminative signal is often globally distributed—a floating bottle is detectable precisely because it _disrupts_ the global wave pattern, a relationship that purely local operators cannot encode.

Transformer-based detectors [CITE: DETR, Deformable DETR, RT-DETR] address this through global self-attention, but at the cost of $\mathcal{O}(N^2)$ complexity with respect to the number of spatial tokens. While RT-DETR [CITE] achieves competitive real-time performance through hybrid attention and encoder optimizations, its computational footprint remains substantially higher than nano-scale YOLO variants at equivalent accuracy tiers, limiting its viability for edge deployment on battery-constrained USVs. Our work resolves this tension by selectively introducing global modeling capacity only at the specific network layers where it is diagnostically necessary, preserving the computational efficiency of the CNN backbone elsewhere.

## 2.4 State Space Models for Visual Recognition

State Space Models (SSMs), rooted in linear dynamical systems theory, have recently emerged as a compelling alternative to both CNNs and Transformers for sequence modeling. The S4 model [CITE] demonstrated that structured state space representations could match or exceed Transformer performance on long-range dependency benchmarks at linear complexity. Mamba [CITE] extended this with a selective state space mechanism—input-dependent parameterization of the transition matrices $\bar{\mathbf{A}}$ and $\bar{\mathbf{B}}$—enabling the model to dynamically filter irrelevant context while retaining task-relevant long-range information. Crucially, Mamba achieves $\mathcal{O}(N)$ time and space complexity through a hardware-aware parallel scan algorithm, making it computationally tractable for high-resolution visual inputs.

The adaptation of SSMs to visual recognition has proceeded rapidly. VMamba [CITE] proposed a 2D selective scanning strategy to address the non-sequential nature of image patches, demonstrating competitive classification performance with favorable efficiency. PlainMamba [CITE] and LocalMamba [CITE] further refined scanning patterns and local-global integration. In the object detection domain, MambaYOLO [CITE] and related works [CITE] explored replacing YOLO backbone stages with SSM-based modules, reporting accuracy improvements particularly on small and occluded objects where global context is critical.

Our work makes a distinct methodological contribution relative to prior Mamba-in-YOLO efforts: rather than applying SSM modules heuristically or uniformly across the backbone, we employ FCE diagnosis to _quantitatively identify_ the specific layers where long-range dependency is needed, then perform targeted replacement exclusively at those positions. This metric-guided specificity yields a more parameter-efficient integration—our model uses fewer parameters than the YOLO11n baseline while outperforming it by over 8% mAP—and produces interpretable evidence that the architectural change addresses a precisely characterized representational failure rather than providing a generic boost of unknown origin.

## 2.5 Feature Analysis and Interpretability in Detection

Understanding the internal representational quality of deep detectors has been approached through several complementary lenses. Grad-CAM [CITE] and its variants provide post-hoc gradient-based attribution maps that visualize which spatial regions most influence a prediction, but do not quantify the statistical structure of learned feature distributions. Centered Kernel Alignment (CKA) [CITE] measures representational similarity between layers or models, offering insight into information flow but not into class separability per se. Probing classifiers [CITE] assess the linear decodability of intermediate representations but require supervised probe training for each target layer.

Fisher's linear discriminant [CITE], by contrast, provides a direct, supervision-free measure of class separability in feature space: it simultaneously quantifies inter-class divergence (signal) and intra-class compactness (noise) in a single scalar ratio. Our FCE framework operationalizes this principle within a hierarchical detection backbone by combining RoIAlign-based region feature extraction, PCA-based manifold projection for high-dimensional stability, and layer-wise Fisher scoring across the key feature pyramid levels. Unlike post-hoc attribution methods, FCE is applied _before_ architectural modification to generate a diagnostic prescription, and _after_ modification to verify that the intervention produced the intended representational improvement. This before-after diagnostic loop—which we term the _diagnose-repair-verify_ cycle—constitutes a reusable methodology for principled CNN architecture refinement that is independent of the specific detection task or domain.
# 3. Methodology

This section presents the proposed physically-informed and metric-driven detection framework in full detail. We begin with an overview of the overall architecture (Section 3.1), followed by the heterogeneous data strategy that introduces polarimetric physical priors into training (Section 3.2). We then describe the FCE diagnostic framework that localizes representational bottlenecks in the baseline detector (Section 3.3), and conclude with the C3k2-Mamba module design that surgically addresses the identified failure modes (Section 3.4). The four components are not independent additions but form a coherent pipeline: physical sensing provides the training signal, FCE provides the diagnostic prescription, and C3k2-Mamba provides the targeted architectural remedy.

---

## 3.1 Overall Framework

Water surface floating object detection faces two physically distinct challenges that cannot be simultaneously resolved by any single existing technique. First, transparent and semi-transparent plastic bottles are inherently low-contrast targets: their transmissive surfaces mirror the surrounding water texture, making them nearly indistinguishable from dynamic wave patterns in RGB imagery under specular illumination. This is a _sensing_ problem—the input modality lacks the physical discriminative information needed to separate target from background. Second, even when a suitable input representation is provided, standard CNN backbones exhibit a localized representational failure in their deep feature fusion stages: high intra-class variance caused by wave-induced deformations and reflection artifacts prevents the model from forming compact, stable feature clusters for the same object class. This is a _representation_ problem—the network architecture lacks the global context modeling capacity to aggregate fragmented local evidence into coherent semantic units.

Addressing each challenge independently would be insufficient. A richer sensing modality without a capable architecture cannot exploit the additional physical information. Conversely, a more powerful architecture without physics-aware training data will overfit to modality-specific artifacts. Our framework addresses both simultaneously through two coupled innovations, as illustrated in Fig. 3.

On the **data side**, we propose a Physically-Informed Heterogeneous Data Strategy (Section 3.2) that decomposes raw polarimetric sensor output into $K=6$ physically distinct image views and trains the backbone via Dynamic Modality Dropout, forcing it to internalize modality-invariant representations of bottle targets.

On the **model side**, we propose a two-phase metric-driven architecture optimization. In the diagnostic phase, the Hierarchical Feature Clustering Evaluation (FCE) framework (Section 3.3) is applied to the baseline YOLO11n detector to quantify layer-wise feature separability using Fisher Discriminant Scoring. In the repair phase, C3k2-Mamba modules (Section 3.4) are inserted exclusively at the layers identified as bottlenecks by FCE—the P4 and P5 backbone stages—replacing convolutional bottlenecks with Selective State Space Models that provide $\mathcal{O}(N)$-complexity global context modeling.

The result is a detector that is simultaneously more physically aware (through polarimetric training), more interpretably designed (through FCE-guided architecture selection), and more computationally efficient (through targeted rather than uniform architectural modification) than prior approaches.

---

## 3.2 Physically-Informed Heterogeneous Data Strategy

### 3.2.1 Physical Motivation

Conventional water surface detectors process RGB images, implicitly assuming that luminance-based texture statistics are sufficient for foreground-background discrimination. This assumption fails in two regimes that are central to the target application. Under direct sunlight, specular reflection from the water surface saturates RGB channels and creates bright artifacts whose local statistics are indistinguishable from the diffuse reflectance of floating objects. In low-illumination or turbid conditions, the absence of RGB contrast renders the target effectively invisible.

Physical optics provides an alternative discriminative axis. When unpolarized light reflects from a dielectric surface (water) at near-Brewster angles—the typical geometry for a USV-mounted camera looking across the water surface—the reflected light acquires a strong linear polarization component described by the Stokes parameters $[S_0, S_1, S_2]$. The Degree of Linear Polarization (DoLP) of specular water reflections is typically high ($\text{DoLP} > 0.5$), whereas floating plastic debris, being predominantly diffusely reflective, exhibits substantially lower DoLP values ($\text{DoLP} < 0.2$). This physical asymmetry directly encodes the foreground-background contrast that RGB cannot provide, and it is robust to illumination intensity—it is a geometric property of the reflection geometry, not a photometric property of the scene.

### 3.2.2 Heterogeneous Physical View Decomposition

For samples captured with a polarimetric camera, we decompose the raw Stokes measurements into $K=6$ physically interpretable image views, each providing complementary information about the scene:

- **RGB**: Standard three-channel color image derived from $S_0$ filtered through color channels. Provides texture, color, and morphological cues.
- **Mono**: Grayscale total intensity ($S_0$). Provides luminance distribution without chromatic bias.
- **DoLP** (Degree of Linear Polarization): $\text{DoLP} = \sqrt{S_1^2 + S_2^2} / S_0$. Encodes the proportion of linearly polarized light; high values indicate specular reflection (water surface), low values indicate diffuse reflection (target).
- **AoP** (Angle of Polarization): $\text{AoP} = \frac{1}{2}\arctan(S_2 / S_1)$. Encodes the orientation of the polarization ellipse, providing surface normal and curvature information that distinguishes bottle body geometry from flat water surface.
- **$S_0$ (Total Intensity)**: The first Stokes parameter, equivalent to total unpolarized intensity, used as a physics-consistent grayscale reference.
- **Pauli-RGB**: A false-color composite derived from Pauli decomposition of the polarimetric backscatter matrix, encoding surface roughness and material dielectric properties as color channels.

All six views share identical bounding-box annotations for the "Bottle" class, as the physical transformation preserves spatial extent. This yields a heterogeneous training corpus:

$$\mathcal{D} = {(I_m^{(i)}, \mathbf{y}^{(i)}) \mid m \in {\text{RGB, Mono, DoLP, AoP, }S_0\text{, Pauli}},\ i = 1, \dots, N}$$

where $\mathbf{y}^{(i)}$ denotes the shared label for the $i$-th scene across all modalities.

### 3.2.3 Dynamic Modality Dropout

Naively training on all six views simultaneously risks the backbone developing modality-specific shortcuts—exploiting the characteristic value distributions of a single channel (e.g., the bimodal DoLP histogram of water versus object) rather than learning geometry- and material-invariant features generalizable across modalities. To prevent this, we introduce Dynamic Modality Dropout: during each training iteration, the active input modality is sampled uniformly from the $K=6$ views with probability $p = 1/K$, treating each view as an independent sample draw.

Formally, for iteration $t$ with sampled modality $m_t \sim \text{Uniform}(\mathcal{M})$, the backbone receives input $I_{m_t}^{(i)}$ and optimizes the standard YOLO detection loss $\mathcal{L}_{\text{det}}$:

$$\mathcal{L} = \mathbb{E}_{m \sim \text{Uniform}(\mathcal{M})} \left[ \mathcal{L}_{\text{det}}(f_\theta(I_m), \mathbf{y}) \right]$$

This formulation creates an implicit adversarial constraint: the backbone $f_\theta$ must produce consistent, accurate predictions regardless of which physical view it receives. The only stable solution is to learn features that are invariant to modality-specific appearance variation—precisely the modality-invariant geometric and material properties of bottle targets (curved surfaces, hollow interiors, characteristic aspect ratios) that remain constant across all six physical representations.

For RGB-only samples (from the FloW-Img and Wild-Bottle subsets), the modality is fixed to RGB, and these samples are mixed into the training stream without modification. This asymmetry is intentional: it ensures the model retains high performance under standard RGB-only inference conditions, which is the deployment scenario for USVs without polarimetric cameras.

---

## 3.3 Metric-Driven Diagnosis: Hierarchical Feature Clustering Evaluation (FCE)

### 3.3.1 Motivation and Design Principles

Deep neural network architecture design has historically relied on empirical intuition: practitioners add layers, change modules, or insert attention mechanisms based on general knowledge of their properties, then evaluate the result on a validation metric. This trial-and-error process is computationally expensive, difficult to interpret, and provides no mechanistic understanding of _why_ a modification succeeds or fails.

We propose Hierarchical Feature Clustering Evaluation (FCE) as a principled diagnostic alternative. FCE operates on the feature representations of a trained (or partially trained) baseline model to answer a precise question: _at each layer of the feature pyramid, how well does the learned representation separate the foreground object class from the background?_ The answer is expressed as a single interpretable scalar per layer—the Fisher Discriminant Score—that simultaneously quantifies inter-class separability (desired: high) and intra-class compactness (desired: low). Layers where this score is anomalously low relative to adjacent layers are identified as representational bottlenecks, providing a quantitative prescription for targeted architectural intervention.

### 3.3.2 Feature Extraction via RoIAlign

Let $\mathcal{F}_l \in \mathbb{R}^{H_l \times W_l \times C_l}$ denote the feature tensor output by layer $l$ of the backbone. For a training image with ground-truth bounding boxes $\mathcal{B}_{gt} = {b_1, \dots, b_K}$, we extract foreground feature vectors using RoIAlign [CITE: Mask R-CNN] with a fixed output resolution of $7 \times 7$, followed by spatial average pooling to obtain a $C_l$-dimensional feature vector per box:

$$\mathbf{v}_{fg}^{(k)} = \text{AvgPool}(\text{RoIAlign}(\mathcal{F}_l, b_k)) \in \mathbb{R}^{C_l}, \quad k = 1, \dots, K$$

Background feature vectors ${\mathbf{v}_{bg}^{(j)}}$ are sampled from regions with zero IoU overlap with any ground-truth box, using random spatial sampling with density proportional to the feature map area. We collect $N_{fg}$ foreground and $N_{bg}$ background vectors across a fixed diagnostic subset of $M=500$ images drawn uniformly from the validation set.

### 3.3.3 PCA Manifold Projection

The raw feature vectors $\mathbf{v} \in \mathbb{R}^{C_l}$ reside in a high-dimensional space where Euclidean distance is an unreliable proxy for semantic similarity due to the curse of dimensionality. We apply Principal Component Analysis (PCA) to project all feature vectors onto a $d$-dimensional subspace that captures the dominant axes of variation:

$$\tilde{\mathbf{v}} = \mathbf{P}_d^\top (\mathbf{v} - \boldsymbol{\mu}), \quad \tilde{\mathbf{v}} \in \mathbb{R}^d$$

where $\mathbf{P}_d \in \mathbb{R}^{C_l \times d}$ contains the top-$d$ eigenvectors of the joint feature covariance matrix, and $\boldsymbol{\mu}$ is the global feature mean. We set $d = 32$ across all layers, a value chosen such that the cumulative explained variance ratio exceeds 90% for all evaluated layers (verified empirically; see Section 4.4). This threshold ensures that the projected subspace faithfully represents the full-dimensional feature topology while eliminating noise dimensions that would otherwise inflate intra-class variance estimates.

### 3.3.4 Fisher Discriminant Score

In the projected space $\mathbb{R}^d$, we compute class-conditional statistics for foreground and background:

$$\boldsymbol{\mu}_{fg} = \frac{1}{N_{fg}} \sum_k \tilde{\mathbf{v}}_{fg}^{(k)}, \quad \boldsymbol{\Sigma}_{fg} = \frac{1}{N_{fg}} \sum_k (\tilde{\mathbf{v}}_{fg}^{(k)} - \boldsymbol{\mu}_{fg})(\tilde{\mathbf{v}}_{fg}^{(k)} - \boldsymbol{\mu}_{fg})^\top$$

and analogously for the background. The FCE score at layer $l$ is defined as the Fisher Discriminant ratio:

$$S_{\text{fce}}^{(l)} = \frac{|\boldsymbol{\mu}_{fg} - \boldsymbol{\mu}_{bg}|_2^2}{\text{Tr}(\boldsymbol{\Sigma}_{fg}) + \text{Tr}(\boldsymbol{\Sigma}_{bg}) + \epsilon}$$

where $\text{Tr}(\boldsymbol{\Sigma})$ denotes the trace of the covariance matrix—equal to the sum of per-dimension variances and thus a scalar measure of total intra-class spread—and $\epsilon = 10^{-6}$ is a numerical stability constant.

This ratio has a direct signal-theoretic interpretation: the numerator measures the squared Euclidean distance between class centroids in the projected feature space (inter-class signal strength), while the denominator measures the total statistical dispersion within each class (intra-class noise power). A high $S_{\text{fce}}^{(l)}$ indicates that layer $l$ produces features that cluster tightly within each class and are well-separated between classes—the ideal configuration for a downstream linear classifier or anchor-based detection head. A low score indicates one of two failure modes, which can be diagnosed by examining numerator and denominator independently.

### 3.3.5 Diagnostic Application to YOLO11n

We applied FCE analysis to the YOLO11n baseline model at four key feature pyramid layers: Layer 13 (P4 output), Layer 16 (P4 refined), Layer 19 (P5 output), and Layer 22 (P5 refined, detector head input). The diagnostic results, shown in Fig. 1, reveal two distinct failure modes.

**Semantic collapse at Layer 19.** Layer 19 exhibits the lowest $S_{\text{fce}}$ score across all evaluated layers, driven primarily by an anomalously high $\text{Tr}(\boldsymbol{\Sigma}_{fg})$—the intra-class variance term. Examination of the projected feature manifold (Fig. 1b) confirms that foreground feature vectors for the same bottle instances are highly dispersed: features extracted from different spatial regions of a single bottle (cap, body, base) or from the same region under different lighting conditions form separate, non-overlapping clusters. This intra-class fragmentation reflects the fundamental limitation of local convolutional operators: because each feature vector at Layer 19 integrates information only from its local receptive field, spatially separated parts of the same bottle object cannot be associated with each other, and lighting-induced local variations cannot be contextualized against the global scene statistics needed to recognize them as artifacts.

**Background entanglement at Layer 13.** Layer 13 exhibits the minimum inter-class separation (numerator of $S_{\text{fce}}$), with foreground and background centroids nearly coincident in the projected space (Fig. 1c). This indicates that the mid-level features at P4 cannot distinguish the texture of bottle surfaces from the texture of water ripples—a structurally plausible failure given that both exhibit quasi-periodic patterns at the scale of the P4 receptive field.

**Diagnostic conclusion.** Both failure modes share a common cause: the absence of global context. A model with access to the full scene—the global spatial distribution of wave patterns, the characteristic shape of a floating bottle across its full extent, the consistency of background texture across the entire image—could use this information to contextualize local ambiguities and resolve both the intra-class dispersion and the background entanglement observed at Layers 13 and 19. This diagnosis motivates the targeted introduction of a global context modeling mechanism at precisely the P4/P5 backbone stages, as described in Section 3.4.

---

## 3.4 C3k2-Mamba: Context-Aware Backbone Reconfiguration

### 3.4.1 Selection of the State Space Model

The FCE diagnosis requires a mechanism that can aggregate information across the full spatial extent of the feature map with sub-quadratic complexity—ruling out dense self-attention—while integrating naturally into the existing CSP-based backbone architecture. We select the Selective State Space Model (Mamba) [CITE] for three reasons.

First, Mamba operates at $\mathcal{O}(N)$ time and space complexity through a hardware-aware parallel scan, making it computationally tractable for $640 \times 640$ input images at the feature scales of P4 and P5. Second, its input-selective state transition—where the matrices $\bar{\mathbf{A}}$ and $\bar{\mathbf{B}}$ are functions of the input token—enables the model to dynamically suppress irrelevant context (wave textures that are globally consistent) while amplifying anomalous local signals (bottle features that deviate from the global background distribution). This selective filtering property is precisely what is needed to resolve the background entanglement identified at Layer 13. Third, the recurrent state representation $h_t$ accumulates global context across the entire sequence, ensuring that features at any position are informed by the full scene statistics—directly addressing the intra-class fragmentation at Layer 19.

The core Mamba state equations in discretized form are:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t, \qquad y_t = \mathbf{C} h_t$$

where $x_t \in \mathbb{R}^D$ is the input feature at position $t$ in the flattened sequence, $h_t \in \mathbb{R}^{N_s}$ is the hidden state encoding accumulated global context, $y_t \in \mathbb{R}^D$ is the output, and $\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C}$ are input-dependent learned transition parameters. The 2D feature map $\mathcal{F}_l \in \mathbb{R}^{H \times W \times C}$ is flattened to a sequence of $N = H \times W$ tokens prior to SSM processing and reshaped back to 2D after.

### 3.4.2 C3k2-Mamba Module Design

Rather than replacing the entire backbone with an SSM-based architecture—which would discard the well-optimized local feature extraction properties of the shallow convolutional layers—we apply metric-guided local surgery: only the C3k2 bottleneck modules at the P4 and P5 stages (Layers 4 and 6 in YOLO11n notation, corresponding to the FCE-identified bottleneck positions) are replaced with C3k2-Mamba. All other layers retain their original convolutional structure.

The C3k2-Mamba module integrates SSM blocks within a Cross-Stage Partial (CSP) dual-stream architecture, as illustrated in Fig. 3. The design follows three sequential operations.

**Step 1: Feature Split.** The input tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$ is first processed by a $1 \times 1$ pointwise convolution that maps it to $\mathbb{R}^{H \times W \times C_{mid}}$ with $C_{mid} = C_{in}/2$, and then split along the channel dimension into two equal streams:

$$\mathbf{X}_{id} = \mathbf{X}_{:C_{mid}}, \qquad \mathbf{X}_{comp} = \mathbf{X}_{C_{mid}:}$$

The identity stream $\mathbf{X}_{id}$ bypasses all computation and serves as a lossless gradient pathway, preventing vanishing gradients in the deep backbone. The computational stream $\mathbf{X}_{comp}$ carries the full representational burden of global context extraction.

**Step 2: Cascaded SSM Modeling.** The computational stream enters a cascade of $N$ Mamba blocks. Denoting the $i$-th Mamba block as $\mathcal{M}_i$, the cascade produces intermediate representations:

$$\mathbf{Y}_0 = \mathbf{X}_{comp}, \qquad \mathbf{Y}_i = \mathcal{M}_i(\mathbf{Y}_{i-1}), \quad i = 1, \dots, N$$

Each $\mathcal{M}_i$ applies the selective SSM to the flattened spatial sequence of $\mathbf{Y}_{i-1}$, accumulating progressively longer-range context with each block. We set $N=2$ for both P4 and P5 replacement stages, balancing representational depth against computational cost.

**Step 3: Full-Dimensional Context Reassembly.** The final output aggregates all intermediate representations—the identity stream, the initial computational stream, and all $N$ Mamba block outputs—via dense concatenation followed by a $1 \times 1$ projection:

$$\mathbf{Y}_{out} = \text{Conv}_{1\times1}\left(\text{Concat}\left[\mathbf{X}_{id},\ \mathbf{Y}_0,\ \mathbf{Y}_1,\ \dots,\ \mathbf{Y}_N\right]\right)$$

This dense aggregation serves two purposes. First, it provides the downstream detection head with a multi-scale semantic index: $\mathbf{X}_{id}$ carries local texture features, $\mathbf{Y}_1$ carries short-to-medium range context, and $\mathbf{Y}_N$ carries full-global-range context. The detection head can exploit whichever scale is most discriminative for each spatial location. Second, the explicit retention of $\mathbf{X}_{id}$ and all $\mathbf{Y}_i$ ensures that gradient flow is not bottlenecked through the SSM computation path, maintaining training stability across the full 200-epoch training schedule.

### 3.4.3 Computational Complexity Analysis

The computational cost of C3k2-Mamba relative to the replaced C3k2 module is analyzed as follows. The $1 \times 1$ convolutions in the split and reassembly steps have complexity $\mathcal{O}(HWC^2)$, identical in form to the $1 \times 1$ projections in the original C3k2. Each Mamba block processes a sequence of $N = HW$ tokens at $\mathcal{O}(N \cdot D \cdot N_s)$ complexity, where $D$ is the feature dimension and $N_s$ is the SSM state size. Since $N_s \ll HW$ (we use $N_s = 16$ throughout), the per-Mamba-block cost is $\mathcal{O}(HW \cdot D \cdot N_s) = \mathcal{O}(HWC)$—linear in spatial resolution and strictly lower than the $\mathcal{O}((HW)^2 \cdot C)$ cost of full self-attention at the same resolution.

In practice, replacing C3k2 with C3k2-Mamba at P4 and P5 results in a _net reduction_ in total parameter count from 2.62M (YOLO11n baseline) to 2.22M, because the SSM's compact state representation requires fewer parameters than the stacked convolutional filters in the original bottleneck at these channel widths. The GFLOPs increase from the baseline (3.22 GFLOPs) to 17.85 GFLOPs is attributable primarily to the Mamba block's sequential scan operations over the full spatial sequence, which are memory-bandwidth-bound rather than compute-bound and benefit substantially from the hardware-aware Flash Scan implementation [CITE: Mamba].
# 5. Conclusion

## 5.1 Summary

This paper addressed the problem of real-time transparent floating object detection on dynamic water surfaces—a task where conventional RGB-based CNN detectors fail systematically due to specular reflections, wave-induced texture ambiguity, and the near-zero visual contrast of transparent plastic targets. We identified two root causes underlying this failure: (1) the absence of physics-grounded sensing modalities that encode foreground-background separation as a physical signal rather than a learned statistical pattern, and (2) the lack of a principled diagnostic methodology to localize and remedy representational failures within the detection backbone.

To address these causes jointly, we proposed a physically-informed and metric-driven detection framework built upon YOLO11n. On the data side, we introduced a Heterogeneous Modality Training strategy that interprets polarimetric sensor outputs as $K=6$ independent physical views—RGB, Mono, DoLP, AoP, $S_0$, and Pauli-RGB—and trains the backbone under Dynamic Modality Dropout to learn modality-invariant bottle representations that are robust to specular glare and low-contrast camouflage. On the model side, we proposed the Hierarchical Feature Clustering Evaluation (FCE) framework, which applies Fisher Discriminant Scoring to PCA-projected RoIAlign features across backbone layers to precisely quantify where and how feature discrimination degrades. FCE diagnosis revealed a severe semantic collapse at the P4/P5 fusion stages (Layers 13 and 19) of YOLO11n, characterized by high intra-class dispersion and low inter-class separability. Guided by this diagnosis, we designed C3k2-Mamba—a dual-stream CSP module that embeds cascaded Selective State Space Models at precisely those bottleneck positions—to restore global context modeling at $\mathcal{O}(N)$ complexity without modifying the computationally efficient shallow layers.

Experiments on our Bottle-Centric Heterogeneous Benchmark demonstrate that the proposed method achieves a mAP@50 of **95.86%**, an absolute improvement of **+8.05%** over the YOLO11n baseline, while simultaneously reducing model parameters by **15.3%** (2.22M vs. 2.62M) and sustaining **125.8 FPS** inference on an NVIDIA RTX 5090. Ablation studies confirm that each component contributes independently and that FCE-guided layer placement outperforms all alternative Mamba insertion strategies, validating the diagnostic framework as an effective architecture design navigator. Post-hoc FCE re-evaluation and Grad-CAM visualization provide qualitative confirmation that the proposed architectural intervention produces the representational improvement that the diagnostic analysis prescribed—closing the _diagnose-repair-verify_ loop that constitutes the methodological core of this work.

## 5.2 Broader Implications

Beyond the specific task of waterborne bottle detection, this work offers two methodological contributions of general applicability.

First, the FCE framework is architecture-agnostic and task-agnostic. Any detection pipeline built on a feature pyramid backbone can be subjected to layer-wise FCE analysis to identify representational bottlenecks before committing to architectural modifications. This transforms the network design process from empirical parameter search into a data-informed diagnostic procedure, with the potential to reduce both computational cost and human expertise required for domain-specific model adaptation. We release the FCE analysis code alongside this paper to facilitate adoption by the broader community.

Second, the Physical Domain Randomization strategy—treating heterogeneous physical sensor views as draws from a unified training distribution rather than as separate modalities requiring specialized fusion architectures—offers a practical template for incorporating non-RGB sensing into standard single-input detection frameworks. This approach requires no modification to the detector architecture, no modality-specific preprocessing networks, and no paired multi-modal inference at test time, making it immediately applicable to any scenario where multiple physical sensing channels are available during training but only a subset may be available at deployment.

## 5.3 Limitations and Future Work

We acknowledge several limitations that define the boundary conditions of the current work and motivate future research directions.

**Sensor dependency at training time.** The physical modality strategy relies on polarimetric sensor data (PoTATO subset) for a portion of training samples. While the trained model generalizes to RGB-only inference—as demonstrated by the FloW-Img and Wild-Bottle evaluation subsets—the full benefit of polarimetric training requires access to a polarimetric camera during data collection. Extending the framework to synthesize polarimetric pseudo-labels from RGB via physics-based rendering [CITE] would eliminate this dependency and broaden applicability.

**Single-class specialization.** The current benchmark and model are optimized for the single-class "Bottle" detection task. Real-world waterway monitoring involves diverse debris categories—plastic bags, cans, foam, wood—that may exhibit distinct polarimetric signatures and spatial distributions. Multi-class extension will require revisiting the FCE diagnostic methodology to account for inter-class confusion beyond the binary foreground-background formulation, and may necessitate class-conditional feature analysis.

**Fixed scanning order in Mamba.** The C3k2-Mamba module flattens 2D feature maps into a 1D sequence for SSM processing using a fixed raster-scan order. This introduces a directional bias that may be suboptimal for isotropic spatial patterns such as wave textures, which lack a privileged scanning direction. Adaptive or multi-directional scanning strategies [CITE: VMamba, LocalMamba] could mitigate this limitation and may yield further accuracy improvements, at the cost of increased implementation complexity.

**Evaluation platform homogeneity.** All speed benchmarks are reported on a single high-end GPU (RTX 5090). Practical USV deployment typically targets embedded platforms such as NVIDIA Jetson or Rockchip NPUs, where SSM operations may not receive the same hardware-level acceleration as convolutions. A systematic latency characterization on edge hardware, along with potential quantization and pruning of the Mamba blocks, constitutes an important avenue for future engineering work.

**Generalization to other aquatic domains.** The benchmark consolidates datasets from controlled laboratory settings (PoTATO), freshwater USV deployment (FloW-Img), and open-world web images (Wild-Bottle). Marine environments—with salt spray, foam, and dramatically different lighting conditions—are not represented. Cross-domain generalization studies and domain adaptation experiments would be necessary before deployment in oceanic monitoring contexts.

Future work will explore multi-class debris detection with class-conditional FCE analysis, physics-based polarimetric data synthesis for sensor-agnostic training, adaptive multi-directional Mamba scanning, and systematic edge-hardware benchmarking. We also intend to investigate whether the _diagnose-repair-verify_ paradigm enabled by FCE can be applied iteratively—using post-repair FCE scores to identify residual bottlenecks and guide successive rounds of targeted architectural refinement—as a general strategy for progressive model improvement in specialized detection domains.