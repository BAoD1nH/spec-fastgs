## 1. Reflection Score Extraction
### 1.1 Overview and Rationale of Prior-Guided Initialization

In the paradigm of novel-view synthesis, the standard 3D Gaussian Splatting (3DGS) framework conventionally initializes its 3D spatial representations using a sparse, unstructured point cloud generated as a byproduct of Structure-from-Motion (SfM) algorithms. While computationally convenient, this uniform initialization approach operates under a critical deficit: it is fundamentally agnostic to the intrinsic material properties of the scene. Specifically, it fails to differentiate between Lambertian (diffuse) surfaces and highly specular interfaces prior to the optimization phase. 

This material blindness leads to severe inefficiencies during training. If the model attempts to fit high-frequency, view-dependent specular highlights using a uniform distribution of Gaussians, it inevitably triggers uncontrolled densification—spawning millions of unnecessary parameters in an attempt to capture optical phenomena that standard low-frequency Spherical Harmonics (SH) cannot resolve. 

To circumvent this limitation, our methodology introduces a **Prior-Guided Initialization Pipeline** centered around **Reflection Score (Ref Score) Extraction**. The fundamental objective of this stage is to compute a reliable, deterministic confidence metric that quantifies the probability of a spatial coordinate exhibiting specular properties *before* the neural optimization loop commences. By identifying highly specular regions *a priori*, the system can strategically allocate computationally expensive parameters (such as Anisotropic Spherical Gaussians) exclusively to the geometry that requires them, thereby preserving VRAM sparsity and accelerating convergence.

This initialization pipeline is executed through a deterministic, offline sequence comprising three distinct milestones: (1) 2D Optical Prior Extraction, (2) 3D Visibility Filtering via Space Carving, and (3) Multi-view Confidence Accumulation.

---

### 1.2 Milestone 1: 2D Optical Heuristic Models and Prior Extraction

The first phase of the pipeline operates strictly in the 2D image domain. The theoretical objective is to evaluate the radiometric properties of every pixel across the multi-view dataset to estimate the presence of specular reflection. This estimation is grounded in established physics-based optical heuristics, specifically leveraging the Dichromatic Reflection Model.

#### 1.2.1 Theoretical Foundations
According to the Dichromatic Reflection Model proposed by Shafer and Klinker, the total radiance $L$ reflected from an inhomogeneous dielectric surface is a linear combination of two distinct optical components: the diffuse body reflection ($L_d$) and the specular surface reflection ($L_s$). 

$$ L(\lambda, \theta) = m_d(\theta) c_d(\lambda) + m_s(\theta) c_s(\lambda) $$

Where $\lambda$ represents the wavelength, $\theta$ denotes the photometric angles (viewing and illumination vectors), $m$ represents the geometric scaling factors, and $c$ represents the spectral power distribution. A critical deduction from this model is that the specular component $c_s(\lambda)$ typically assumes the spectral distribution of the ambient illuminant. In standardized lighting conditions, this manifests as a high-intensity, perfectly desaturated (white) signal on the camera sensor. 

Concurrently, the Tan-Ikeuchi model posits that specular pixels exhibit maximum chromaticity intensity compared to their diffuse neighbors. By synthesizing these optical theories, we can isolate pixels that demonstrate simultaneously extreme luminance (high intensity) and extreme chromatic depletion (low saturation).

#### 1.2.2 Mathematical Formulation of the Hybrid Confidence Score
To compute the 2D Reflection Score ($RefScore_{2D}$) for a given image, we evaluate the normalized RGB tensor $[3, H, W] \in [0, 1]$. Let $I_{max}$ and $I_{min}$ denote the maximum and minimum channel intensities for a given pixel, respectively. Pixel saturation is defined inversely proportional to the intensity ratio:

$$ Saturation = 1.0 - \left( \frac{I_{min}}{I_{max} + \epsilon} \right) $$

To ensure smooth gradient transitions and prevent binary thresholding artifacts, we define a continuous activation function $\sigma_{soft}(x, \tau, t)$ known as a Soft-step:

$$ \sigma_{soft}(x, \tau, t) = \frac{1}{1 + \exp\left(-\frac{x - \tau}{t}\right)} $$

The final Hybrid Confidence Score is formulated as a weighted linear combination of four sub-heuristics:

$$ RefScore_{2D} = \omega_1 \cdot S_{Tan} + \omega_2 \cdot S_{Shafer} + \omega_3 \cdot S_{Gray} + \omega_4 \cdot S_{Local} $$

Where the sub-components are mathematically defined as:
1.  **Tan-Ikeuchi Component ($S_{Tan}$):** Isolates pixels maintaining high global intensity thresholds.
    $$ S_{Tan} = \sigma_{soft}(I_{min}, \tau_{min}) \cdot \sigma_{soft}(I_{max}, \tau_{max}) $$
2.  **Shafer-Klinker Component ($S_{Shafer}$):** Penalizes chromatic saturation while rewarding peak intensity.
    $$ S_{Shafer} = \sigma_{soft}(I_{max}, \tau_{int}) \cdot \sigma_{soft}(\tau_{sat} - Saturation, 0) $$
3.  **Gray-Bright Component ($S_{Gray}$):** A direct mathematical evaluation of the desaturated illuminant assumption.
    $$ S_{Gray} = I_{max} \cdot (1.0 - Saturation) $$
4.  **Local Highlight Component ($S_{Local}$):** Evaluates high-frequency spatial contrast relative to a local mean luminance field $\mu_{local}$ (computed via a 2D Box Blur kernel), ensuring the pixel is a localized peak rather than a uniformly bright surface.
    $$ S_{Local} = \max(0, I_{max} - \mu_{local}) $$

**Algorithm 1: 2D Prior Extraction**
```text
Input: Image tensor I ∈ ℝ^(3 × H × W)
Parameters: ω = [0.35, 0.35, 0.20, 0.10], thresholds τ
Output: Confidence Map M ∈ ℝ^(H × W)

1. Compute I_max = max(I, axis=channel)
2. Compute I_min = min(I, axis=channel)
3. Compute Saturation = 1.0 - (I_min / (I_max + ε))
4. Compute S_Tan using Soft-step functions
5. Compute S_Shafer prioritizing low saturation
6. Compute S_Gray = I_max * (1.0 - Saturation)
7. Compute local mean μ_local via convolution(I_max, kernel_size=3)
8. Compute S_Local = max(0, I_max - μ_local)
9. Map M = normalize(ω_1*S_Tan + ω_2*S_Shafer + ω_3*S_Gray + ω_4*S_Local)
10. Return M
```


#### 1.2.3 Analysis of the Algorithmic Design and Mathematical Meaning

The mathematical architecture of Algorithm 1 is deliberately designed to operate in a continuous probability space rather than relying on discrete, binary decision boundaries. The prevalent use of the Soft-step function ($\sigma_{soft}$) over standard Heaviside step functions (hard thresholds) is a critical design choice. In optical physics, the transition between diffuse scattering and specular reflection on a dielectric surface is rarely absolute; it manifests as a continuous gradient governed by the Fresnel equations and surface microfacet roughness. Hard thresholds would introduce severe aliasing and artificially jagged boundaries in the resulting confidence map. By employing parameterized Soft-steps, the algorithm yields a differentiable, smooth probability distribution that accurately reflects the gradual decay of specular energy as the viewing angle shifts away from the ideal reflection vector.

Furthermore, the integration of the spatial convolution step to compute the local highlight component ($S_{Local}$) addresses a significant vulnerability in purely pixel-wise radiometric models. A pixel-wise model evaluates intensity in a vacuum. Consequently, a brightly illuminated, matte white wall could yield identical raw intensity and saturation values as a small specular glare on a dark car. By evaluating the maximum intensity $I_{max}$ against the local spatial mean $\mu_{local}$, the algorithm enforces a geometric constraint: true specular highlights are intrinsically high-frequency, localized phenomena. If a pixel is uniformly surrounded by pixels of equal brightness, $S_{Local}$ approaches zero, effectively suppressing false positives on large, brightly lit diffuse surfaces.

#### 1.2.4 The Rationale for a Hybrid Heuristic Pipeline

One might question the necessity of a weighted, four-component hybrid equation ($RefScore_{2D}$) instead of relying on a singular, established physical model like Shafer-Klinker. The fundamental reason for this hybrid design is the extreme unpredictability of real-world datasets captured in uncontrolled lighting environments (e.g., "in-the-wild" captures). 

A singular model is inherently fragile when its foundational assumptions are violated. For instance, the Shafer-Klinker model assumes the illuminant is perfectly white (desaturated). If the scene is lit by a colored light source (e.g., a neon sign), the specular reflection will retain color saturation, causing the pure Shafer-Klinker model to fail and miss the highlight entirely. Conversely, the Tan-Ikeuchi model is robust to colored lighting due to its focus on peak intensity, but it struggles to differentiate between a physical highlight and a brightly painted diffuse texture. 

By unifying $S_{Tan}$, $S_{Shafer}$, $S_{Gray}$, and $S_{Local}$ into a singular linear combination, the algorithm constructs a highly robust, multi-dimensional filter. The weights ($\omega$) function conceptually as a continuous-space logical AND-gate. To achieve a near-maximum final confidence score, a pixel must simultaneously satisfy the criteria of peak global intensity, extreme chromatic desaturation, and acute local spatial contrast. This hybrid consensus drastically minimizes the false-positive rate across diverse material types and lighting conditions.

#### 1.2.5 Conclusion of the 2D Extraction Stage

Upon the completion of Algorithm 1 for all images in the dataset, the pipeline successfully achieves its first major objective: translating raw, unstructured RGB color data into a deterministic, single-channel 2D probability distribution—the Confidence Map $M$. We have effectively distilled the complex physical phenomenon of specular reflection into a quantifiable mathematical metric on a per-image basis, achieved entirely offline without the exorbitant computational overhead of training a preliminary neural network. 

However, despite the robustness of the hybrid heuristic, these 2D Confidence Maps remain theoretically incomplete. A high score on a single 2D image cannot definitively distinguish between a true view-dependent specular highlight, a static white paint speckle, or momentary sensor noise. Furthermore, 2D pixels inherently lack the depth and geometric context required for 3D Gaussian initialization. To resolve this ambiguity and transition from 2D pixel space to a verified 3D geometric space, the pipeline must advance to the volumetric intersection phase detailed in Milestone 2.

---

### 1.3 Milestone 2: Space Carving and Visibility Filtering

The extraction of 2D heuristics is insufficient for 3D Gaussian initialization, as 2D pixels lack depth information. The pipeline must transition into Cartesian 3D space to establish the actual geometry of the scene. This is achieved through a volumetric intersection technique known as Space Carving.

#### 1.3.1 Volumetric Initialization and Projection
The procedure initiates by establishing a dense, uniform 3D coordinate grid encompassing the bounding volume of the scene, generating a preliminary set of homogeneous points $\mathcal{P} = \{P_1, P_2, \dots, P_N\}$ where $P_i = [x, y, z, 1]^T$. 

To evaluate the validity of these points, they are projected onto the image plane of every available camera $c \in \mathcal{C}$. Let $E_c \in \mathbb{R}^{4 \times 4}$ denote the extrinsic transformation matrix (mapping world coordinates to camera coordinates) and $K_c \in \mathbb{R}^{4 \times 4}$ denote the intrinsic perspective projection matrix. The homogeneous projection of point $P_i$ onto camera $c$ is computed as:

$$ p^{hom}_{i,c} = K_c E_c P_i $$

To transition from Normalized Device Coordinates (NDC) to absolute pixel coordinates $(u, v)$ and depth $Z$, a perspective division operation is executed:

$$ Z_{i,c} = p^{hom}_{i,c}[2], \quad u_{i,c} = \left( \frac{p^{hom}_{i,c}[0]}{Z_{i,c}} + 1 \right) \frac{W}{2}, \quad v_{i,c} = \left( \frac{p^{hom}_{i,c}[1]}{Z_{i,c}} + 1 \right) \frac{H}{2} $$

#### 1.3.2 Strict Visibility Formulation
Space Carving relies on the assumption that a valid 3D surface point must project exclusively onto the foreground of the captured images and must not be occluded. By utilizing the foreground alpha matte $A_c \in [0,1]^{H \times W}$ associated with each image, the system defines a strict boolean Visibility Filter function $V(P_i, c) \in \{0, 1\}$. 

A point is deemed visible and geometrically valid with respect to camera $c$ if and only if it satisfies all frustum boundaries and intersects a foreground pixel:

$$ V(P_i, c) = \begin{cases} 
1 & \text{if } Z_{i,c} > 0 \land u_{i,c} \in [0, W) \land v_{i,c} \in [0, H) \land A_c(u_{i,c}, v_{i,c}) \ge 0.5 \\
0 & \text{otherwise}
\end{cases} $$

If $\exists c \in \mathcal{C}$ such that $P_i$ projects within the frustum bounds but yields $A_c(u_{i,c}, v_{i,c}) < 0.5$ (indicating it lies in the empty background), the point is categorically invalidated (carved) from the spatial grid. This morphological operation ensures the survival of only those points adhering strictly to the visual hull of the target object.


#### 1.3.3 Analysis of the Projection Mathematics and Boolean Constraints

The mathematical formulation of the Visibility Filter $V(P_i, c)$ represents a paradigm shift from the continuous probability space of Milestone 1 to a strict, discrete boolean logic. The operation begins with a rigorous coordinate transformation. Multiplying the homogeneous 3D point $P_i$ by the extrinsic matrix $E_c$ translates and rotates the point from the absolute world coordinate system into the local coordinate space of camera $c$. Subsequent multiplication by the intrinsic matrix $K_c$ and the perspective division by depth $Z$ projects this 3D coordinate onto the 2D image plane. This ensures exact pixel-to-voxel correspondence. 

The design choice to enforce a strict boolean intersection constraint—specifically carving points where $A_c(u_{i,c}, v_{i,c}) < 0.5$—is fundamental to volumetric carving. Unlike the soft, differentiable boundaries required for neural optimization, geometric initialization demands absolute structural rigidity. If a hypothesized 3D point projects onto the background (empty space) in *even a single camera view*, it fundamentally violates the geometric law of occlusion. By unconditionally deleting (carving) these invalid points across all camera projections, the algorithm enforces a continuous morphological intersection. The initial dense, cubic grid of millions of points is systematically whittled away, forcing the surviving points to collapse precisely onto the maximum bound volume (the visual hull) of the target object.

#### 1.3.4 The Rationale for Volumetric Carving over SfM Point Clouds

A critical question arises in the design of this pipeline: Why undertake the computationally intensive process of dense grid generation and Space Carving when Structure-from-Motion (SfM) algorithms like COLMAP already provide a sparse 3D point cloud "for free" during camera pose estimation?

The answer lies in the inherent mathematical limitations of SfM algorithms. Traditional SfM relies on feature matching and photometric consistency—the assumption that a physical point on a surface will maintain the exact same color regardless of the viewing angle. While this holds true for Lambertian (matte/diffuse) surfaces, highly specular materials explicitly violate this assumption. Because specular highlights shift across the surface as the camera moves, SfM algorithms fail to find consistent feature matches on shiny objects like glass, polished metal, or wet ceramics. Consequently, the resulting COLMAP point cloud is heavily biased; it provides dense point coverage in textured, diffuse areas but leaves highly specular surfaces entirely barren.

If the pipeline were to initialize the 3D Gaussians using only the COLMAP point cloud, the model would possess virtually no spatial parameters on the shiny surfaces precisely where the computationally heavy Anisotropic Spherical Gaussians (ASG) are most desperately needed. By discarding the reliance on SfM points and instead executing silhouette-based Space Carving via alpha masks, the pipeline completely bypasses the photometric consistency requirement. The resulting volumetric reconstruction relies solely on the object's geometric boundary, guaranteeing dense, uniform point coverage across the entire surface, regardless of its texture or reflectivity. 

#### 1.3.5 Conclusion of the Visibility Filtering Stage

Upon the completion of the Space Carving algorithm across all camera views, the pipeline achieves its second major milestone: the deterministic generation of a structurally valid 3D geometric canvas. The initial, arbitrary grid of empty space has been successfully sculpted into a dense, tightly bounded 3D point cloud that faithfully represents the physical geometry of the scene, uncompromised by the optical failures of standard SfM techniques.

However, while we have successfully established the "where" of the object, these surviving 3D points remain mathematically blank regarding the "what." They possess valid Cartesian coordinates but carry no optical intelligence regarding their material properties. To resolve this and finalize the initialization state, the pipeline must execute a synthesis of our two milestones: projecting the 2D optical priors extracted in Milestone 1 onto the validated 3D geometry established in Milestone 2. This critical intersection is formalized in the Multi-view Accumulation phase of Milestone 3.

---

### 1.4 Milestone 3: Multi-view Accumulation and Confidence Formulation

The final theoretical challenge involves mapping the 2D optical priors onto the carved 3D geometry. Due to sensor noise, specular aliasing, and the inherent fragility of single-view heuristics, a high $RefScore_{2D}$ on a single image is mathematically insufficient to confirm the presence of a physical specular surface. 

#### 1.4.1 The Theory of View-Dependent Consensus
True specular phenomena are defined by their strict view-dependency; the reflected energy lobe is geometrically consistent across a specific angular trajectory relative to the surface normal and the light source. Consequently, a valid specular point will manifest high luminance and low saturation across multiple adjacent camera poses within that trajectory. 

To formalize this consensus, the system eschews variance-based metrics in favor of **Multi-view Accumulation**. By integrating the 2D confidence maps over the entire valid camera set, the accumulation acts as a spatial low-pass filter. Uncorrelated 2D artifacts are attenuated, whereas physically grounded specular regions receive constructive mathematical reinforcement.

The final 3D Reflection Score for a surviving geometric point $P_i$ is defined as the summation of the projected 2D prior scores, strictly gated by the Visibility Filter:

$$ RefScore_{3D}(P_i) = \sum_{c=1}^{|\mathcal{C}|} \Big( RefScore_{2D, c}(u_{i,c}, v_{i,c}) \cdot V(P_i, c) \Big) $$

#### 1.4.2 Strategic Allocation and Point Cloud Generation
Upon completion of the accumulation loop, the continuous field of $RefScore_{3D}$ provides a definitive probability map of material specularity across the 3D visual hull. To finalize the initialization state, the system employs a rigid memory-budgeting mechanism. 

Let $\mathcal{B}$ denote the total permissible point budget (e.g., 100,000 points). The budget is bifurcated into a base allocation and a specular allocation. The points are sorted in descending order based on their $RefScore_{3D}$. The subset of points exhibiting the maximum accumulated scores are formally classified as Specular Points and are initialized with Anisotropic Spherical Gaussian capacity. The remainder of the budget is uniformly sampled from the surviving points to ensure adequate diffuse geometric coverage.

**Algorithm 2: Space Carving and Multi-view Accumulation**
```text
Input: Uniform 3D Grid P, Camera Poses {E_c, K_c}, Alpha Masks {A_c}, Prior Maps {RefScore_2D_c}
Output: Prior-guided 3D Point Cloud with ASG assignments

1. Initialize Accumulated_Score[P] = 0 for all P
2. Initialize Valid_Mask[P] = True for all P
3. For each camera c in C:
4.     Project all P to (u, v, Z) using E_c and K_c
5.     For each point P_i in P:
6.         If Z_i > 0 AND (u_i, v_i) inside bounds:
7.             If A_c(u_i, v_i) < 0.5:
8.                 Valid_Mask[P_i] = False // Carving step
9.             Else If Valid_Mask[P_i] == True:
10.                Accumulated_Score[P_i] += RefScore_2D_c(u_i, v_i)
11. Discard all points where Valid_Mask == False
12. Sort surviving points by Accumulated_Score descending
13. Assign top N points as Specular Gaussians
14. Sample remaining budget randomly as Diffuse Base Gaussians
15. Return Final Point Cloud
```

Through this rigorous theoretical pipeline, the system decisively resolves the material-agnostic flaw of standard 3DGS initialization, ensuring that the subsequent optimization loop is constrained by physically accurate, prior-guided spatial intelligence.

#### 1.4.3 Analysis of the Accumulation Mathematics and Gating Mechanisms

The mathematical formulation of the $RefScore_{3D}$ equation operates fundamentally as a discrete integration of specular probability over the angular domain. By evaluating the summation, the system transforms fragile, isolated 2D observations into a robust, geometrically grounded 3D consensus. 

The interplay between the two variables in the summation—the magnitude $RefScore_{2D}$ and the boolean gate $V(P_i, c)$—is of paramount importance. If the system were to simply project and average 2D pixel values without strict visibility constraints, occluded cameras or cameras observing the background would project arbitrary noise or zeros onto the 3D point, irreparably corrupting the calculation. By explicitly multiplying the continuous 2D prior by the discrete Visibility function, $V(P_i, c)$ acts as a strict gating mechanism. It guarantees that the accumulation equation only integrates data from verified, unoccluded lines of sight. Consequently, the resulting $RefScore_{3D}$ is an absolute measure of verified specular energy rather than a diluted statistical average.

#### 1.4.4 The Rationale for Constructive Accumulation over Statistical Variance

In classical photometric stereo and early view synthesis literature, specular surfaces are often identified by calculating the statistical variance ($\sigma^2$) of luminance across multiple viewpoints. A critical design decision in this pipeline is the explicit rejection of standard variance in favor of **Constructive Accumulation** ($\sum$). 

The rationale for this divergence is rooted in the spatial sparsity of specular lobes. A physical specular highlight (e.g., the glare of a light source on polished metal) is highly directional. If a point on a metallic surface is observed by 50 cameras, the intense specular glare may only intersect the viewing frustum of 3 cameras, while the remaining 47 cameras observe a dark, matte diffuse color. If standard variance were applied, the 47 low-intensity observations would anchor the statistical mean near zero, severely suppressing the variance output and causing the system to overlook the specular point.

Constructive accumulation circumvents this statistical suppression. By directly summing the raw probability scores, the metric is highly sensitive to peak energy. Even if a highlight is only visible across a narrow trajectory of 3 cameras, the high $RefScore_{2D}$ values from those specific views will constructively interfere (sum together) to create a distinct, mathematically significant peak in the 3D score. 

Furthermore, this accumulation inherently acts as an aggressive low-pass filter against optical anomalies. A "fake" specular highlight—such as momentary sensor noise, a dead pixel, or an algorithmic false positive on a single 2D image—lacks view-dependent geometric consistency. It will only register a high score on a single camera. Because it lacks multi-view corroboration, its accumulated 3D score remains negligible, preventing the system from falsely assigning an ASG to a noisy coordinate.

#### 1.4.5 Strategic Budget Allocation and Complexity Bounds

The final algorithmic step—sorting the 3D points and rigidly bifurcating the point budget—is a necessary intervention to enforce computational tractability. Anisotropic Spherical Gaussians (ASG) demand exponentially higher memory bandwidth and floating-point operations per second (FLOPS) during rasterization compared to standard Spherical Harmonics. If the system allowed unchecked ASG initialization based on a soft probability threshold, a highly reflective scene could trigger a VRAM overflow.

By sorting the $RefScore_{3D}$ values in descending order and applying a hard numerical cut-off (e.g., restricting ASG allocation to the top 50,000 points), the pipeline establishes a strict upper bound on computational complexity. This guarantees that rendering will achieve real-time frame rates regardless of the scene's material composition. Conversely, by uniformly sampling the remaining point budget from the lower-scoring points, the algorithm ensures that the diffuse, matte surfaces of the scene retain sufficient geometric coverage to accurately reconstruct the base structure without wasting computational capacity.

#### 1.4.6 Conclusion of the Prior-Guided Initialization Pipeline

The execution of Milestone 3 marks the completion of the Prior-Guided Initialization stage. Through a deterministic, three-stage pipeline, the system has successfully bridged the gap between 2D optical physics and 3D geometry. We initiated the process with uncalibrated multi-view images and an empty spatial grid. We extracted robust 2D optical priors, sculpted a rigorous 3D visual hull via volumetric space carving, and synthesized them through multi-view constructive accumulation. 

The resulting output is not merely a spatial point cloud, but a highly optimized, *material-aware* geometric canvas. Every 3D coordinate now possesses an analytically verified probability of specular behavior. This achieves the ultimate goal of the offline pre-processing stage: it perfectly sets the stage for the neural optimization loop, allowing the 3D Gaussian Splatting architecture to focus its computational power exactly where the physics of light dictate it is most required.