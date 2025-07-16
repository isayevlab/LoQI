# Megalodon Model Card 

# Megalodon Overview

## Description:  

Megalodon is a transformer-based 3D molecule generative model augmented with simple equivariant layers and trained using a joint continuous and discrete denoising co-design objective. Megalodon achieves state-of-the-art results in 3D molecule generation, conditional structure generation, and structure energy benchmarks using diffusion and flow matching. Megalodon produces up to 49x more valid molecules at large sizes and 2-10x lower energy compared to the prior best generative models.

This model is ready for commercial use.  

### License/Terms of Use:   
Megalodon source code is licensed under Apache 2.0 and the model is licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). By using Megalodon, you accept the terms and conditions of this license.  

### Deployment Geography:  
Global  

### Use Case:  
Megalodon can be used to generate valid, diverse, novel molecules with optimal low energy structures. The model can be used by chemists, researchers, and academics to design new small molecules.  

### Release Date:  
Github 07/22/2025 via [https://github.com/NVIDIA-Digital-Bio/megalodon](https://github.com/NVIDIA-Digital-Bio/megalodon)  

## Reference(s):  
Reidenbach et al. [https://arxiv.org/abs/2505.18392](https://arxiv.org/abs/2505.18392)  

## Model Architecture:   
**Architecture Type:** Equivariant Graph Transformer  
**Network Architecture:** EGNN layers with Transformer  

## Input:  
**Input Type(s):**  
-Random gaussian noise  
-Random charge  
-Atom type  
-Edge type discrete variables  
**Input Format(s):** Continuous 3D vector, 1D one hot vector, 1D 1 hot vector, 2D 1 hot vector  
**Input Parameters:** (3D, 1D, 1D, 1D)  
**Other Properties Related to Input:** The model tested up to 125 atoms (sequence length). The input gaussian noise is centered.

## Output:  
**Output Type(s):** 3D molecules  
**Output Format:** sdf files  
**Output Parameters:** The molecule will have an Nx3 feature for the 3D positions of the atom coordinates. There are also Nx1 discrete features for the edge types, charge and atom types. All of this can be processed with RDKit.  
**Other Properties Related to Output:** N/A  

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.   

## Software Integration:  

**Supported Hardware Microarchitecture Compatibility:**  
* [NVIDIA Ampere A100]  

**[Preferred/Supported] Operating System(s):** Linux  

## Model Version(s):   
V1.0  

# Training, Testing, and Evaluation Datasets:   

** The total size (in number of data points): 274906  
** Total number of datasets: 1  
** Dataset partition: Training 98[%], testing 1[%], validation 1[%]  

## Training Dataset 

Dataset: [GEOM-Drugs](https://www.nature.com/articles/s41597-022-01288-4)  
Data Collection Method by Dataset: Automated  
Data Labeling Method by Dataset: Automated  
Properties: The GEOM dataset (Axelrod & Gomez-Bombarelli, 2022) is widely used for 3D molecular structure (conformer) generation tasks, containing 3D conformations from both the QM9 and drug-like molecule (DRUGS) databases, with the latter presenting more complex and realistic molecular challenges. Conformers in the dataset were generated using CREST (Pracht et al., 2024), which performs extensive conformational sampling based on the semi-empirical extended tight-binding method (GFN2-xTB) (Bannwarth et al., 2019). This ensures that each conformation represents a local minimum in the GFN2-xTB energy landscape.

## Testing Dataset

Dataset: GEOM-Drugs  
Data Collection Method by Dataset: Automated  
Data Labeling Method by Dataset: Automated

## Evaluation Dataset  
Dataset: GEOM-Drugs  
Data Collection Method by Dataset: Automated  
Data Labeling Method by Dataset: Automated

## Performance:  
We show that Megalodon achieves state-of-the-art results in 3D molecule generation, conditional structure generation, and structure energy benchmarks using diffusion and flow matching. Furthermore, doubling the number of parameters in Megalodon to 40M significantly enhances its performance, generating up to 49x more valid large molecules and achieving energy levels that are 2-10x lower than those of the best prior generative models. [https://arxiv.org/pdf/2505.18392](https://arxiv.org/pdf/2505.18392) 

## Inference:  
Engine: PyTorch  
Test Hardware: A6000, A100

## Ethical Considerations:  
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.  

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [Insert Link to Model Card++ subcards here]. 

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Bias Subcard

| Participation considerations from adversely impacted groups protected classes in model design and testing | Not Applicable |
| :---- | :---- |
| Measures taken to mitigate against unwanted bias | Not Applicable |

# Explainability Subcard

| Intended Task/Domain | 3D molecule generation |
| :---- | :---- |
| Model Type | Transformer |
| Intended Users: | Chemists, GenAi creators for drug discovery |
| Output: | 3D molecule (xyz positions, atom types, atom charges, bond types) |
| Describe how the model works: | Specify the number of molecules and the number of atoms of each and the model generates 3D molecules of the desired sizes |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  | Not Applicable |
| Technical Limitations & Mitigation: | Model may not perform well for larger molecules outside the training dataset. The model cannot generate molecules with atom types not seen in the training data. |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics: | 2D and 3D molecular validity |
| Potential Known Risks: | Invalid and unphysical geometry molecules are still possible to be generated.  |
| License | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)  |

# Privacy Subcard

| Generatable or reverse engineerable personal data? | No |
| :---- | :---- |
| Personal data used to create this model? | No |
| How often is dataset reviewed? | Before release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Yes |
| Applicable Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/)  |

# Safety Subcard

| Model Application Field(s): | Healthcare |
| :---- | :---- |
| Describe the life critical impact (if present). | Experimental drug discovery and medicine. Additional in silico and in vitro tests are recommended before using the molecules for downstream applications. |
| Use Case Restrictions: | Abide by [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |

