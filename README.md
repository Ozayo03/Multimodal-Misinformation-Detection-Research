# TRUE Video-Claim Detection: MiniCPM and Qwen Research Adaptation

This repository contains research adaptations of the original TRUE dataset and 3MFact code associated with "[Pioneering Explainable Video Fact-Checking with a New Dataset and Multi-role Multimodal Model Approach](https://ojs.aaai.org/index.php/AAAI/article/view/35048)" (AAAI 2025). The original authors, citation, and licenses are retained below. This adaptation is not the official upstream repository and its experimental results should not be presented as a reproduction of the complete original system.

## Research adaptation

The primary experiment evaluates binary video-claim detection with pretrained MiniCPM-V 2.6, without and with curated TRUE dataset evidence. Qwen2.5-VL-7B-Instruct was evaluated as a secondary development-sample comparator. Model weights were **not trained or fine-tuned**. The local direct-judgment mode disables web retrieval and the question-generation/video-QA loop. It uses saved video information, including generated descriptions and dataset-provided transcripts; it does not perform automatic speech recognition. The final direct judge does not revisit the raw video frames.

Curated dataset evidence is supplied text, not live retrieved evidence. It may include fact-check conclusions or quoted misinformation. Dataset ratings, reference rationales, and evidence-relationship annotations are excluded from the evidence selector, but content-level leakage still requires assessment. A binary False verdict must not be interpreted as proof that an unsupported claim was independently disproven.

### Experimental scope and limitations

- MiniCPM's full available train/validation comparison scored 2,308 paired valid claims: accuracy 67.50% without versus 71.32% with dataset evidence; macro-F1 64.46% versus 68.21%. These are paired scores, not independent full-split coverage.
- A separate MiniCPM test comparison scored 385 paired valid claims: accuracy 64.94% versus 71.43% and macro-F1 63.27% versus 69.66%. Test results were inspected during development; the test set is not untouched confirmation.
- The later 100-claim MiniCPM/Qwen comparison used shared MiniCPM-generated video information and saved evidence, so it is a judgment-stage comparison, not an end-to-end comparison of each model's independent video understanding. Its four conditions vary instruction style and answer order; all receive saved evidence context, which can be empty.
- Historical evidence comparisons also changed evidence-specific instructions; the resulting gains are associations between configurations, not an isolated causal estimate for evidence text alone. Independent evidence-run coverage and failure-inclusive scores remain to be reconciled.
- Explanation generation and faithfulness remain future research. Automated judge experiments showed position sensitivity and format failures; human review was not completed. Similarity metrics are not factuality guarantees.

### Dataset and checkpoints

Obtain TRUE from the upstream links below, subject to its license and any institutional requirements. Place the full dataset and model checkpoints outside the repository. Expected dataset directories include `test`, `train_val`, `test_video`, `train_val_video`, `test_output`, and `train_val_output`, plus `test_set.txt` and `train_val_set.txt`. The `*_output` directories contain cached video-description JSON inputs and are not distributed as part of this source-only workflow.

Set `DATASET_ROOT_DIR`, `MINICPM_LLM_MODEL_PATH`, `MINICPM_VIDEO_MODEL_PATH`, and `MINICPM_IMAGE_MODEL_PATH` to your own absolute paths. If needed, set `SPACY_MODEL_PATH` to an installed spaCy model. Never commit model weights, raw videos, downloaded dataset contents, credentials, or experiment outputs without confirming permissions and reviewing what will be published.

### Environments

The historical MiniCPM cluster environment used Python 3.10, PyTorch 2.11.0+cu128, NumPy 1.26.4, Transformers 4.44.0, spaCy 3.7.5, and scikit-image 0.25.2. The Qwen environment used Transformers 4.51.3 and `qwen-vl-utils` 0.0.14 with the cluster's PyTorch installation. Exact CUDA, model revision, and remaining package versions must be verified for the target machine.

**Do not assume `3MFact_Code/requirements.txt` reproduces these environments:** it retains older pins, including Torch 2.4.0 and NumPy 2.2.3. Reconcile dependencies in isolated environments before use; installing that file unchanged can recreate earlier compatibility problems. The earlier `3MFact_Code/Code_USAGE.md` describes upstream usage and is not an authoritative setup guide for this local-only adaptation. Model checkpoints using `trust_remote_code=True` execute checkpoint-provided Python; review and trust the source before loading.

### Running on a cluster

Use the cluster's approved GPU allocation procedure and confirm CUDA is visible on the compute node. The original experiments used a Slurm `biggpu` partition, but GPU allocation, node exclusions, and memory limits are site-specific. Do not run model inference or video decoding on the login node. The following environment settings illustrate a **small MiniCPM evidence-assisted run after cached descriptions already exist**, and should be placed in an approved compute-node job script:

```bash
export DATASET_ROOT_DIR=/path/to/TRUE_Dataset
export MINICPM_LLM_MODEL_PATH=/path/to/MiniCPM-V-2_6
export MINICPM_VIDEO_MODEL_PATH=/path/to/MiniCPM-V-2_6
export MINICPM_IMAGE_MODEL_PATH=/path/to/MiniCPM-V-2_6
export USE_EVIDENCE_RETRIEVAL=false
export USE_DATASET_EVIDENCE=true
export JUDGMENT_STRATEGY=evidence_v2
export JUDGMENT_INPUT_MODE=all
export DATASET_SPLITS=train_val
export SKIP_VIDEO_DESCRIPTIONS=true
export CLAIM_LIMIT=5
export CLAIM_SAMPLE_SEED=13
export VERDICT_RUN_NAME=example_evidence_new

cd /path/to/TRUE-3MFact/3MFact_Code
test ! -e "$DATASET_ROOT_DIR/train_val_output_$VERDICT_RUN_NAME" || { echo "Choose a fresh run name"; exit 1; }
python main_Pipeline.py
```

For a no-dataset-evidence comparison, use `USE_DATASET_EVIDENCE=false` with a different fresh `VERDICT_RUN_NAME`, keeping the split, seed, inputs, and other settings fixed. Historical evidence runs included additional evidence-specific instructions, so matching these settings does not make the historical comparison evidence-text-only. Inspect logs and per-item outputs: a completed Slurm job does not necessarily mean every claim succeeded.

**Output safety:** `main_Pipeline.py` may delete a pre-existing per-claim output directory when it retries an invalid verdict. Use a new run name and never point an exploratory run at irreplaceable results. Its thread-based timeout does not terminate the underlying work; do not rely on it as a hard cancellation boundary.

### Comparison tools

`3MFact_Code/compare_runs.py` reads two output directories and reports paired classification metrics and an exact McNemar test. Its validity check requires a binary Answer and nonempty Reasons, but does not check every verdict-schema field. Candidate IDs come from candidate directories, so independently account for selected claims that never created a directory and report each run's coverage.

The Qwen pilot and frozen prompt diagnostics are in `qwen_pilot.py`, `qwen_shared_judgment.py`, and `qwen_prompt_comparison.py`; run their `--help` for the required input paths and options in a compatible GPU environment. The Qwen pilot requires existing cached MiniCPM descriptions and seven cached keyframes per case. Research outputs, prompts, metrics, and sampled IDs must be recorded with model versions and data provenance; do not treat development results as untouched evaluation.

## Overview

We present TRUE (Truthfulness and Rationale with Underlying Evidence), a comprehensive video fact-checking dataset that includes detailed annotations for veracity labels, rationales, and supporting evidence. Along with the dataset, we also introduce 3MFact, a multi-role collaboration framework for video fact-checking.



## Project Poster
<img src="./Poster.png" width="90%">




## Dataset Highlights

### Dataset Statistics

<img src="./TRUE_Dataset/Dataset_Statistics.png" width="70%">

The TRUE dataset contains:
- **Total Samples**: 2,925 video fact-checking cases (2016-2024)
- **Video Length**: All videos are under 5 minutes
- **Veracity Labels**:
  - True Labels (1,097 samples):
    - Including True, Mostly True , and Correct Attribution
  - False Labels (1,828 samples):
    - Including False, Miscaptioned, Mixture, Fake, and Mostly False
- **Multi-platform Coverage**:
   - Social Media (SM): YouTube, X (Twitter), TikTok, Facebook, Instagram
   - Official Media (OM): New York Times (NYT), CNBC, Fox News

### Key Features
1. **Dual Rationales System**:
   - **Expert-Crafted Rationales (ECR)**: 
     - Direct extraction from professional fact-checking articles (Snopes)
     - Preserves original fact-checking reasoning and expert insights
     - Contains main rationales for direct rating justification
     - Includes additional supporting rationales for comprehensive verification
   - **LLM-Summary Rationales (LSR)**:
     - Systematically synthesized explanations through LLMs
     - Transforms fact-checking articles into concise yet comprehensive summaries
     - Decomposes the verification process into structured reasoning chains


2. **Comprehensive Annotations**:
   - Detailed Evidence Collection
   - Evidence-Rationale Relationship Analysis
   - Video Information (transcripts, metadata)





## Getting Started

### TRUE Dataset
- Download the TRUE dataset from [Baidu Netdisk]( https://pan.baidu.com/s/1yNA4dg5H1sxm5-XqUr-SQw?pwd=r8ws) | [OneDrive](https://whueducn-my.sharepoint.com/:u:/g/personal/kaipengniu_whu_edu_cn/EcMBGGGA7VVHsTwLui4gJMYBZDH-p3A8GJiXHamIiVTkoA?e=GPhcfv)
- For detailed documentation of the dataset structure, components, and statistics, please refer to [`TRUE_Dataset/TRUE_Dataset.md`](./TRUE_Dataset/TRUE_Dataset.md)

### 3MFact Code
- The adapted inference workflow, environment notes, evaluation limitations and cluster example are documented in [`3MFact_Code/Code_USAGE.md`](./3MFact_Code/Code_USAGE.md) and summarized in the Research adaptation section above.
- This repository contains pretrained-model inference and comparative evaluation code; it does not train or fine-tune MiniCPM or Qwen.



## **Citation**
If you use this dataset or code in your research, please temporarily cite our paper as follows :


```bibtex
@inproceedings{niu2025pioneering,
  title={Pioneering Explainable Video Fact-Checking with a New Dataset and Multi-role Multimodal Model Approach},
  author={Niu, Kaipeng and Xu, Danni and Yang, Bingjian and Liu, Wenxuan and Wang, Zheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28276--28283},
  year={2025}
}
```
