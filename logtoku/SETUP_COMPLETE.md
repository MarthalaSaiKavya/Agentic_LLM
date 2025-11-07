# LogTokU Setup Complete

## Setup Summary

LogTokU has been successfully set up and tested. The following has been completed:

### 1. Environment Setup
- Created Python virtual environment (`venv/`)
- Installed all required dependencies from `requirements.txt`
- Verified CUDA support is available (NVIDIA RTX A5000)

### 2. Bug Fixes
- Fixed incorrect config file path in `DynDecoding/gene.py`
- Changed from `"end/configs/prompt.json"` to use proper relative path `configs/prompts.json`

### 3. Test Results
- All package imports successful
- LogTokU uncertainty calculation working correctly
- EU (Epistemic Uncertainty) calculation verified
- AU (Aleatoric Uncertainty) calculation verified
- LogTokU (EU * AU) combination verified
- Test correctly identifies high vs low uncertainty cases

## Usage

### Activate Virtual Environment
```bash
cd /home/dpendya/Documents/dlba/logtoku
source venv/bin/activate
```

### Run Test
```bash
python test_logtoku.py
```

### Run Experiments

#### Experiment 1: LogTokU-guided Decoding
```bash
cd DynDecoding
python gene.py --exp llama2_7b --gpuid 0 --quantize False
```

Note: You'll need to:
1. Update model paths in `gene.py` (currently set to `/path/to/your/models/...`)
2. Have Hugging Face token configured: `huggingface-cli login`
3. Download required models or use existing model paths

#### Experiment 2: LogTokU-guided Response Uncertainty Estimation
```bash
cd SenU
python generate.py --model_name llama2_chat_7B --gene 1 --mode one_pass --gpuid 0
python one_eval.py
python muti_eval.py --model 7b --gpuid 0
```

## Next Steps

1. **Configure Hugging Face Access** (if using Hugging Face models):
   ```bash
   huggingface-cli login
   ```

2. **Update Model Paths**:
   - Edit `DynDecoding/gene.py` to set correct model paths
   - Or download models using `download_model.py` (after configuring HF token)

3. **Test with Real Models**:
   - Ensure you have GPU access
   - Run experiments with actual language models
   - Verify LogTokU uncertainty estimation on real tasks

## Files Created/Modified

- `requirements.txt` - Python dependencies
- `setup.sh` - Setup script
- `test_logtoku.py` - Test script for verification
- `DynDecoding/gene.py` - Fixed config path bug

## System Information

- Python: 3.10
- PyTorch: 2.9.0+cu128
- CUDA: Available (NVIDIA RTX A5000)
- Transformers: 4.57.1

