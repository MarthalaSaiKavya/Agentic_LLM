# LogTokU Test Results

## Test Queries

1. **"Who is Deepak Sai Pendyala"**
2. **"What is Deep Learning"**

## Results Summary

### Query 1: "Who is Deepak Sai Pendyala"
- **Average LogTokU**: 0.6172
- **Average EU (Epistemic Uncertainty)**: 0.9016
- **Average AU (Aleatoric Uncertainty)**: 0.6842
- **Average Entropy**: 2.7817
- **Generated Response**: "Deepak Sai Pendyala is a former member of the BJP and a former member of the Congress..."

**Observations**:
- The model generated factually incorrect information (claiming membership in political parties)
- Despite being wrong, the model shows relatively low uncertainty (LogTokU: 0.6172)
- This demonstrates the well-known issue where LLMs can be confidently wrong

### Query 2: "What is Deep Learning"
- **Average LogTokU**: 0.6354
- **Average EU (Epistemic Uncertainty)**: 0.9271
- **Average AU (Aleatoric Uncertainty)**: 0.6853
- **Average Entropy**: 2.5413
- **Generated Response**: "Deep learning is a new field of research that has been around for a while..."

**Observations**:
- Slightly higher uncertainty than Query 1 (LogTokU: 0.6354 vs 0.6172)
- The response is more general and less specific
- Lower entropy (2.5413) compared to Query 1, but higher LogTokU

## Key Findings

1. **Uncertainty Comparison**:
   - "What is Deep Learning" shows **higher LogTokU** (0.6354) indicating higher uncertainty
   - "Who is Deepak Sai Pendyala" shows **lower LogTokU** (0.6172) indicating lower uncertainty

2. **EU vs AU**:
   - Both queries show similar AU values (~0.68), indicating similar aleatoric uncertainty
   - EU values are higher for "What is Deep Learning" (0.9271 vs 0.9016), indicating higher epistemic uncertainty

3. **Entropy vs LogTokU**:
   - "Who is Deepak Sai Pendyala" has higher entropy (2.7817) but lower LogTokU (0.6172)
   - "What is Deep Learning" has lower entropy (2.5413) but higher LogTokU (0.6354)
   - This demonstrates that LogTokU captures different aspects of uncertainty compared to entropy

## Interpretation

The results show that LogTokU provides a different perspective on uncertainty compared to traditional entropy-based measures. Notably:

- **LogTokU considers both epistemic (EU) and aleatoric (AU) uncertainty**, providing a more comprehensive measure
- **The model's confidence doesn't always correlate with correctness** - as seen with Query 1 generating incorrect information with relatively low uncertainty
- **LogTokU can help identify cases where the model should be less certain**, even when traditional metrics might suggest high confidence

## Usage

To run the test again:
```bash
cd /home/dpendya/Documents/dlba/logtoku
source venv/bin/activate
python test_queries.py --queries "Who is Deepak Sai Pendyala" "What is Deep Learning" --max_tokens 40
```

To test with different models:
```bash
python test_queries.py --model "model-name" --queries "query1" "query2"
```

## Model Used

- **Model**: GPT-2
- **Device**: CUDA (NVIDIA RTX A5000)
- **Max Tokens**: 40

Note: Using a larger or more recent model (e.g., LLaMA, GPT-3) might show different uncertainty patterns.

