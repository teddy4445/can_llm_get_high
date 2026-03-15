
# Can LLM Get High

## Files
- `llm_psychedelic_reproduction.ipynb` — main Colab notebook
- `api_keys.template.json` — placeholder secrets file
- `requirements.txt` — pip dependencies
- `llm_psychedelic_repro/` — package with the class-based implementation

## Recommended Colab workflow
1. Upload the whole folder or zip to Colab.
2. Copy `api_keys.template.json` to `api_keys.json`.
3. Fill each required key with a real value.
4. Put the approved Erowid export at:
   - `data/input/erowid_approved_export.csv`
5. Run the notebook top to bottom.
6. Check outputs under:
   - `outputs/figures`
   - `outputs/tables`

## Full-run cost note
A full run with 5 models × 6 conditions × 100 samples triggers:
- 3,000 generation calls
- 3,000 MEQ-scoring calls