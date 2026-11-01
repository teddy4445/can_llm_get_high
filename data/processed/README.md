# Processed dataset — per-run scores

`llm_narrative_scores.csv` contains the two outcome measures for every one of the
3,000 model-generated narratives analysed in the manuscript
*Can LLMs Get High? A Dual-Metric Framework for Evaluating Psychedelic Simulation
in Large Language Models* (Ben-Zion, Simon & Lazebnik).

One row = one generation run (5 models x 6 conditions x 100 samples = 3,000 rows).

| Column | Description |
|---|---|
| (index) | Run identifier, 0-2999 |
| `Model` | LLM that generated the narrative. Labels as stored in the raw output: `chatGPT` = ChatGPT-5 (OpenAI); `Gemini` = Gemini 2.5 (Google DeepMind); `Sonnat` = Claude Sonnet 3.5 (Anthropic); `Llama ` (trailing space) = Llama-2 70B (Meta AI); `Falcon` = Falcon 40B (TII) |
| `condition` | Prompt condition: `neutral` (no substance; ordinary meaningful day) or one of the five classic-psychedelic induction prompts `psilocybin`, `LSD`, `DMT`, `Ayahuasaca` (sic; = ayahuasca), `Mescaline` |
| `NLP_distance` | Semantic similarity to the 1,085 human Erowid reports: mean cosine similarity of the narrative's Sentence-BERT embedding to all human-report embeddings, linearly rescaled to [0, 1] across all generated outputs. Higher = closer to human psychedelic reports |
| `MEQ_score` | Mystical Experience Questionnaire-30 total, completed by the same model instance from the narrator's perspective (30 items x 0-5), divided by 150 to give a 0-1 score |

Sanity check against the manuscript (means): neutral NLP 0.156 / MEQ 0.046;
pooled psychedelic NLP 0.548 / MEQ 0.748.

The full text of the generated narratives and the item-level MEQ-30 responses are
not included here; they are available from the corresponding author on reasonable
request. The human reports were obtained from the Erowid Experience Vaults and are
subject to Erowid Center's terms of use, so they are not redistributed.
