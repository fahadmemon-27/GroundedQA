from sentence_transformers import CrossEncoder
import re

print("Loading NLI model...")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
print("Model loaded.\n")

LABELS = ["contradicted", "supported", "unclear"]


def split_into_sentences(text):
    """
    Simple sentence splitter - splits on '.', '!', '?' followed by a space.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def score_answer(answer, evidence_chunks):
    """
    Splits the answer into sentences, checks each sentence against
    the evidence using NLI, and returns a groundedness score +
    per-sentence breakdown.

    Scoring: "supported" = full credit, "unclear" = half credit
    (since testing showed "unclear" is often correct-but-strict,
    not actually wrong), "contradicted" = no credit.
    """
    sentences = split_into_sentences(answer)

    if len(sentences) == 0:
        return {"groundedness_score": 0.0, "details": []}

    combined_evidence = " ".join(evidence_chunks)
    pairs = [(combined_evidence, sentence) for sentence in sentences]
    scores = nli_model.predict(pairs)

    details = []
    credit_total = 0

    for i, sentence in enumerate(sentences):
        raw_scores = scores[i]
        predicted_label = LABELS[raw_scores.argmax()]

        if predicted_label == "supported":
            credit_total += 1
        elif predicted_label == "unclear":
            credit_total += 0.5
        # "contradicted" adds 0

        details.append({
            "sentence": sentence,
            "label": predicted_label,
            "raw_scores": raw_scores.tolist()
        })

    groundedness_score = (credit_total / len(sentences)) * 100

    return {
        "groundedness_score": groundedness_score,
        "details": details
    }


if __name__ == "__main__":
    evidence_chunks = [
        "The Grotto is a Marian place of prayer and reflection."
    ]

    claim = "The Grotto at Notre Dame is a Marian place of prayer and reflection."
    result = score_answer(claim, evidence_chunks)
    print(f"Groundedness Score: {result['groundedness_score']:.1f}%")
    for item in result["details"]:
        print(f"[{item['label'].upper()}] {item['sentence']}")