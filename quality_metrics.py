from nltk.tokenize import word_tokenize
from collections import Counter
from math import exp, log
from rouge import Rouge

def calculate_bleu_score(reference, candidate, max_n=4):
    """
    Calculate a simplified BLEU score for a given reference and candidate text.
    
    :param reference: str, the reference text
    :param candidate: str, the candidate text generated by the model
    :param max_n: int, the maximum n-gram order to consider (default: 4)
    :return: float, the BLEU score
    """
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    reference_length = len(reference_tokens)
    candidate_length = len(candidate_tokens)
    
    clipped_counts = {}
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(zip(*[reference_tokens[i:] for i in range(n)]))
        cand_ngrams = Counter(zip(*[candidate_tokens[i:] for i in range(n)]))
        clipped_counts[n] = sum((cand_ngrams & ref_ngrams).values())
    
    brevity_penalty = min(1, exp(1 - reference_length / candidate_length)) if candidate_length > 0 else 0
    
    if sum(clipped_counts.values()) == 0:
        return 0
    
    geometric_mean = exp(sum(log(clipped_counts[n] / max(1, candidate_length - n + 1)) for n in range(1, max_n + 1)) / max_n)
    
    return brevity_penalty * geometric_mean

def calculate_rouge_scores(reference, candidate):
    """
    Calculate ROUGE scores for a given reference and candidate text.
    
    :param reference: str, the reference text
    :param candidate: str, the candidate text generated by the model
    :return: dict, containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

def calculate_quality_metrics(reference, candidate):
    """
    Calculate all quality metrics for a given reference and candidate text.
    
    :param reference: str, the reference text
    :param candidate: str, the candidate text generated by the model
    :return: dict, containing BLEU and ROUGE scores
    """
    bleu_score = calculate_bleu_score(reference, candidate)
    rouge_scores = calculate_rouge_scores(reference, candidate)
    
    return {
        'bleu': bleu_score,
        **rouge_scores
    }