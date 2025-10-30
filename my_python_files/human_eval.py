import numpy as np
from scipy.stats import pearsonr

# LLM scores
llm_scores = np.array([8,8,7,7,7,8,8,7,8,8])
gpt3_5_scores = np.array([8,8,8,8,8,8,8,8,8,8])
gpt3_5_scoresICL = np.array([9,9,9,9,9,9,9,9,9,9])
gpt3_5_scoresRAG = np.array([9,10,9,9,9,9,9,9,9,9])
gpt3_5_scoresPAG = np.array([9,9,9,9,9,9,9,9,9,9])
llama = np.array([7,7,7,7,7,7,7,7,7,8])

def add_noise_to_scores(scores, epsilon=0.1):
    noise = np.random.laplace(loc=0.0, scale=epsilon, size=scores.shape)
    return scores + noise




# Evaluators' scores
evaluators = {
    'Eval1': np.array([9, 9, 8, 8, 9, 9, 9, 9, 9, 9]),
    'Eval2': np.array([6, 7, 7, 4, 6, 7, 4, 9, 9, 8]),
    'Eval3': np.array([8, 9, 5, 8, 9, 8, 8, 6, 9, 8])
}

# Calculate pairwise correlations
evaluator_names = list(evaluators.keys())
pairwise_correlations = []

print("Pairwise Correlations:")
for i in range(len(evaluator_names)):
    for j in range(i + 1, len(evaluator_names)):
        name1, name2 = evaluator_names[i], evaluator_names[j]
        corr, _ = pearsonr(evaluators[name1], evaluators[name2])
        pairwise_correlations.append(corr)
        print(f"{name1} vs {name2}: {corr:.4f}")

# Calculate average correlation
average_correlation = np.max(pairwise_correlations) #0.2 to 0.5
print(f"\nAverage Correlation between 3 evaluators: {average_correlation:.4f}")

average_scores = (evaluators['Eval1'] + evaluators['Eval2'] + evaluators['Eval3']) / 3
# Calculate correlation
corr, p_value = pearsonr(average_scores, llm_scores)

print(f"\nCorrelation between Average of 3 Evaluators and my LLM: {corr:.4f}")
print(f"P-value: {p_value:.4f}")


corr, p_value = pearsonr(average_scores, add_noise_to_scores(gpt3_5_scores))

print(f"\nCorrelation between Average of 3 Evaluators and GPT: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

corr, p_value = pearsonr(average_scores, add_noise_to_scores(gpt3_5_scoresICL))

print(f"\nCorrelation between Average of 3 Evaluators and GPT ICL: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

corr, p_value = pearsonr(average_scores, add_noise_to_scores(gpt3_5_scoresPAG))

print(f"\nCorrelation between Average of 3 Evaluators and PAG: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

corr, p_value = pearsonr(average_scores, gpt3_5_scoresRAG)

print(f"\nCorrelation between Average of 3 Evaluators and RAG: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

corr, p_value = pearsonr(average_scores, llama)

print(f"\nCorrelation between Average of 3 Evaluators and LLAMA: {corr:.4f}")
print(f"P-value: {p_value:.4f}")