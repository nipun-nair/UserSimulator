## Evaluation Metrics

Each task employs specific metrics designed to capture different aspects of human behavioral fidelity:

1. **ItemsTalk**: Item entropy quantifying diversity in movie selections. Let `x` be the list of movies:
   ```
   H(X) = - ∑₁ⁿ p(xᵢ) log p(xᵢ)
   ```

2. **BinPref**: Pearson correlation between acceptance rates `x` and movie ratings `y`:
   ```
   r = ∑₁ⁿ (xᵢ - x̄)(yᵢ - ȳ) / √[∑₁ⁿ (xᵢ - x̄)² ∑₁ⁿ (yᵢ - ȳ)²]
   ```

3. **OpenPref**: Aspect and sentiment entropy measuring review richness. Aspect list is denoted by `x`, sentiment list by `y`:
   ```
   H_aspect(X) = - ∑₁ⁿ p(xᵢ) log p(xᵢ)
   H_sentiment(Y) = - ∑₁ᵐ p(yᵢ) log p(yᵢ)
   ```

4. **RecRequest**: Multiple diversity measures. Let `w` be word vectors and `s` be sentence vectors:
   ```
   TTR = |Types| / |Tokens|

   W2V = 1 - (1 / |Words|²) ∑ᵢⱼ cos(wᵢ, wⱼ)

   SentDiv = 1 - (1 / |Sentences|²) ∑ᵢⱼ cos(sᵢ, sⱼ)
   ```
