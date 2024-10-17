import numpy as np
import sys

class ModelComparison:
    def __init__(self, results_A, results_B):
        self.results_16 = results_A
        self.results_63 = results_B

    def bayes_factor(self, logz_i, logz_j):
        return logz_i - logz_j

    def interpret_evidence(self, lnB_ij):
        if lnB_ij > 5:
            return "Strong evidence for model i"
        elif 2.5 < lnB_ij <= 5:
            return "Moderate evidence for model i"
        elif 1 < lnB_ij <= 2.5:
            return "Weak evidence for model i"
        elif -1 <= lnB_ij <= 1:
            return "Inconclusive"
        elif -2.5 <= lnB_ij < -1:
            return "Weak evidence for model j"
        elif -5 <= lnB_ij < -2.5:
            return "Moderate evidence for model j"
        else:
            return "Strong evidence for model j"

    def aic(self, logz, num_params):
        return 2 * num_params - 2 * logz

    def bic(self, logz, num_params, n_data):
        return np.log(n_data) * num_params - 2 * logz

    def compare_aic_bic(self, models, n_data):
        aic_scores = {}
        bic_scores = {}

        for model_name, result in models.items():
            logz = result['logz']
            num_params = result['num_params']
            aic_scores[model_name] = self.aic(logz, num_params)
            bic_scores[model_name] = self.bic(logz, num_params, n_data)

        best_aic_model = min(aic_scores, key=aic_scores.get)
        best_bic_model = min(bic_scores, key=bic_scores.get)

        print("\nResultados do AIC:")
        for model, score in aic_scores.items():
            print(f"{model}: AIC = {score:.2f}")
        print(f"Melhor modelo segundo AIC: {best_aic_model}")

        print("\nResultados do BIC:")
        for model, score in bic_scores.items():
            print(f"{model}: BIC = {score:.2f}")
        print(f"Melhor modelo segundo BIC: {best_bic_model}")

    def compare_models(self, results, n_data):
        models = list(results.keys())
        print(f"Comparando {len(models)} modelos...")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_i = models[i]
                model_j = models[j]
                lnB_ij = self.bayes_factor(results[model_i]['logz'], results[model_j]['logz'])
                interpretation = self.interpret_evidence(lnB_ij)
                print(f"{model_i} vs {model_j}: lnB_ij = {lnB_ij:.2f} -> {interpretation}")

                if interpretation == "Inconclusive":
                    print("\nFator de Bayes inconclusivo. Comparando AIC e BIC...")
                    self.compare_aic_bic(results, n_data)

    def run_comparisons(self, n_data_16, n_data_63, save_to_file=None):
        """Executa as comparações e salva em um arquivo, se especificado."""
        if save_to_file:
            # Redireciona a saída para o arquivo
            original_stdout = sys.stdout
            with open(save_to_file, 'w') as f:
                sys.stdout = f  # Redireciona o print para o arquivo
                self._execute_comparisons(n_data_16, n_data_63)
                sys.stdout = original_stdout  # Restaura a saída padrão
        else:
            # Executa normalmente, imprimindo na tela
            self._execute_comparisons(n_data_16, n_data_63)

    def _execute_comparisons(self, n_data_16, n_data_63):
        """Executa as comparações para 16 e 63 FRBs."""
        print("Comparações para 16 FRBs:")
        self.compare_models(self.results_16, n_data_16)
        print("\nComparações para 63 FRBs:")
        self.compare_models(self.results_63, n_data_63)
