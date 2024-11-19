import numpy as np
import sys

"""class ModelComparison:

    def __init__(self, samples):
        
        Initializes the class with a dictionary of samples.
        Examples of `samples`: 
        {
            "Sample_1": {"results": {...}, "n_data": 16},
            "Sample_2": {"results": {...}, "n_data": 63},
            ...
        }
        
        self.samples = samples

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

        print("\nAIC Results:")
        for model, score in aic_scores.items():
            print(f"{model}: AIC = {score:.2f}")
        print(f"Best-model according to AIC: {best_aic_model}")

        print("\nBIC Results:")
        for model, score in bic_scores.items():
            print(f"{model}: BIC = {score:.2f}")
        print(f"Best-model according to BIC: {best_bic_model}")

    def compare_models(self, results, n_data):
        models = list(results.keys())
        print(f"Comparing {len(models)} models...")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_i = models[i]
                model_j = models[j]
                lnB_ij = self.bayes_factor(results[model_i]['logz'], results[model_j]['logz'])
                interpretation = self.interpret_evidence(lnB_ij)
                print(f"{model_i} vs {model_j}: lnB_ij = {lnB_ij:.2f} -> {interpretation}")

                if interpretation == "Inconclusive":
                    print("\nBayes Factor inconclusive. Comparing AIC and BIC...")
                    self.compare_aic_bic(results, n_data)

    def run_comparisons(self, save_to_file=None):
        #Performs comparisons for all samples and saves to a file if specified.
        if save_to_file:
            original_stdout = sys.stdout
            with open(save_to_file, 'w') as f:
                sys.stdout = f
                self._execute_comparisons()
                sys.stdout = original_stdout
        else:
            self._execute_comparisons()

    def _execute_comparisons(self):
        #Performs comparisons for all provided samples.
        for sample_name, data in self.samples.items():
            print(f"\nComparisons for {sample_name}:")
            results = data['results']
            n_data = data['n_data']
            self.compare_models(results, n_data)"""

class ModelComparison:

    def __init__(self, samples):
        """
        Initializes the class with a dictionary of samples.
        Examples of `samples`: 
        {
            "Sample_1": {"results": {...}},
            "Sample_2": {"results": {...}},
            ...
        }
        """
        self.samples = samples

    def bayes_factor(self, logz_i, logz_j):
        """Calculates the Bayes Factor between two models."""
        return logz_i - logz_j

    def interpret_evidence(self, lnB_ij):
        """Interprets the strength of the evidence based on the Bayes factor."""
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
        """Calculates the Akaike Information Criterion (AIC)."""
        return 2 * num_params - 2 * logz

    def compare_aic(self, models):
        """Compares models using AIC scores."""
        aic_scores = {}
        for model_name, result in models.items():
            logz = result['logz']
            num_params = result['num_params']
            aic_scores[model_name] = self.aic(logz, num_params)

        best_aic_model = min(aic_scores, key=aic_scores.get)

        print("\nAIC Results:")
        for model, score in aic_scores.items():
            print(f"{model}: AIC = {score:.2f}")
        print(f"Best model according to AIC: {best_aic_model}")

    def compare_models(self, results):
        """Compares models using Bayes factor and AIC when needed."""
        models = list(results.keys())
        print(f"Comparing {len(models)} models...")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_i = models[i]
                model_j = models[j]
                lnB_ij = self.bayes_factor(results[model_i]['logz'], results[model_j]['logz'])
                interpretation = self.interpret_evidence(lnB_ij)
                print(f"{model_i} vs {model_j}: lnB_ij = {lnB_ij:.2f} -> {interpretation}")

                if interpretation == "Inconclusive":
                    print("\nBayes Factor inconclusive. Comparing AIC...")
                    self.compare_aic(results)

    def run_comparisons(self, save_to_file=None):
        """Performs comparisons for all samples and saves to a file if specified."""
        if save_to_file:
            original_stdout = sys.stdout
            with open(save_to_file, 'w') as f:
                sys.stdout = f
                self._execute_comparisons()
                sys.stdout = original_stdout
        else:
            self._execute_comparisons()

    def _execute_comparisons(self):
        """Performs comparisons for all provided samples."""
        for sample_name, data in self.samples.items():
            print(f"\nComparisons for {sample_name}:")
            results = data['results']
            self.compare_models(results)


class SaveResults:
    """
    A class to save fitted parameters from MCMC results to a .txt file.
    """

    def __init__(self, output_file):
        """
        Initializes the SaveResults class.

        Parameters:
        - output_file: Name of the .txt file where results will be saved.
        """
        self.output_file = output_file

    def save_to_txt(self, mcsample, model_name):
        """
        Saves the mean values and uncertainties of the fitted parameters to the .txt file.

        Parameters:
        - mcsample: MCSamples object containing the model samples.
        - model_name: Name of the model (e.g., 'wCDM_16').
        """
        # Calculate mean values and uncertainties (68% confidence interval)
        stats = mcsample.getMargeStats()

        with open(self.output_file, 'a') as f:
            f.write(f"Results for {model_name}:\n")
            f.write("-" * 40 + "\n")

            # Extract parameter names as strings
            param_names = [param.name for param in mcsample.paramNames.names]

            for param in param_names:
                mean = stats.parWithName(param).mean
                lower, upper = stats.parWithName(param).limits[0].lower, stats.parWithName(param).limits[0].upper
                f.write(f"{param}: {mean:.4f} (+{upper - mean:.4f}, -{mean - lower:.4f})\n")
            f.write("\n")


    def reset_file(self):
        """Clears the content of the output file."""
        with open(self.output_file, 'w') as f:
            f.write("")  # Write an empty string to clear the file.


class CompareMCMCResults:
    """
    A class to compare uncertainties between two MCMC models using the formula
    for Relative Error Reduction and save the results to a .txt file.
    """

    def __init__(self, output_file):
        """
        Initializes the CompareMCMCResults class.

        Parameters:
        - output_file: Name of the .txt file where comparison results will be saved.
        """
        self.output_file = output_file

    def compare_errors(self, mcsample1, mcsample2, model1_name, model2_name):
        """
        Compares the uncertainties between two MCMC models using the 
        Relative Error Reduction formula.

        Parameters:
        - mcsample1: MCSamples object for the first model.
        - mcsample2: MCSamples object for the second model.
        - model1_name: Name of the first model (e.g., 'Model 1').
        - model2_name: Name of the second model (e.g., 'Model 2').
        """
        # Extract marginal statistics from the MCSamples objects
        stats_model1 = mcsample1.getMargeStats()
        stats_model2 = mcsample2.getMargeStats()

        with open(self.output_file, 'a') as f:
            f.write(f"Comparing {model1_name} vs {model2_name}:\n")
            f.write("-" * 40 + "\n")

            param_names = [param.name for param in stats_model1.names]

            for param in param_names:
                central_value1 = stats_model1.parWithName(param).mean
                central_value2 = stats_model2.parWithName(param).mean
                std1 = stats_model1.parWithName(param).err
                std2 = stats_model2.parWithName(param).err

                model1_lower = abs(stats_model1.parWithName(param).limits[0].lower)
                model2_lower = abs(stats_model2.parWithName(param).limits[0].lower)

                model1_upper = abs(stats_model1.parWithName(param).limits[0].upper)
                model2_upper = abs(stats_model2.parWithName(param).limits[0].upper)

                # Calculate relative error reductions
                reduction_lower = ((model1_lower - model2_lower) / model1_lower) * 100
                reduction_upper = ((model1_upper - model2_upper) / model1_upper) * 100

                # Calculate % difference in central values
                diff_central = ((central_value2 - central_value1) / central_value1) * 100

                f.write(f"{param}:\n")
                f.write(f"  % Difference in Central Value: {diff_central:.2f}%\n")
        
                # Write results to file
                f.write(f"  Lower Error Reduction: {reduction_lower:.2f}%\n")
                f.write(f"  Upper Error Reduction: {reduction_upper:.2f}%\n")

                # Print interpretation of the reduction results
                f.write("  Interpretation:\n")
                if reduction_lower > 0:
                    f.write(f"    Lower: {model2_name} has lower uncertainty (improvement).\n")
                else:
                    f.write(f"    Lower: {model2_name} has higher uncertainty (worse performance).\n")

                if reduction_upper > 0:
                    f.write(f"    Upper: {model2_name} has lower uncertainty (improvement).\n")
                else:
                    f.write(f"    Upper: {model2_name} has higher uncertainty (worse performance).\n")

                # Calculate and write relative uncertainties
                rel_uncertainty1 = (std1 / abs(central_value1)) * 100
                rel_uncertainty2 = (std2 / abs(central_value2)) * 100

                f.write(f"  Relative Uncertainty ({model1_name}): {rel_uncertainty1:.2f}%\n")
                f.write(f"  Relative Uncertainty ({model2_name}): {rel_uncertainty2:.2f}%\n")
            f.write("\n")

    def reset_file(self):
        """Clears the content of the output file."""
        with open(self.output_file, 'w') as f:
            f.write("")  # Write an empty string to clear the file.





