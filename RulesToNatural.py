import random

import GenerativeResponse as gen

class RulesToNatural:
    def __init__(self, predRange, antecedentLabels):
        self.prediction_range = predRange
        self.antecedents = antecedentLabels
        self.antecedent_terms = ["very low ", "low ", "middling ", "high ", "very high "]
        self.consequent_terms = ["strong sell ", "sell ", "hold ", "buy ", "strong buy "]
        self.equal_terms = ["is ", "shows to be ", "can be deduced as "]
        self.connective_terms = ["and ", "furthermore ", "also "]
        self.initial_term = ["the "]
        self.resulting_terms = ["indicating a ", "leading to a "]
        self.rule_significance = ["very dominant ", "dominant ", "neutral dominance", "weak dominance", "very dominant"]

    def getPredictiveStatement(self, prediction):
        min_range = self.prediction_range[0]
        max_range = self.prediction_range[1]

        predictiveTerm = ""

        if prediction > 0.5 * max_range:
            predictiveTerm = "strong buy "
        elif prediction >= 0.25 * max_range:
            predictiveTerm = "buy "
        elif prediction >= 0.25 * min_range:
            predictiveTerm = "hold "
        elif prediction < 0.25 * min_range:
            predictiveTerm = "sell "
        elif prediction < 0.5 * min_range:
            predictiveTerm = "strong sell "

        predictiveStatement = ("The model has predicted a change of {0:.4f}% indicating a {1}, "
                               "this can be derived from the following rules. ".format(prediction, predictiveTerm))

        return predictiveStatement

    def getRand(self, words):
        return words[random.randint(0, len(words)-1)]

    def getEqualTerm(self):
        return self.getRand(self.equal_terms)

    def getConnectiveTerm(self):
        return self.getRand(self.connective_terms)

    def getInitialTerm(self):
        return self.getRand(self.initial_term)

    def getResultingTerm(self):
        return self.getRand(self.resulting_terms)

    def getSignificance(self, firing):
        if firing >= 0.8:
            return self.rule_significance[0]
        elif firing >= 0.6:
            return self.rule_significance[1]
        elif firing >= 0.4:
            return self.rule_significance[2]
        elif firing >= 0.2:
            return self.rule_significance[3]
        else:
            return self.rule_significance[4]

    def explainPrediction(self, rules, prediction):

        statement = ""
        for rule in rules.values():
            statement += self.getInitialTerm()
            for idx, antecedentMF in enumerate(rule['antecedent']):
                if antecedentMF == -1:
                    continue

                statement += self.antecedents[idx] + self.getEqualTerm() + self.antecedent_terms[antecedentMF]
                if idx < len(rule['antecedent'])-1:
                    statement += self.getConnectiveTerm()

            statement += self.getResultingTerm() + self.consequent_terms[rule['consequent']] + ". "
            statement += "This rules firing strength suggest it was " + self.getSignificance(rule['firing'])

        response = self.getPredictiveStatement(prediction) + statement + "\n"

        print(response)
        print(gen.cleanResponse(response))
