import random

class RulesToNatural:
    def __init__(self, predRange, antecedentLabels):
        self.predictionRange = predRange
        self.antecedents = antecedentLabels
        self.antecedentTerms = ["very low ", "low ", "middling ", "high ", "very high "]
        self.consequentTerms = ["strong sell ", "sell ", "hold ", "buy ", "strong buy "]
        self.equalTerms = ["is ", "shows to be ", "can be deduced as "]
        self.connectiveTerms = ["and ", "furthermore ", "also "]
        self.initialTerm = ["the "]
        self.resultingTerms = ["indicating a ", "leading to a "]

    def getPredictiveStatement(self, prediction):
        min_range = self.predictionRange[0]
        max_range = self.predictionRange[1]

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
        return words[random.randint(0, len(words)-1)];

    def getEqualTerm(self):
        return self.getRand(self.equalTerms)

    def getConnectiveTerm(self):
        return self.getRand(self.connectiveTerms)

    def getInitialTerm(self):
        return self.getRand(self.initialTerm)

    def getResultingTerm(self):
        return self.getRand(self.resultingTerms)

    def explainPrediction(self, rules, prediction):

        statement = ""
        for rule in rules.values():
            statement += self.getInitialTerm()
            for idx, antecedentMF in enumerate(rule['antecedent']):
                if antecedentMF == -1:
                    continue

                statement += self.antecedents[idx] + self.getEqualTerm() + self.antecedentTerms[antecedentMF]
                if idx < len(rule['antecedent'])-1:
                    statement += self.getConnectiveTerm()

            statement += self.getResultingTerm() + self.consequentTerms[rule['consequent']] + ". "

        print(self.getPredictiveStatement(prediction) + statement + "\n")
