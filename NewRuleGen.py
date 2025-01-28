import torch.nn as nn
import torch
import membershipFunctions as mf
import itertools
from typing import List
import ANFIS as anfis


def getMaxMembershipFunctionIndex(values: torch.Tensor) -> int:
    return torch.argmax(values).item()


def generateAntecedentPermutations(number_of_features: int) -> List[List[int]]:
    return list(map(list, itertools.product([0, -1], repeat=number_of_features)))


def generateIndependentFeaturePermutations(independent_features: List[int], number_of_features) -> List[List[int]]:
    return [[-1 if r != i else 0 for r in range(1, number_of_features + 1)] for i in independent_features]


def ruleGeneration(x_data, x_ranges, y_data, y_range, model, injection_rule_base):
    X = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    num_epochs = 2
    learning_rate = 0.02
    batch_size = 10

    num_input_mfs = 5
    num_output_mfs = 5

    minimum_rule_degree = 0.95

    num_features = len(x_ranges)

    input_membership_functions = nn.ModuleList()
    for value_range in x_ranges:
        params = mf.generateBellParams(num_input_mfs, value_range)
        input_mf = mf.TrainableBellMF(params)
        input_membership_functions.append(input_mf)

    params = mf.generateTrapezoidParams(num_output_mfs, y_range)
    output_membership_function = mf.TrainableTrapezoidMF(params)

    # Fuzzify the data using the input and output membership functions
    fuzzy_data = []
    for x_row, y_row in zip(x_data, y_data):
        fuzzy_row = [input_membership_functions[idx](
            torch.tensor([value], dtype=torch.float32))
            for idx, value in enumerate(x_row)]
        fuzzy_row.append(output_membership_function(torch.tensor([y_row], dtype=torch.float32)))
        fuzzy_data.append(fuzzy_row)

    antecedent_permutations = generateAntecedentPermutations(num_features)
    # remove permutation that is just -1s
    antecedent_permutations.pop(len(antecedent_permutations) - 1)

    print("Antecedent permutation count : {0}".format(len(antecedent_permutations)))

    # remove permutations based on feature dependencies
    # if a feature is independent then it should only be use in its own rule
    permutation_storage = []
    independent_features = [2, 8]  # not zero indexed
    independent_feature_permutations = generateIndependentFeaturePermutations(independent_features, num_features)
    if independent_features:
        for perm in antecedent_permutations:
            valid_perm = True
            for idx, ind_feature in enumerate(independent_features):
                if perm[ind_feature - 1] == 0 and perm != tuple(independent_feature_permutations[idx]):
                    valid_perm = False
                    break
            if valid_perm:
                permutation_storage.append(perm)

    permutation_storage = permutation_storage + independent_feature_permutations

    print("Antecedent permutation count after independence filter : {0}".format(len(permutation_storage)))

    # if two or more features are dependant then they should always be used together
    antecedent_permutations = []
    dependant_feature_relations = [(1, 3)]  # not zero indexed
    for perm in permutation_storage:
        valid_perm = True
        for relation in dependant_feature_relations:
            previous = perm[relation[0] - 1]
            valid_relation = True
            for feature in relation:
                if perm[feature - 1] != previous:
                    valid_relation = False
                    break
                previous = perm[feature - 1]

            if not valid_relation:
                valid_perm = False
                break

        if valid_perm:
            antecedent_permutations.append(perm)

    print("Antecedent permutation count after dependence filter : {0}".format(len(antecedent_permutations)))

    # After feature relations have been processed
    # generate initial and complement rule base

    # rule base structure
    #   {antecedent_permutation: rule}
    initial_rule_base = {}

    # complement rule base structure
    #   {antecedent_permutation: {consequent_membership_function_index: rule}}
    complement_rule_base = {}

    # Rules that do not have a certain degree will be placed here
    discarded_rule_base = {}

    sample_count = 1
    for sample in fuzzy_data:
        # index of the consequent mf with the highest membership value
        consequent_index = getMaxMembershipFunctionIndex(sample[-1])
        # value of the consequent mf with the highest membership value
        consequent_value = sample[-1][consequent_index]

        #print("processed sample {0}".format(sample_count))
        #sample_count += 1
        #rule_count = 0

        for perm in antecedent_permutations:
            # will be used as a rule base key, holding the antecedent mf with the highest membership value
            # unused features will be represented as -1
            antecedent_indexes = perm.copy()
            # will be used to calculate rule degree
            antecedent_values = []

            for ant_idx, antecedent in enumerate(perm):
                if antecedent == -1:
                    continue

                # index of the antecedent mf with the highest membership value
                antecedent_indexes[ant_idx] = getMaxMembershipFunctionIndex(sample[ant_idx])
                # value of the antecedent mf with the highest membership value
                antecedent_values.append(sample[ant_idx][antecedent_indexes[ant_idx]].item())

            antecedent_indexes = tuple(antecedent_indexes)
            rule_degree = torch.prod(torch.tensor(antecedent_values + [consequent_value]))

            # create the rule instance
            rule = {
                'antecedent': antecedent_indexes,  # antecedant indexes
                'consequent': consequent_index,  # consequent indexes
                'degree': rule_degree  # rule degree, can increase degree to make rule more predominant in the rule base
            }

            #rule_count += 1

            # complement rules will have the same structure
            # however the complement rule base will store a dictionary within each entry
            # this inner base used consequent indexes as keys

            key = antecedent_indexes

            if key in initial_rule_base:
                if rule_degree > initial_rule_base[key]['degree']:
                    initial_rule = initial_rule_base[key]
                    if key in complement_rule_base:
                        complement_key = initial_rule['consequent']
                        if complement_key in complement_rule_base[key]:
                            if complement_rule_base[key][complement_key]['degree'] < initial_rule['degree']:
                                complement_rule_base[key][complement_key] = initial_rule
                            elif key in discarded_rule_base:
                                rules = discarded_rule_base[key]
                                rules.append(rule)
                                discarded_rule_base[key] = rules
                            else:
                                discarded_rule_base[key] = [rule]
                        else:
                            complement_rule_base[key][complement_key] = initial_rule
                    else:
                        complement_key = initial_rule['consequent']
                        inner_base = {complement_key: initial_rule}
                        complement_rule_base[key] = inner_base
                    initial_rule_base[key] = rule
                # add to complement rule base
                elif rule_degree >= minimum_rule_degree:
                    if key in complement_rule_base:
                        complement_key = rule['consequent']
                        if complement_key in complement_rule_base[key]:
                            if complement_rule_base[key][complement_key]['degree'] < rule_degree:
                                complement_rule_base[key][complement_key] = rule
                            elif key in discarded_rule_base:
                                rules = discarded_rule_base[key]
                                rules.append(rule)
                                discarded_rule_base[key] = rules
                            else:
                                discarded_rule_base[key] = [rule]
                        else:
                            complement_rule_base[key][complement_key] = rule
                    else:
                        complement_key = rule['consequent']
                        inner_base = {complement_key: rule}
                        complement_rule_base[key] = inner_base
                # discard the rule
                elif key in discarded_rule_base:
                    rules = discarded_rule_base[key]
                    rules.append(rule)
                    discarded_rule_base[key] = rules
                else:
                    discarded_rule_base[key] = [rule]

            else:
                if rule_degree >= minimum_rule_degree:
                    initial_rule_base[key] = rule
                else:
                    if key in discarded_rule_base:
                        rules = discarded_rule_base[key]
                        rules.append(rule)
                        discarded_rule_base[key] = rules
                    else:
                        discarded_rule_base[key] = [rule]

        #print("processed {0} rules".format(rule_count))

    print("Initial rule base count : {0} ".format(len(initial_rule_base)))

    complement_rule_base_count = 0
    for inner_base in complement_rule_base.values():
        complement_rule_base_count += len(inner_base)
    print("Complement rule base count : {0}".format(complement_rule_base_count))

    discarded_rule_base_count = 0
    for inner_base in discarded_rule_base.values():
        discarded_rule_base_count += len(inner_base)
    print("Discarded rule base count : {0}".format(discarded_rule_base_count))

    print("Selection stage ...")

    # selection stage
    # rules are swapped for their complement rules and the model is re-tested
    # if the complement rule results in lower error then it will remain

    # train model with rule base
    model.train_model(train_data=X,
                      train_labels=y,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      rule_base=initial_rule_base)
    model_loss = anfis.getModelLoss(model, X, y)
    print("Initial model loss : {0}".format(model_loss))
    altered_rule_base = initial_rule_base
    for key in altered_rule_base.keys():
        # if rule antecedent permutation is not in complement rb then continue
        if key not in complement_rule_base:
            continue

        # swap initial rule with every complement rule and train
        for complement_key in complement_rule_base[key].keys():
            initial_rule = altered_rule_base[key]
            complement_rule = complement_rule_base[key][complement_key]
            altered_rule_base[key] = complement_rule
            model.train_model(train_data=X,
                              train_labels=y,
                              num_epochs=2,
                              learning_rate=learning_rate,
                              batch_size=batch_size,
                              rule_base=altered_rule_base)
            complement_loss = anfis.getModelLoss(model, X, y)
            # if complement rule gives higher loss then switch back the rule
            if complement_loss >= model_loss:
                altered_rule_base[key] = initial_rule
            else:
                print("Complement remains")
                print("Complement loss : {0}".format(complement_loss))
                model_loss = complement_loss

    print("Reduction stage ...")

    # Reduction stage
    # remove each rule from the rule base
    # if the loss is decreased then it is permanently removed
    model.train_model(train_data=X,
                      train_labels=y,
                      num_epochs=2,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      rule_base=altered_rule_base)
    model_loss = anfis.getModelLoss(model, X, y)
    print("Initial model loss : {0}".format(model_loss))
    keys = altered_rule_base.keys()
    rule_storage = altered_rule_base.copy()
    for key in keys:
        rule = altered_rule_base[key]
        rule_storage.pop(key)
        model.train_model(train_data=X,
                          train_labels=y,
                          num_epochs=2,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          rule_base=rule_storage)
        reduced_model_loss = anfis.getModelLoss(model, X, y)
        if reduced_model_loss > model_loss:
            rule_storage[key] = rule
        else:
            print("Reduction stands")
            print("Reduced model loss : {0}".format(reduced_model_loss))
            model_loss = reduced_model_loss

    altered_rule_base = rule_storage

    print("Rule base count after reduction : {0}".format(len(altered_rule_base)))

    print("Injection stage ...")

    # Injection stage
    for key in injection_rule_base.keys():
        if key not in altered_rule_base:
            altered_rule_base[key] = injection_rule_base[key]

    final_rule_base = altered_rule_base
    model.train_model(train_data=X,
                      train_labels=y,
                      num_epochs=5,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      rule_base=final_rule_base)
    final_loss = anfis.getModelLoss(model, X, y)
    print("Final model loss : {0}".format(final_loss))

    return model