# Gene values = Binary
# Population = Each population item is a combination of parameter values
# Population at any point of time is set of candidate solutions
# Each item in Population is a candidate solution. Each probable solution is also called as chromosome
# Genes = individual attribute/parameter value comprising of a population item
# Fitment function is also called as Objective or Cost function
# Cross Over is also called as Recombination
# Solution is found out over multiple generations
# Initial population is generally based on random values
# Genes can be called as Decision variables also

# Imports
import pandas as pd
import random
import numpy as np
import warnings

# Configs
genes = ['gene_0', 'gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5']
cross_over_prob = 0.5
num_chromosomes = 10
return_chromosomes = 2
cross_over_type = "single"
mutation_prob = 0.1


def create_initial_population():
     index = np.array(range(0, num_chromosomes))
     random_list = []
     l1 = len(genes)
     #random.seed(100)
     for i in range(0, l1 * num_chromosomes):
         random_list.append(random.randint(0, 1))

     data = np.array(random_list).reshape(num_chromosomes, l1)
     df = pd.DataFrame(data=data, columns=genes, index=index)
     return df


def get_basic_fitment_values(df):
    # Maximize the fitness value
    df['fitness_value'] = (df[genes[0]] + df[genes[1]]) * (df[genes[2]] + \
                          df[genes[3]]) + 26 * df[genes[4]] + 30 * df[genes[5]]
    return df


def get_population_indexes(df):
    i = 0
    population_index_list = []
    while i < return_chromosomes:
        l = df.shape[0]
        random_number = round(random.random(), 2)
        j = l - 1
        while random_number > df['cum_sum_value'][j]:
            j = j - 1
        if j not in population_index_list:
            print("random number {} = {}".format(i + 1, random_number))
            population_index_list.append(j)
            i = i + 1
    print("")
    return population_index_list


def get_scaled_fitment_values(df, scale_fitment=False):
    if scale_fitment:
        const1 = abs(min(df['fitness_value']))

        # to avoid division by 0 (if all chromosomes have same negative value, sum of fitness values will be 0)
        const2 = 1
        df['fitness_value'] = df['fitness_value'] + const1 + const2

    sum_fitness_values = np.sum(df['fitness_value'])
    df['scaled_fitness_value'] = df['fitness_value'] / sum_fitness_values
    df.sort_values(['scaled_fitness_value'], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def roulette_selection(df):
    df = get_basic_fitment_values(df)

    if min(df['fitness_value']) < 0:
        df1 = get_scaled_fitment_values(df, True)
    else:
        df1 = get_scaled_fitment_values(df, False)

    df1['cum_sum_value'] = 0
    for i in range(0, df1.shape[0]):
        cum_sum_val = 0
        for j in range(i, df.shape[0]):
            cum_sum_val = cum_sum_val + df1['scaled_fitness_value'][j]

        df1['cum_sum_value'].iloc[i] = cum_sum_val

    population_index_list = get_population_indexes(df1)
    print("Selected Indexes after Roulette Wheel selection are = ", population_index_list, "\n")

    selected_parents = df.iloc[population_index_list]
    selected_parents.reset_index(inplace=True, drop=True)

    return selected_parents


def single_point_cross_over(selected_parents):
    n_genes = len(genes)
    lower_bound = 1
    upper_bound = n_genes - 2
    cross_over_pt = random.randint(lower_bound, upper_bound)
    print("Cross over point = ", cross_over_pt)

    children = pd.DataFrame(columns=genes)

    # Child 1
    children.loc[0, genes[0]:genes[cross_over_pt]] = selected_parents.loc[0, genes[0]:genes[cross_over_pt]]
    children.loc[0, genes[cross_over_pt + 1]:genes[n_genes - 1]] = selected_parents.loc[1,
                                                                   genes[cross_over_pt + 1]:genes[n_genes - 1]]

    # Child 2
    children.loc[1, genes[0]:genes[cross_over_pt]] = selected_parents.loc[1, genes[0]:genes[cross_over_pt]]
    children.loc[1, genes[cross_over_pt + 1]:genes[n_genes - 1]] = selected_parents.loc[0,
                                                                   genes[cross_over_pt + 1]: genes[n_genes - 1]]

    return children


def double_point_cross_over(selected_parents):
    n_genes = len(genes)
    lower_bound = 1
    upper_bound = n_genes - 2
    cross_over_pt1 = random.randint(lower_bound, upper_bound)
    cross_over_pt2 = random.randint(lower_bound, upper_bound)
    while cross_over_pt2 <= cross_over_pt1:
        cross_over_pt2 = random.randint(lower_bound, upper_bound)

    print("Cross over point 1 = ", cross_over_pt1)
    print("Cross over point 2 = ", cross_over_pt2)

    children = pd.DataFrame(columns=genes)

    # Child 1
    children.loc[0, genes[0]:genes[cross_over_pt1]] = selected_parents.loc[0, genes[0]:genes[cross_over_pt1]]
    children.loc[0, genes[cross_over_pt1+1]:genes[cross_over_pt2]] = selected_parents.loc[1, genes[cross_over_pt1+1]:
                                                                                             genes[cross_over_pt2]]
    children.loc[0, genes[cross_over_pt2 + 1]:genes[n_genes - 1]] = selected_parents.loc[0,
                                                                   genes[cross_over_pt2 + 1]:genes[n_genes - 1]]

    # Child 2
    children.loc[1, genes[0]:genes[cross_over_pt1]] = selected_parents.loc[1, genes[0]:genes[cross_over_pt1]]
    children.loc[1, genes[cross_over_pt1+1]:genes[cross_over_pt2]] = selected_parents.loc[0, genes[cross_over_pt1+1]:
                                                                                             genes[cross_over_pt2]]
    children.loc[1, genes[cross_over_pt2 + 1]:genes[n_genes - 1]] = selected_parents.loc[1,
                                                                   genes[cross_over_pt2 + 1]:genes[n_genes - 1]]

    return children


def cross_over(parents):
    if cross_over_type=="single":
        children = single_point_cross_over(parents)
    elif cross_over_type=="double":
        children = double_point_cross_over(parents)
    else:
        print("Cross Over type not correct !")
        children = None

    print("Children before applying cross over probability -")
    print(children)
    print("")

    r1 = np.round(random.random(), 2)
    r2 = np.round(random.random(), 2)
    print("r1 prob = ", r1)
    print("r2 prob = ", r2)
    print("")

    print("Children after applying cross over probability - ")

    if r1 > cross_over_prob:
        print("parent 0 is passed instead of child 0")
        children.loc[0] = parents.loc[0]

    if r2 > cross_over_prob:
        print("parent 1 is passed instead of child 1")
        children.loc[1] = parents.loc[1]

    for i in range(0, len(genes)):
        children[genes[i]] = children[genes[i]].astype(int)

    print(children)
    return children


def mutation(children):
    for i in range(0, children.shape[0]):
          for j in range(0, len(genes)):
                r = np.round(random.random(), 2)
                if r <= mutation_prob:
                    print("Random Num {} : Chromosome {} : Gene {} is mutated".format(r, i, j))
                    children.loc[i, genes[j]] = 1 - children.loc[i, genes[j]]

    return children


def elitism():
    pass


if __name__ == "__main__":
    print("<---- In main function ---->\n")
    warnings.filterwarnings("ignore")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 10)

    initial_population = create_initial_population()

    new_population = pd.DataFrame(columns=genes)

    for i in range(0, num_chromosomes, 2):
        selected_parents = roulette_selection(initial_population)
        print("Selected Parents")
        print(selected_parents)
        children = cross_over(selected_parents)
        mutated_children = mutation(children)
        new_population.loc[i] = mutated_children.loc[0]
        new_population.loc[i+1] = mutated_children.loc[1]

    print("")
    print("New population")
    print(new_population)