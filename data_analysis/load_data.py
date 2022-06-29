import numpy as np
import copy


def load_full_subject_ID_data(trimmed_path):
    '''
    Loads each datapoint including the subject ID, doesn't treat as timeseries. Doesn't include perturbations
    '''

    f = open(trimmed_path)
    f.readline() # skip header

    results_dict = {}

    for line in f:
        items = line.split(',')
        timepoint = int(items[1])
        subject_ID = items[2]
        taxas = [int(items[i]) for i in range(22, 185)]

        if subject_ID not in results_dict.keys():
            results_dict[subject_ID] = []

        results_dict[subject_ID].append((timepoint,taxas))

    return results_dict

def get_absent_flags(results_dict):
    '''
    returns flags that equal 1 if a species is absent across all samples
    '''
    # check to see if any taxa are always 0

    flags = [1] * 163

    for subject in results_dict.keys():
        for taxa in results_dict[subject]:

            for i, t in enumerate(taxa[1]):

                if t > 0:
                    flags[i] = 0

    return flags

def remove_absent_species(results_dict, flags):
    flags = np.array(flags)
    new_dict = copy.deepcopy(results_dict)

    for subject in results_dict.keys():
        for i, taxa in enumerate(results_dict[subject]):


            new_taxa  = (taxa[0], np.array(taxa[1])[np.where(1-flags)[0]])

            new_dict[subject][i] = new_taxa

    return new_dict


def load_subject_ID_dict(trimmed_path = '/Users/neythen/Desktop/Projects/gMLV/data/maria_multiomics/processed/trimmed.csv'):
    results_dict = load_full_subject_ID_data(trimmed_path)
    flags = get_absent_flags(results_dict)
    new_results_dict = remove_absent_species(results_dict,flags)

    return new_results_dict


def z_scale(X):
    '''
    scales data by z = (x-mu)/sigma for each feature
    '''

    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)

    return X

def normalise(X):
    print(np.max(X, axis=0).shape)
    print(X.shape)
    X = X/np.max(X, axis = 0).reshape(1,-1)

    return X

def get_X_subjects(results_dict):
    '''
    get the design matrix X and the corresponding subjects list
    '''

    # extract the taxa data ordered by subject and time
    X = []
    subjects = []  # the subject and number of datapoints belonging to that subject

    count = 0
    for subject in results_dict.keys():
        subjects.append((subject, count, count + len(results_dict[subject])))
        X.extend([d[1] for d in sorted(results_dict[subject])])
        count += len(results_dict[subject])
    X = np.array(X)

    return X, subjects

def load_perturbations_dict(pert_path = '/Users/neythen/Desktop/Projects/gMLV/data/maria_multiomics/metadata/perturbations.csv', unique_perts = ['Flu', 'Amik', 'antifungals', 'Metro', 'Vanc', 'Gent', 'Benz', 'Taz', 'Rifam', 'Aug']):
    f = open(pert_path, 'r')

    f.readline()


    results_dict = {}

    for line in f:
        items = line.split(',')
        try:
            subject_ID = items[0].strip('"')
            pert = unique_perts.index(items[1].strip('"'))
            start = int(items[2])
            end = int(items[3])

            if subject_ID not in results_dict:
                results_dict[subject_ID] = []

            results_dict[subject_ID].append((pert, start, end))
        except:
            pass



    return results_dict



def combine_taxa_pert_dicts(taxa_dict, pert_dict):


    combined_dict = {}

    for subject_ID in taxa_dict.keys():
        for timepoint, taxas in taxa_dict[subject_ID]:

            current_perts = []

            if subject_ID in pert_dict.keys():

                for pert, start, end in pert_dict[subject_ID]:
                    if start <= timepoint <= end:
                        current_perts.append(pert)

            if subject_ID not in combined_dict:
                combined_dict[subject_ID] = []
            if subject_ID == 'M10':
                print(current_perts)

            combined_dict[subject_ID].append((timepoint, taxas, current_perts))

    return combined_dict


def multihot(perts, n_unique = 10):
    mh = np.zeros((n_unique,))


    if len(perts) > 0:
        mh[np.array(perts)] = 1

    return mh

def get_X_P_subjects(combined_dict, subject_IDs = False):
    '''
    get the design matrix X, perturbation amtrix P and the corresponding subjects list from the combined taxa and perturbation dict

    if subject_IDs argument is supplied (list of strings) gets data for those IDs, otherwise gets data for all subjects
    '''

    # extract the taxa data ordered by subject and time
    X = []
    P = []
    subjects = []  # the subject and number of datapoints belonging to that subject

    count = 0

    if not subject_IDs:
        subject_IDs = combined_dict.keys()

    for subject in subject_IDs:
        subjects.append((subject, count, count + len(combined_dict[subject])))
        X.extend([d[1] for d in sorted(combined_dict[subject])])

        P.extend([multihot(d[2]) for d in sorted(combined_dict[subject])])

        count += len(combined_dict[subject])
    X = np.array(X)
    P = np.array(P)

    return X, P, subjects


def get_subject_status(metadata_path =  '/Users/neythen/Desktop/Projects/gMLV/data/maria_multiomics/metadata/data_subject_metadata.csv'):
    f = open(metadata_path)

    f.readline()
    results_dict = {}
    for line in f:
        items = line.split(',')

        subject_ID = items[0].strip('"').upper()
        clss = items[13].strip('"')
        outcome = items[17].strip('"')

        #if clss != 'NA' or outcome != 'NA':
        results_dict[subject_ID] = {'class': clss, 'outcome':outcome}

    return results_dict

def combine_subject_dicts(subjects, subject_status):

    results_dict = {}

    for subject_ID, start, end in subjects:

        if subject_ID in subject_status:

            clss = subject_status[subject_ID]['class']
            outcome = subject_status[subject_ID]['outcome']
        else:
            clss = 'NA'
            outcome = 'NA'


        results_dict[subject_ID] = {'start':start, 'end':end, 'class':clss, 'outcome': outcome}
    return results_dict