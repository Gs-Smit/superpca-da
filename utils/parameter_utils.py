def super_pca_parameters(database):
    # type: (str) -> (int, int)
    """
    set the parameters in the super pca
    """
    if database not in ['Indian', 'PaviaU']:
        raise ValueError(f"this dataset {database} is not supported")
    if database == 'Indian':
        num_PC = 30  # THE OPTIMAL PCA DIMENSION.
        num_Pixels = 30  # The value of Sf
    elif database == 'PaviaU':
        num_PC = 5  # THE OPTIMAL PCA DIMENSION.
        num_Pixels = 15  # The value of Sf
    return num_PC, num_Pixels
