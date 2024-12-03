Folder structure:
    - utils: Auxiliar files.
    - plots: Images and plots obtained
    - datasets: Datasets used in the codes

To obtain preprocessed dataset ("support2_cleaned"), please execute the following command:
- julia loadPackages.jl (only if some package is missing)
- julia dataCleaning.jl

Once obtained the cleaned dataset, to execute the binary and multiclass approaches, please
execute the following command:

- Binary: julia mainBin.jl
- Binary with feature selection: julia mainBinDimReduction.jl
- Categorical: julia mainCat.jl
- Categorical with feature selection: julia mainCatDimReduction.jl

More details can be found in the notebooks:
    - support2Analysis.ipynb
    - support2Bin.ipynb
    - support2BinDimReduction.ipynb
    - support2Cat.ipynb
    - support2CatDimReduction.ipynb