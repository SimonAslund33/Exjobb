
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
import statsmodels.stats.multitest as mc
from pandas import read_table
from gsea_api.expression_set import ExpressionSet
from gsea_api.gsea import GSEADesktop
from gsea_api.molecular_signatures_db import GeneSets
#from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase

import gseapy as gp
#import gsea_api
#names = gp.get_library_name()
#print(names)
#human = gp.get('Human')
#print(human[:10])
#from gseapy.gsea import 
#"C:\Users\Simon\Downloads\msigdb_v7.1_files_to_download_locally\msigdb"
from gsea_api.gsea import GSEApy
from gsea_api.molecular_signatures_db import MolecularSignaturesDatabase

#reactome_pathways = GeneSets.from_gmt('ReactomePathways.gmt')
#msigdb = MolecularSignaturesDatabase('C:\msigdb\msigdb_v7.1_GMTs')
#msigdb = MolecularSignaturesDatabase(r"C:\msigdb\msigdb_v7.1_GMTs", version=7.1)
#kegg_pathways = msigdb.load('c2.cp.kegg', 'symbols')
#print(msigdb.gene_sets)
#print(gp.get_library_name())
#gsea = GSEApy()
#print(gp.__version__)

def load_file(path):
    #xls = pd.read_table(path)
    xls = pd.read_table(path, delim_whitespace=True)
    #print(xls)
    #df1 = pd.read_excel(xls, 'file')
    return xls



def biological_analysis(file):


    
    #data = filename.dropna()
    #data_values = data.iloc[:,0:-1]
    #data_values = data_values.dropna()
    #file = pd.DataFrame(file)
   # data = pd.read_table(file, delim_whitespace=True)
    #print(type(file))
    #file = (file[file["fc"]])
    X = file["fc"]
    y = file["p"]
    
    y_fdr = mc.fdrcorrection(y, alpha=0.05, method='indep', is_sorted=False)
    #print(sorted(y_fdr[1])[0:20])
    #print(sorted(y_fdr[1])[-20:-1])
    #labeledfiles = (file["p"]< 0.001)
    #labels = labeledfiles["gene"]
    rank = []
    #print(file.iloc[0,3])
    for i in range(len(file["p"])):
        
        rank.append(np.sign(file.iloc[i,3])*(-1)*np.log10(file.iloc[i,2]))
        if ((-1)*np.log10(file.iloc[i,2])) < 2.5 and abs((np.log2(file.iloc[i,3]))) < 5:
            file.iloc[i,1] = None
    #print(len(rank))
    file['Rank'] = rank
    #a = file[["p"] > 0.001]
    print(file)
    #a['gene'] = None
    y = (-1)*(np.log10(y))
    X = np.log2(X)
    #file["fc"]< 100)
    #sortedP = np.sort(abs(X))
    #PrankTop = sorted(sortedP[-20:], reverse=True)
    #PrankBot = sortedP[0:10]
    ranking = file[["gene", "Rank"]]
    print(ranking.shape)
    print(ranking)
    print(ranking["Rank"])
    print(ranking["Rank"].shape)
    #gene_sets=gp.get_library_name(name='MSigDB_Hallmark_2020')
    #print(gene_sets)
    #pre_res = gp.prerank(rnk=ranking["gene"],gene_sets='MSigDB_Hallmark_2020', seed=6)
    """
    pre_res = gp.prerank(rnk=ranking, # or rnk = rnk,
                     gene_sets='MSigDB_Hallmark_2020',
                     #threads=4,
                     min_size=5,
                     max_size=1000,
                     permutation_num=1000, # reduce number to speed up testing
                     outdir=None, # don't write to disk
                     seed=6,
                     verbose=True, # see what's going on behind the scenes
                    )
    """
    #pre_res.results
    #print(PrankTop)
    #print(PrankBot)
    #print(max(X))
    #print(np.where(X == max(X)))
    plt.figure()
    #binwidth = 0.05
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #plt.title('Elongation')
    #ax1.hist(y, range=[0,1],bins=20)
    #ax2.hist(y_fdr[1], range=[0,1], bins=20)
    #colors = ["orange", "navy"]
    #lw = 2
    #plt.hist(y_fdr[1], range=[0,1])
    #plt.hist(y_fdr[1], range=[0,1])
    # plotting second histogram
    #plt.hist(X_r2[y=="AS"],color="navy")
    labels = file["gene"]
    
    plt.scatter(X,y,s=0.3,)
    for X, y, s in zip(X, y, labels):
        plt.text(X, y, s, fontsize=7)
    
    

    #plt.legend(loc="best", shadow=False, scatterpoints=1)
    #plt.title("PCA of dataset")
    #plt.figtext(0.01, 0.95, "explained variance ratio (first two components): %s"
    plt.show()
#msig = Msigdb()
# mouse hallmark gene sets
#gmt = msig.get_gmt(category='mh.all', dbver="2023.1.Mm")
#msigdb = MolecularSignaturesDatabase('msigdb', version=7.1)
#msigdb.gene_sets
#reactome_pathways = GeneSets.from_gmt('ReactomePathways.gmt')

#gsea = GSEADesktop()
#msig = Msigdb()
file_Flat = load_file(r"C:\Users\Simon\Downloads\Simon.shape.comparison.Flatness.top.bottom.fifteen.RNASeq.plaque.txt")
file_Elong = load_file(r"C:\Users\Simon\Downloads\Simon.shape.comparison.Elongation.top.bottom.fifteen.RNASeq.plaque.txt")
file_Spher = load_file(r"C:\Users\Simon\Downloads\Simon.shape.comparison.Sphericity.top.bottom.fifteen.RNASeq.plaque.txt")
#print(file_Elong["gene"].head())
#gene_list = file_Elong["gene"]
#glist = gene_list.squeeze().str.strip().to_list()
#print(glist)
#gene_sets='MSigDB_Hallmark_2020'

#enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
#                 gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'],
                 #organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
#                 outdir=None, # don't write to disk
#                )
biological_analysis(file_Elong)