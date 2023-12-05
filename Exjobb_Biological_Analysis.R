
library(fgsea)
library(data.table)
library(ggplot2)
library(GSA)
library(tidyverse)

Flat = read.table("C:/Users/simon/Downloads/Top15Bot15FlatnessBio.txt", header = TRUE, sep = ",")
Elong = read.table("C:\Users\simon\Downloads\Top15Bot15ElongationBio.txt", header = TRUE, sep = ",")
Spher = read.table("C:/Users/simon/Downloads/Top15Bot15SphericityBio.txt", header = TRUE, sep = ",")
foldchange = Flat$fc
foldchange2 = lapply(foldchange, log2)
gmt_file <-GSA.read.gmt(paste("C:/Users/Simon/Downloads/",
                              "h.all.v2023.2.Hs.symbols.gmt",
                              sep = "/"))
gmt_file2 <-GSA.read.gmt(paste("C:/Users/Simon/Downloads/",
                              "h.all.v2023.2.Hs.entrez.gmt",
                              sep = "/"))

library(clusterProfiler)

 

# Convert data frame to a named vector 

geneList <- setNames(Flat$fc, Flat$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneList <- sort(geneList, decreasing = TRUE)

#########################################################################################

all_gene_sets = msigdbr::msigdbr(species = "Homo sapiens", category= 'H')

msigdbr_t2g = all_gene_sets  %>%  dplyr::distinct(gs_name, gene_symbol) %>% as.data.frame()


results <- GSEA(geneList = geneList, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 


results %>% as_tibble() -> hallmark_results
