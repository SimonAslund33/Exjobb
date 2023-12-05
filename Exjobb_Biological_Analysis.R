library(fgsea)
library(data.table)
library(ggplot2)
library(GSA)
library(tidyverse)
library(RColorBrewer) # for a colourful plot
library(ggrepel) # for nice annotations

Flat = read.table("C:/Users/simon/Downloads/Top15Bot15FlatnessBio.txt", header = TRUE, sep = "")
Elong = read.table("C:/Users/simon/Downloads/Top15Bot15ElongationBio.txt", header = TRUE, sep = "")
Spher = read.table("C:/Users/simon/Downloads/Top15Bot15SphericityBio.txt", header = TRUE, sep = "")
#foldchange = Flat$fc
#foldchange2 = lapply(foldchange, log2)

#Volcano plots
#pval = (-1)*(log10(Flat$fc))
#X = log2(Flat$p)

#ggplot(data = Flat, aes(x = log2(fc.), y = -log10(p.))) +
#  geom_point()
# Add threshold lines
theme_set(theme_classic(base_size = 20) +
            theme(
              axis.title.y = element_text(face = "bold", margin = margin(0,20,0,0), size = rel(1.1), color = 'black'),
              axis.title.x = element_text(hjust = 0.5, face = "bold", margin = margin(20,0,0,0), size = rel(1.1), color = 'black'),
              plot.title = element_text(hjust = 0.5)
            ))

# Create a new column "delabel" to de, that will contain the name of the top 30 differentially expressed genes (NA in case they are not)
df$delabel <- ifelse(Flat$gene. %in% head(Flat[order(Flat$padj), "gene_symbol"], 30), df$gene_symbol, NA)

ggplot(data = Flat, aes(x = log2(fc.), y = -log10(p.), col = diffexpressed, label = delabel)) +
  geom_vline(xintercept = c(-0.6, 0.6), col = "gray", linetype = 'dashed') +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') + 
  geom_point(size = 2) + 
  scale_color_manual(values = c("#00AFBB", "grey", "#bb0c00"), # to set the colours of our variable  
                     labels = c("Downregulated", "Not significant", "Upregulated")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)
  coord_cartesian(ylim = c(0, 250), xlim = c(-10, 10)) + # since some genes can have minuslog10padj of inf, we set these limits
  labs(color = 'Severe', #legend_title, 
       x = expression("log"[2]*"FC"), y = expression("-log"[10]*"p-value")) + 
  scale_x_continuous(breaks = seq(-10, 10, 2)) + # to customise the breaks in the x axis
  ggtitle('Thf-like cells in severe COVID vs healthy patients') + # Plot title 
  geom_text_repel(max.overlaps = Inf) # To show all labels 


#GSEA

gmt_file <-GSA.read.gmt(paste("C:/Users/Simon/Downloads/",
                              "h.all.v2023.2.Hs.symbols.gmt",
                              sep = "/"))
gmt_file2 <-GSA.read.gmt(paste("C:/Users/Simon/Downloads/",
                              "h.all.v2023.2.Hs.entrez.gmt",
                              sep = "/"))

library(clusterProfiler)

 

# Convert data frame to a named vector 

geneListFlat <- setNames(Flat$fc, Flat$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListFlat <- sort(geneListFlat, decreasing = TRUE)

geneListElong <- setNames(Elong$fc, Elong$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListElong <- sort(geneListElong, decreasing = TRUE)

geneListSpher <- setNames(Spher$fc, Spher$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListSpher <- sort(geneListSpher, decreasing = TRUE)

#########################################################################################

all_gene_sets = msigdbr::msigdbr(species = "Homo sapiens", category= 'H')

msigdbr_t2g = all_gene_sets  %>%  dplyr::distinct(gs_name, gene_symbol) %>% as.data.frame()


resultsFlat <- GSEA(geneList = geneListFlat, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 
resultsElong <- GSEA(geneList = geneListElong, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 
resultsSpher <- GSEA(geneList = geneListSpher, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 


resultsFlat %>% as_tibble() -> hallmark_resultsFlat
resultsElong %>% as_tibble() -> hallmark_resultsElong
resultsSpher %>% as_tibble() -> hallmark_resultsSpher

