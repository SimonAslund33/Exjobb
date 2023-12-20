library(fgsea)
library(data.table)
library(ggplot2)
library(GSA)
library(tidyverse)
library(RColorBrewer) # for a colourful plot
library(ggrepel) # for nice annotations
library("readxl")
Flat = read.table("C:/Users/Simon/Downloads/Simon.shape.comparison.Flatness.top.bottom.fifteen.RNASeq.plaque.txt", header = TRUE, sep = "")
Elong = read.table("C:/Users/Simon/Downloads/Simon.shape.comparison.Elongation.top.bottom.fifteen.RNASeq.plaque.txt", header = TRUE, sep = "")
Spher = read.table("C:/Users/Simon/Downloads/Simon.shape.comparison.Sphericity.top.bottom.fifteen.RNASeq.plaque.txt", header = TRUE, sep = "")
TopZeroCalc = read_excel("C:/Users/Simon/Downloads/BikeTopvsZero.xlsx")
TopBotCalc = read_excel("C:/Users/Simon/Downloads/BikeTopvsBot.xlsx")
#foldchange = Flat$fc
#foldchange2 = lapply(foldchange, log2)

#Volcano plots
#pval = (-1)*(log10(Flat$fc))
#X = log2(Flat$p)
Flat$padj = p.adjust(Flat$p, method = "fdr", n = length(Flat$p))
Elong$padj = p.adjust(Elong$p, method = "fdr", n = length(Elong$p))
Spher$padj = p.adjust(Spher$p, method = "fdr", n = length(Spher$p))
TopZeroCalc$padj = p.adjust(TopZeroCalc$p, method = "fdr", n = length(TopZeroCalc$p))
TopBotCalc$padj = p.adjust(TopBotCalc$p, method = "fdr", n = length(TopBotCalc$p))
#ggplot(data = Flat, aes(x = log2(fc.), y = -log10(p.))) +
#  geom_point()
# Add threshold lines
theme_set(theme_classic(base_size = 20) +
            theme(
              axis.title.y = element_text(face = "bold", margin = margin(0,20,0,0), size = rel(1.1), color = 'black'),
              axis.title.x = element_text(hjust = 0.5, face = "bold", margin = margin(20,0,0,0), size = rel(1.1), color = 'black'),
              plot.title = element_text(hjust = 0.5)
            ))

# Add a column to the data frame to specify if they are UP- or DOWN- regulated (log2fc respectively positive or negative)<br /><br /><br />
Elong$diffexpressed <- "NO"
Flat$diffexpressed <- "NO"
Spher$diffexpressed <- "NO"
# if log2Foldchange > 0.6 and pvalue < 0.05
Elong$diffexpressed[log2(Elong$fc) > 0.0 & -log10(Elong$p) > 1.3] <- "UP"
Flat$diffexpressed[log2(Flat$fc) > 0.0 & -log10(Flat$p) > 1.3] <- "UP"
Spher$diffexpressed[log2(Spher$fc) > 0.0 & -log10(Spher$p) > 1.3] <- "UP"
# if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"<br /><br /><br />
Elong$diffexpressed[log2(Elong$fc) < -0.0 & -log10(Elong$p) > 1.3] <- "DOWN"
Flat$diffexpressed[log2(Flat$fc) < -0.0 & -log10(Flat$p) > 1.3] <- "DOWN"
Spher$diffexpressed[log2(Spher$fc) < -0.0 & -log10(Spher$p) > 1.3] <- "DOWN"
#<p># Explore a bit<br /><br /><br />

TopZeroCalc$diffexpressed <- "NO"
TopZeroCalc$diffexpressed[log2(TopZeroCalc$fc) > 0.0 & -log10(TopZeroCalc$p) > 1.3] <- "UP"
TopZeroCalc$diffexpressed[log2(TopZeroCalc$fc) < -0.0 & -log10(TopZeroCalc$p) > 1.3] <- "DOWN"

TopBotCalc$diffexpressed <- "NO"
TopBotCalc$diffexpressed[log2(TopBotCalc$fc) > 0.0 & -log10(TopBotCalc$p) > 1.3] <- "UP"
TopBotCalc$diffexpressed[log2(TopBotCalc$fc) < -0.0 & -log10(TopBotCalc$p) > 1.3] <- "DOWN"

head(Elong[order(Elong$padj) & Elong$diffexpressed == 'DOWN', ])
head(Flat[order(Flat$padj) & Flat$diffexpressed == 'DOWN', ])
head(Spher[order(Spher$padj) & Spher$diffexpressed == 'DOWN', ])
head(TopZeroCalc[order(TopZeroCalc$padj) & TopZeroCalc$diffexpressed == 'DOWN', ])

# Create a new column "delabel" to de, that will contain the name of the top 30 differentially expressed genes (NA in case they are not)
#Flat$delabel <- ifelse(Flat$gene %in% head(Flat[order(Flat$padj), "gene"], 10), Flat$gene, NA)
Flat$delabel <- ifelse(-log10(Flat$p) > 2.6, Flat$gene, Flat$delabel)
Flat$delabel <- ifelse(abs(log2(Flat$fc)) > 3.8 & -log10(Flat$p) > 1.3, Flat$gene, Flat$delabel)
Elong$delabel <- ifelse(-log10(Elong$p) > 2.6, Elong$gene, Elong$delabel)
Elong$delabel <- ifelse(abs(log2(Elong$fc)) > 4 & -log10(Elong$p) > 1.3, Elong$gene, Elong$delabel)
Spher$delabel <- Spher$delabel <- ifelse(-log10(Spher$p) > 2.6, Spher$gene, Spher$delabel)
Spher$delabel <- ifelse(abs(log2(Spher$fc)) > 4 & -log10(Spher$p) > 1.3, Spher$gene, Spher$delabel)
TopZeroCalc$delabel <- ifelse(-log10(TopZeroCalc$p) > 2.5, TopZeroCalc$feature, NA)
TopZeroCalc$delabel <- ifelse(abs(log2(TopZeroCalc$fc)) > 4 & -log10(TopZeroCalc$p) > 1.3, TopZeroCalc$feature, TopZeroCalc$delabel)
TopBotCalc$delabel <- ifelse(-log10(TopBotCalc$p) > 3, TopBotCalc$feature, NA)
TopBotCalc$delabel <- ifelse(abs(log2(TopBotCalc$fc)) > 4 & -log10(TopBotCalc$p) > 1.3, TopBotCalc$feature, TopBotCalc$delabel)
#TopZeroCalc$delabel <- ifelse(-log10(TopZeroCalc$p) > 1.3, TopZeroCalc$feature, NA)
ggplot(data = TopBotCalc, aes(x = log2(fc), y = -log10(p), col = diffexpressed, label = delabel)) +
  geom_vline(xintercept = c(-0.0, 0.0), col = "gray", linetype = 'dashed') +
  geom_hline(yintercept = 1.3, col = "gray", linetype = 'dashed') + 
  geom_point(size = 2) + 
  scale_color_manual(values = c("#00AFBB", "grey", "#bb0c00"), # to set the colours of our variable  
                     labels = c("Downregulated", "Not significant", "Upregulated")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)
  coord_cartesian(ylim = c(0, 5), xlim = c(-10, 10)) + # since some genes can have minuslog10padj of inf, we set these limits
  labs(color = 'Severe', #legend_title, 
       x = expression("log"[2]*"FC"), y = expression("-log"[10]*"p-value")) + 
  scale_x_continuous(breaks = seq(-10, 10, 2)) + # to customise the breaks in the x axis
  ggtitle('Top vs Bot calc Proportion') + # Plot title 
  geom_text_repel(max.overlaps = Inf) # To show all labels 


#GSEA



library(clusterProfiler)

 

# Convert data frame to a named vector 
#Spher$diffexpressed[log2(Spher$fc) < -0.0 & -log10(Spher$p) > 1.3] <- "DOWN"
#[-log10(Flat$p) > 1.3]
#geneListFlat %>% 
#  filter(p >= 0,05)
#Flat_treshhold <- Flat$p > 0.05
Flat_treshhold <- Flat %>% 
  filter(-log10(p) > 1.3)
Flat_Up <- Flat_treshhold %>% 
  filter(diffexpressed == "DOWN")
Elong_treshhold <- Elong %>% 
  filter(-log10(p) > 1.3)
Elong_Up <- Elong_treshhold %>% 
  filter(diffexpressed == "UP")
Spher_treshhold <- Spher %>% 
  filter(-log10(p) > 1.3)
Spher_Up <- Spher_treshhold %>% 
  filter(diffexpressed == "DOWN")
geneListFlat <- setNames(Flat_treshhold$fc, Flat_treshhold$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListFlat <- sort(geneListFlat, decreasing = TRUE)

geneListElong <- setNames(Elong_treshhold$fc, Elong_treshhold$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListElong <- sort(geneListElong, decreasing = TRUE)

geneListSpher <- setNames(Spher_treshhold$fc, Spher_treshhold$gene)  #geneList + runif(length(geneList), min=-1e-6, max=1e-6)  

# Sort geneList in decreasing order  
geneListSpher <- sort(geneListSpher, decreasing = TRUE)

#########################################################################################

all_gene_sets = msigdbr::msigdbr(species = "Homo sapiens", category= 'C2', subcategory ='CP:KEGG' )

msigdbr_t2g = all_gene_sets  %>%  dplyr::distinct(gs_name, gene_symbol) %>% as.data.frame()


resultsFlat <- GSEA(geneList = geneListFlat, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 
resultsElong <- GSEA(geneList = geneListElong, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 
resultsSpher <- GSEA(geneList = geneListSpher, TERM2GENE = msigdbr_t2g,                 pvalueCutoff = 1) 


resultsFlat %>% as_tibble() -> hallmark_resultsFlat
resultsElong %>% as_tibble() -> hallmark_resultsElong
resultsSpher %>% as_tibble() -> hallmark_resultsSpher

dotplot(resultsFlat)
cnetplot(resultsFlat)
dotplot(resultsElong)
cnetplot(resultsElong)
dotplot(resultsSpher)
cnetplot(resultsSpher)
