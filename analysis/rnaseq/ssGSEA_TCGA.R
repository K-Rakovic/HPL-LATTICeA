library(tidyverse)
library(ggrepel)
library(DESeq2)
library(GSVA)
library(circlize)
library(ComplexHeatmap)
library(pheatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')

tcga_tpm <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD_star_tpm_symbols.csv") %>% 
  column_to_rownames(var = "gene")

tcga_malignant <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Documents\\PhD_Workspace\\HE_LUAD_5x_paper\\tcga_agg_by_sum_5sc.csv") %>% 
  dplyr:::select(c("samples", "Hot, cohesive", "Hot, discohesive", "HPC 13", "Cold, discohesive", 'Cold, cohesive')) %>% 
  filter(., samples %in% colnames(tcga_tpm)) %>% 
  arrange(samples) %>% 
  column_to_rownames(., var = "samples")

tcga_cluster_labs <- as.data.frame(names(tcga_malignant))
colnames(tcga_cluster_labs) <- c("Supercluster")

tcga_tpm_filtered <- tcga_tpm %>% 
  select(rownames(tcga_malignant)) %>%
  mutate(keep = rowMeans(. >=1) >=0.2) %>% 
  filter(keep) %>% 
  dplyr:::select(-keep)

###

msigdb_hallmark <- loadDB("h.all.v2024.1.Hs.symbols.gmt")

ssgsea_param <- ssgseaParam(expr = as.matrix(tcga_tpm_filtered),
                            geneSets = msigdb_hallmark)

ssgsea_results <- gsva(ssgsea_param,
                       BPPARAM = SnowParam(workers = detectCores() - 1))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

Heatmap(
  ssgsea_results %>% pheatmap:::scale_rows(),
  show_column_names = FALSE,
  col = col_colors
)

write.csv(ssgsea_results, file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\HALLMARKS_ssGSEA_TCGA_250424.csv")

### 

go_cc <- loadDB("GO_Cellular_Component_2023.txt")

ssgsea_param <- ssgseaParam(expr = as.matrix(tcga_tpm_filtered),
                            geneSets = go_cc)

ssgsea_results <- gsva(ssgsea_param,
                       BPPARAM = SnowParam(workers = detectCores() - 1))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

Heatmap(
  ssgsea_results %>% pheatmap:::scale_rows(),
  show_column_names = FALSE,
  col = col_colors
)

write.csv(ssgsea_results, file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\GO_CC_ssGSEA_TCGA_250424.csv")

###

go_bp <- loadDB("GO_Biological_Process_2023.txt")

ssgsea_param <- ssgseaParam(expr = as.matrix(tcga_tpm_filtered),
                            geneSets = go_bp)

ssgsea_results <- gsva(ssgsea_param,
                       BPPARAM = SnowParam(workers = detectCores() - 1))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

Heatmap(
  ssgsea_results %>% pheatmap:::scale_rows(),
  show_column_names = FALSE,
  col = col_colors
)

write.csv(ssgsea_results, file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\GO_BP_ssGSEA_TCGA_250424.csv")

###

kegg <- loadDB("KEGG_2021_Human.txt")

ssgsea_param <- ssgseaParam(expr = as.matrix(tcga_tpm_filtered),
                            geneSets = kegg)

ssgsea_results <- gsva(ssgsea_param,
                       BPPARAM = SnowParam(workers = detectCores() - 1))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

Heatmap(
  ssgsea_results %>% pheatmap:::scale_rows(),
  show_column_names = FALSE,
  col = col_colors
)

write.csv(ssgsea_results, file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\KEGG_ssGSEA_TCGA_250424.csv")
