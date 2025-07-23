library(tidyverse)
library(ggrepel)
library(DESeq2)
library(GSVA)
library(circlize)
library(ComplexHeatmap)
library(pheatmap)
library(exCITingpath)

hallmarks_ssgsea <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\HALLMARKS_ssGSEA_TCGA_250424.csv") %>% 
  column_to_rownames(var = "...1")

tcga_tpm <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD_star_tpm_symbols.csv")

tcga_malignant <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Documents\\PhD_Workspace\\HE_LUAD_5x_paper\\tcga_agg_by_sum_5sc.csv") %>% 
  dplyr:::select(c("samples", "Hot, cohesive", "Hot, discohesive", "HPC 13", "Cold, discohesive", 'Cold, cohesive')) %>% 
  filter(., samples %in% colnames(tcga_tpm)) %>% 
  arrange(samples) %>% 
  column_to_rownames(., var = "samples")

tcga_malignant <- tcga_malignant[colnames(hallmarks_ssgsea), ]

tcga_cluster_labs <- as.data.frame(names(tcga_malignant))
colnames(tcga_cluster_labs) <- c("Supercluster")

###

hallmarks_corr <- cor(t(hallmarks_ssgsea),
                      tcga_malignant,
                      method = "spearman")

colnames(hallmarks_corr) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(hallmarks_corr) <- sub("^HALLMARK_", "", rownames(hallmarks_corr))

col_colors <- colorRamp2(c(-0.4, 0, 0.4), c("#4E79A7", "white", "#E15759"))

p <- Heatmap(hallmarks_corr,
             col = col_colors,
             border = TRUE,
             name = "Spearman\ncorrelation",
             row_dend_width = unit(4, "mm"),
             column_dend_height = unit(4, "mm"))  

pdf(file="C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA_HALLMARKS_WSICORR_250424.pdf", width = 6, height = 10.2)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 40), "mm"))
dev.off()

hex_cols <- c("#F77189", "#36ADA4", "#FC8D62", "#A48CF5","#97A431")
names(hex_cols) <- colnames(hallmarks_corr)

col_anno <- HeatmapAnnotation(
  foo = anno_simple(
    x     = colnames(hallmarks_corr),,   # dummy values
    col   = hex_cols,                        # your colours
    border = FALSE,
    width = unit(4, "mm")                   # height of the ba
  ),
  annotation_height = unit(4, "mm"),
  show_annotation_name = FALSE
)


p <- Heatmap(hallmarks_corr,
             col = col_colors,
             border = TRUE,
             name = "Spearman\ncorrelation",
             row_dend_width = unit(4, "mm"),
             column_dend_height = unit(4, "mm"),
             show_column_names = FALSE,
             top_annotation = col_anno)

pdf(file="C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA_HALLMARKS_WSICORR_250424_col_labs.pdf", width = 6, height = 10.2)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 40), "mm"))
dev.off()

### Add stats

compute_correlations_with_pvals <- function(matrix1, matrix2, corr_method) {
  # Initialize matrices for correlations and p-values
  n_rows <- ncol(matrix1)
  n_cols <- ncol(matrix2)
  cors <- matrix(NA, nrow = n_rows, ncol = n_cols)
  pvals <- matrix(NA, nrow = n_rows, ncol = n_cols)
  
  # Compute correlations and p-values
  for (i in 1:n_rows) {
    for (j in 1:n_cols) {
      test_result <- cor.test(matrix1[,i], matrix2[,j], method = corr_method)
      cors[i,j] <- test_result$estimate
      pvals[i,j] <- test_result$p.value
    }
  }
  
  # Adjust p-values for multiple testing
  # adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows)
  adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows, ncol = n_cols)
  
  return(list(correlations = cors, 
              raw_pvals = pvals, 
              adjusted_pvals = adjusted_pvals))
}

stat_results <- compute_correlations_with_pvals(t(hallmarks_ssgsea), tcga_malignant, corr_method = "spearman")
stat_cor <- stat_results$correlations
colnames(stat_cor) <- c("Hot cohesive", "Hot discohesive\n(low risk)", "Hot discohesive\n(high risk)", "Cold discohesive", "Cold cohesive")
rownames(stat_cor) <- sub("^HALLMARK_", "", rownames(hallmarks_ssgsea))

sig_matrix <- matrix("", nrow = nrow(stat_results$adjusted_pvals), ncol = ncol(stat_results$adjusted_pvals))
sig_matrix[stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[stat_results$adjusted_pvals < 0.01 & stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[stat_results$adjusted_pvals < 0.05 & stat_results$adjusted_pvals >= 0.01] <- "*"

p <- Heatmap(
  stat_cor,
  col = col_colors,
  border = TRUE,
  name = "Spearman\ncorrelation",
  row_dend_width = unit(4, "mm"),
  column_dend_height = unit(4, "mm"),
  show_column_names = FALSE,
  top_annotation = col_anno,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

pdf(file="C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA_HALLMARKS_WSICORR_250424_col_labs_stats.pdf", width = 6.2, height = 10.2)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 40), "mm"))
dev.off()

###

# Correlation with all pathways
gocc_ssgsea <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\GO_CC_ssGSEA_TCGA_250424.csv") %>% 
  column_to_rownames(var = "...1")

gocc_cor <- cor(t(gocc_ssgsea),
                tcga_malignant, 
                method = "spearman")

colnames(gocc_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gocc_cor) <- gsub("\\(.*?\\)", "", rownames(gocc_cor))

Heatmap(gocc_cor,
             col = col_colors,
             border = TRUE,
             name = "Spearman\ncorrelation",
             row_dend_width = unit(4, "mm"),
             column_dend_height = unit(4, "mm"))  

# Add in statistics
compute_correlations_with_pvals <- function(matrix1, matrix2, corr_method) {
  # Initialize matrices for correlations and p-values
  n_rows <- ncol(matrix1)
  n_cols <- ncol(matrix2)
  cors <- matrix(NA, nrow = n_rows, ncol = n_cols)
  pvals <- matrix(NA, nrow = n_rows, ncol = n_cols)
  
  # Compute correlations and p-values
  for (i in 1:n_rows) {
    for (j in 1:n_cols) {
      test_result <- cor.test(matrix1[,i], matrix2[,j], method = corr_method)
      cors[i,j] <- test_result$estimate
      pvals[i,j] <- test_result$p.value
    }
  }
  
  # Adjust p-values for multiple testing
  # adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows)
  adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows, ncol = n_cols)
  
  return(list(correlations = cors, 
              raw_pvals = pvals, 
              adjusted_pvals = adjusted_pvals))
}

gocc_stat_results <- compute_correlations_with_pvals(t(gocc_ssgsea), tcga_malignant, corr_method = "spearman")
gocc_stat_cor <- gocc_stat_results$correlations
colnames(gocc_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gocc_stat_cor) <- gsub("\\(.*?\\)", "", rownames(gocc_stat_cor))

sig_matrix <- matrix("", nrow = nrow(gocc_stat_results$adjusted_pvals), ncol = ncol(gocc_stat_results$adjusted_pvals))
sig_matrix[gocc_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.01 & gocc_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.05 & gocc_stat_results$adjusted_pvals >= 0.01] <- "*"

Heatmap(
  gocc_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

# Statistics with the top 10 most enriched pathways in LATTICeA

rownames(gocc_ssgsea) <- gsub("\\(.*?\\)", "", rownames(gocc_ssgsea))

gocc_ssgsea_top <- gocc_ssgsea %>% 
  rownames_to_column(var = "pathway") %>% 
  filter(pathway %in% c("Collagen-Containing Extracellular Matrix ", "Platelet Alpha Granule ", "Platelet Alpha Granule Lumen ", "Cell-Substrate Junction ",
                      "Focal Adhesion ", "Basement Membrane ", "Keratin Filament ", "Intermediate Filament ", "Lysosomal Lumen ", "Endoplasmic Reticulum Lumen ")) %>% 
  column_to_rownames(var = "pathway")

gocc_stat_results <- compute_correlations_with_pvals(t(gocc_ssgsea_top), tcga_malignant, corr_method = "spearman")
gocc_stat_cor <- gocc_stat_results$correlations
colnames(gocc_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gocc_stat_cor) <- rownames(gocc_ssgsea_top)

sig_matrix <- matrix("", nrow = nrow(gocc_stat_results$adjusted_pvals), ncol = ncol(gocc_stat_results$adjusted_pvals))
sig_matrix[gocc_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.01 & gocc_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.05 & gocc_stat_results$adjusted_pvals >= 0.01] <- "*"

p <- Heatmap(
  gocc_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  show_row_names = TRUE,
  border = TRUE,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

pdf(file="C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA_GOCC_top10_WSIcorr_stats_250424.pdf", width = 6, height = 4)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 20), "mm"))
dev.off()


###

go_cc <- loadDB("GO_Cellular_Component_2023.txt")

# Collagen

colnames(tcga_malignant) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
tcga_malignant_scaled <- tcga_malignant %>% 
  mutate(across(everything(), rescale)) %>% 
  arrange(`Cold discohesive`)

tcga_tpm_ <- tcga_tpm %>% 
  filter(rownames(.) %in% go_cc$`Collagen-Containing Extracellular Matrix (GO:0062023)`) %>% 
  dplyr::select(rownames(tcga_malignant_scaled))

column_ha <- HeatmapAnnotation(df = tcga_malignant_scaled)

Heatmap(tcga_tpm_ %>% pheatmap:::scale_rows(),
        top_annotation = column_ha,
        cluster_columns = FALSE)

gocc_stat_results <- compute_correlations_with_pvals(t(tcga_tpm_), tcga_malignant, corr_method = "spearman")
gocc_stat_cor <- gocc_stat_results$correlations
colnames(gocc_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gocc_stat_cor) <- rownames(tcga_tpm_)

sig_matrix <- matrix("", nrow = nrow(gocc_stat_results$adjusted_pvals), ncol = ncol(gocc_stat_results$adjusted_pvals))
sig_matrix[gocc_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.01 & gocc_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.05 & gocc_stat_results$adjusted_pvals >= 0.01] <- "*"

Heatmap(
  gocc_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  show_row_names = TRUE,
  border = TRUE,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

# Keratin filaments
tcga_tpm_ <- tcga_tpm %>% 
  filter(rownames(.) %in% go_cc$`Keratin Filament (GO:0045095)`) %>% 
  dplyr::select(rownames(tcga_malignant))

colnames(tcga_malignant) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
tcga_malignant_scaled <- tcga_malignant %>% 
  mutate(across(everything(), rescale)) %>% 
  arrange(`Cold discohesive`)

column_ha <- HeatmapAnnotation(df = tcga_malignant_scaled)

Heatmap(tcga_tpm_ %>% pheatmap:::scale_rows(),
        top_annotation = column_ha,
        cluster_columns = FALSE)

gocc_stat_results <- compute_correlations_with_pvals(t(tcga_tpm_), tcga_malignant, corr_method = "spearman")
gocc_stat_cor <- gocc_stat_results$correlations
colnames(gocc_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gocc_stat_cor) <- rownames(tcga_tpm_)

sig_matrix <- matrix("", nrow = nrow(gocc_stat_results$adjusted_pvals), ncol = ncol(gocc_stat_results$adjusted_pvals))
sig_matrix[gocc_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.01 & gocc_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gocc_stat_results$adjusted_pvals < 0.05 & gocc_stat_results$adjusted_pvals >= 0.01] <- "*"

Heatmap(
  gocc_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  show_row_names = TRUE,
  border = TRUE,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

###

# Correlation with all pathways
gobp_ssgsea <- read_csv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\GO_BP_ssGSEA_TCGA_250424.csv") %>% 
  column_to_rownames(var = "...1")

gobp_cor <- cor(t(gobp_ssgsea),
                tcga_malignant, 
                method = "spearman")

colnames(gobp_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gobp_cor) <- gsub("\\(.*?\\)", "", rownames(gobp_cor))

Heatmap(gobp_cor,
        col = col_colors,
        border = TRUE,
        name = "Spearman\ncorrelation",
        row_dend_width = unit(4, "mm"),
        column_dend_height = unit(4, "mm"))  

# Add in statistics
compute_correlations_with_pvals <- function(matrix1, matrix2, corr_method) {
  # Initialize matrices for correlations and p-values
  n_rows <- ncol(matrix1)
  n_cols <- ncol(matrix2)
  cors <- matrix(NA, nrow = n_rows, ncol = n_cols)
  pvals <- matrix(NA, nrow = n_rows, ncol = n_cols)
  
  # Compute correlations and p-values
  for (i in 1:n_rows) {
    for (j in 1:n_cols) {
      test_result <- cor.test(matrix1[,i], matrix2[,j], method = corr_method)
      cors[i,j] <- test_result$estimate
      pvals[i,j] <- test_result$p.value
    }
  }
  
  # Adjust p-values for multiple testing
  # adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows)
  adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows, ncol = n_cols)
  
  return(list(correlations = cors, 
              raw_pvals = pvals, 
              adjusted_pvals = adjusted_pvals))
}

gobp_stat_results <- compute_correlations_with_pvals(t(gobp_ssgsea), tcga_malignant, corr_method = "spearman")
gobp_stat_cor <- gobp_stat_results$correlations
colnames(gobp_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gobp_stat_cor) <- gsub("\\(.*?\\)", "", rownames(gobp_stat_cor))

sig_matrix <- matrix("", nrow = nrow(gobp_stat_results$adjusted_pvals), ncol = ncol(gobp_stat_results$adjusted_pvals))
sig_matrix[gobp_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gobp_stat_results$adjusted_pvals < 0.01 & gobp_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gobp_stat_results$adjusted_pvals < 0.05 & gobp_stat_results$adjusted_pvals >= 0.01] <- "*"

Heatmap(
  gobp_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

###

go_bp <- loadDB("GO_Biological_Process_2023.txt")

rownames(gobp_ssgsea) <- gsub("\\(.*?\\)", "", rownames(gobp_ssgsea))

gobp_ssgsea_top <- gobp_ssgsea %>% 
  rownames_to_column(var = "pathway") %>% 
  # filter(pathway %in% c("Extracellular Matrix Organization ", "External Encapsulating Structure Organization ", "Extracellular Structure Organization ", "Cell-Substrate Junction ",
  #                       "Cell-Matrix Adhesion ", "Regulation Of Vascular Associated Smooth Muscle Cell Proliferation ", "Cellular Component Disassembly (GO:0022411) ", "Negative Regulation Of Smooth Muscle Cell Proliferation ", 
  #                       "Endodermal Cell Differentiation ", "Extracellular Matrix Disassembly ", "Positive Regulation Of Locomotion ", "Collagen Fibril Organization ",
  #                       "Cell−Substrate Junction Assembly ", "Positive Regulation Of Smooth Muscle Cell Proliferation ", "Epithelial To Mesenchymal Transition ",
  #                       "Positive Regulation Of Vascular Associated Smooth Muscle Cell Proliferation ", "Regulation Of Chemotaxis ", "Response To Amyloid−Beta ",
  #                       "Negative Regulation Of Cytokine Production ", "Positive Regulation Of Fibroblast Proliferation ", "Antigen Processing And Presentation Of Exogenous Peptide Antigen ")) %>% 
  filter(pathway %in% c("Extracellular Matrix Organization ", "External Encapsulating Structure Organization ", "Extracellular Structure Organization ", "Cell-Matrix Adhesion ", 
                        "Regulation Of Vascular Associated Smooth Muscle Cell Proliferation ", "Cellular Component Disassembly ", "Negative Regulation Of Smooth Muscle Cell Proliferation ", 
                        "Endodermal Cell Differentiation ", "Extracellular Matrix Disassembly ", "Positive Regulation Of Smooth Muscle Cell Proliferation ")) %>% 
  column_to_rownames(var = "pathway")

gobp_stat_results <- compute_correlations_with_pvals(t(gobp_ssgsea_top), tcga_malignant, corr_method = "spearman")
gobp_stat_cor <- gobp_stat_results$correlations

colnames(gobp_stat_cor) <- colnames(tcga_malignant)
colnames(gobp_stat_cor) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")
rownames(gobp_stat_cor) <- rownames(gobp_ssgsea_top)

sig_matrix <- matrix("", nrow = nrow(gobp_stat_results$adjusted_pvals), ncol = ncol(gobp_stat_results$adjusted_pvals))
sig_matrix[gobp_stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[gobp_stat_results$adjusted_pvals < 0.01 & gobp_stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[gobp_stat_results$adjusted_pvals < 0.05 & gobp_stat_results$adjusted_pvals >= 0.01] <- "*"

p <- Heatmap(
  gobp_stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  show_row_names = TRUE,
  border = TRUE,
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

pdf(file="C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA_GOBP_top10_WSIcorr_stats_250424.pdf", width = 6, height = 4)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 20), "mm"))
dev.off()
