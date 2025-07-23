library(tidyverse)
library(scales)
library(ggrepel)
library(DESeq2)
library(fgsea)
library(pheatmap)
library(ComplexHeatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')

load("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/LUADChohort.RData")

hallmarks <- loadDB("h.all.v2024.1.Hs.symbols.gmt")
go_bp <- loadDB("GO_Biological_Process_2023.txt")
kegg <- loadDB("KEGG_2021_Human.txt")

cluster_labs <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/merged_cluster_lym_log10_density_and_noise.csv') %>% 
  select(c("HPC", "Supercluster"))

cluster_annotations <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/bioclavis_pure_superclusters_split_HPC13.csv') %>% 
  column_to_rownames("...1")

metadata <- LUADCohort$Metadata

reformat_core_id <- function(name) {
  parts <- unlist(strsplit(name, "-"))
  
  tma_num <- parts[1]
  row_num <- parts[2]
  col_num <- parts[3]
  
  if (nchar(tma_num) < 2) {
    new_tma_num <- paste0("0", tma_num)
  } else {
    new_tma_num <- tma_num
  }
  
  if (nchar(row_num) < 2) {
    new_row_num <- paste0("0", row_num)
  } else {
    new_row_num <- row_num
  }
  return(paste(new_tma_num, new_row_num, col_num, sep = "-"))
}
metadata$core_id <- sapply(metadata$Core, reformat_core_id)

metadata_subset = metadata[, c("...1", "core_id", "Sex")] %>%
  merge(., cluster_annotations, by.x=2, by.y=1) %>%
  column_to_rownames((var = '...1'))

metadata_subset <- metadata_subset %>% 
  mutate(supercluster = str_to_sentence(str_replace_all(supercluster, "_", " "))) %>% 
  mutate(hd_risk = recode(consensus_HD_more,
                          "1" = "Hot discohesive (high risk)", 
                          "0" = "Hot discohesive (low risk)"))

metadata_subset_split <- metadata_subset %>%
  mutate(supercluster = case_when(supercluster != "Hot discohesive" ~ supercluster,
                                  TRUE ~ hd_risk))

metadata_subset_split$supercluster <- as.factor(metadata_subset_split$supercluster)

###

hallmarks_ssgsea <- read_csv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/final_figures/v250331_superclusters/all_LATTICeA_ssGSEA_hallmarks_norm.csv") %>% 
  mutate(pathway = sub("^HALLMARK_", "", pathway)) %>% 
  column_to_rownames("pathway") %>% 
  select(rownames(metadata_subset_split))

metadata_subset_split <- metadata_subset_split[colnames(hallmarks_ssgsea), ]

ha <- HeatmapAnnotation(df = metadata_subset_split %>% select(supercluster),
                        col = list(supercluster = c(
                          "Hot cohesive"      = "#F77189",
                          "Hot discohesive (low risk)"   = "#36ADA4",
                          "Cold cohesive"     = "#97A431",
                          "Cold discohesive"  = "#A48CF5",
                          "Hot discohesive (high risk)" = "#FC8D62")))

ha <- HeatmapAnnotation(
  df  = metadata_subset_split %>% select(supercluster),
  col = list(
    supercluster = c(
      "Hot cohesive"                  = "#F77189",
      "Hot discohesive (low risk)"    = "#36ADA4",
      "Cold cohesive"                 = "#97A431",
      "Cold discohesive"              = "#A48CF5",
      "Hot discohesive (high risk)"   = "#FC8D62"
    )
  ),
  annotation_legend_param = list(
    supercluster = list(
      title = "Supercluster"          # <- new legend title
      # optional extras, e.g. title_position = "topleft"
    )
  )
)

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

p <- Heatmap(as.matrix(hallmarks_ssgsea) %>% pheatmap:::scale_rows(),
        show_column_names = FALSE,
        col = col_colors,
        top_annotation = ha,
        border = TRUE,
        name = "NES\nZ-score",
        cluster_columns = cluster_within_group(hallmarks_ssgsea %>% pheatmap:::scale_rows(), metadata_subset_split$supercluster))

pdf(file = "/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/final_figures/v250331_superclusters/LATTICeA_ssGSEA_hallmarks_20250425.pdf", 
    width = 15, 
    height = 8.5)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(10, 2, 2, 30), "mm"))
dev.off()

###

kegg_ssgsea <- read_csv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/KEGG_ssGSEA_LATTICeA.csv") %>% 
  column_to_rownames("...1") %>% 
  select(rownames(metadata_subset))

metadata_subset <- metadata_subset[colnames(kegg_ssgsea), ]

ha <- HeatmapAnnotation(df = metadata_subset %>% select(supercluster))

Heatmap(as.matrix(kegg_ssgsea) %>% pheatmap:::scale_rows(),
        show_column_names = FALSE,
        top_annotation = ha,
        cluster_columns = cluster_within_group(hallmarks_ssgsea %>% pheatmap:::scale_rows(), metadata_subset$supercluster))
