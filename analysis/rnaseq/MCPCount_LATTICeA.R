library(MCPcounter)
library(tidyverse)
library(circlize)
library(ComplexHeatmap)

all_norm_counts <- LUADCohort$normExpr
all_cell_estimates <- MCPcounter.estimate(all_norm_counts,
                                          featuresType = "HUGO_symbols")

Heatmap(all_cell_estimates %>% pheatmap:::scale_rows(),
        show_column_names = FALSE)

###

cluster_annotations <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/bioclavis_pure_superclusters_split_HPC13.csv') %>% 
  column_to_rownames("...1")

cluster_labs <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/merged_cluster_lym_log10_density_and_noise.csv') %>% 
  select(c("HPC", "Supercluster"))

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

metadata_subset <- metadata %>%
  select(c("...1", "core_id", "Sex")) %>% 
  inner_join(cluster_annotations, join_by(core_id == core_ID)) %>% 
  mutate(., supercluster = str_to_sentence(str_replace_all(supercluster, "_", " "))) %>% 
  column_to_rownames("...1")

cell_estimates_subset <- as.data.frame(all_cell_estimates) %>% 
  select(row.names(metadata_subset)) %>% 
  as.matrix(.)

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))
metadata_subset$supercluster <- as.factor(metadata_subset$supercluster)

column_ha <- HeatmapAnnotation(
  supercluster = metadata_subset$supercluster,
  col = list(supercluster = c("Hot cohesive" = "firebrick1",
                              "Hot discohesive" = "coral",
                              "Cold cohesive" = "dodgerblue",
                              "Cold discohesive" = "hotpink1"))
)

Heatmap(cell_estimates_subset %>% pheatmap:::scale_rows(),
        show_column_names = FALSE,
        top_annotation = column_ha,
        col = col_colors,
        cluster_columns = cluster_within_group(cell_estimates_subset, metadata_subset$supercluster))
