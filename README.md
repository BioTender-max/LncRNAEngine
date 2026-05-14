# LncRNAEngine

**Long Non-Coding RNA Regulatory Analysis Pipeline**

A pure-Python pipeline for lncRNA co-expression network analysis, RBP motif enrichment, and ceRNA network construction.

## Features
- lncRNA-mRNA co-expression network (Pearson correlation, BH FDR)
- RNA-binding protein (RBP) motif enrichment (50 RBPs)
- ceRNA network via shared miRNA targets (Jaccard similarity)
- Nuclear vs cytoplasmic localization prediction
- lncRNA conservation scoring (PhyloP)

## Results
- 100 samples × 2000 lncRNAs + 5000 mRNAs
- Significant co-expression pairs: 3 (FDR<0.05, |r|>0.5)
- Significant RBPs: 50/50
- Max ceRNA Jaccard score: 0.456
- Nuclear: 809 (40.5%), Cytoplasmic: 1191 (59.5%)
- Conservation t-stat: 23.69, p=1.39e-109

## Usage
```bash
pip install numpy scipy matplotlib
python lncrna_engine.py
```

## Tags
`lncrna` `long-noncoding-rna` `rna-binding-protein` `ceRNA` `nuclear-retention` `noncoding-rna`
