import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

def load_sequences():
    """Load gene and protein sequences from files"""
    gene_sequences = list(SeqIO.parse("data/gene.fna", "fasta"))
    protein_sequences = list(SeqIO.parse("data/protein.faa", "fasta"))
    return gene_sequences, protein_sequences

def analyze_protein_properties(protein_seq):
    """Analyze protein properties using BioPython"""
    protein = ProteinAnalysis(str(protein_seq))
    return {
        'molecular_weight': protein.molecular_weight(),
        'gravy': protein.gravy(),
        'isoelectric_point': protein.isoelectric_point(),
        'secondary_structure_fraction': protein.secondary_structure_fraction(),
        'amino_acid_percent': protein.get_amino_acids_percent()
    }

def plot_kid_friendly_analysis(protein_sequences):
    """Create kid-friendly visualizations for sequence analysis"""
    # Set a fun color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    # Set the style using seaborn's set_style instead of plt.style.use
    sns.set_style("whitegrid")
    sns.set_palette(colors)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Let\'s Learn About C21orf2 Gene! 🧬', fontsize=20, color='#FF6B6B')
    
    # Length distribution - like measuring how tall different versions are
    lengths = [len(str(seq.seq)) for seq in protein_sequences]
    sns.histplot(lengths, ax=axes[0,0], color=colors[0])
    axes[0,0].set_title('How Long Are Our Proteins? 📏', fontsize=14, color='#4A90E2')
    axes[0,0].set_xlabel('Length (like how many letters)', fontsize=12)
    axes[0,0].set_ylabel('How Many We Found', fontsize=12)
    
    # Amino acid composition - like counting different colored blocks
    aa_composition = pd.DataFrame([ProteinAnalysis(str(seq.seq)).get_amino_acids_percent() 
                                 for seq in protein_sequences])
    sns.boxplot(data=aa_composition, ax=axes[0,1], palette=colors)
    axes[0,1].set_title('Different Building Blocks 🧱', fontsize=14, color='#4A90E2')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # Hydrophobicity plot - like seeing which parts like water
    hydrophobicity = [sum(1 for aa in str(seq.seq) if aa in 'AILMFWV') / len(str(seq.seq)) 
                     for seq in protein_sequences]
    sns.histplot(hydrophobicity, ax=axes[1,0], color=colors[2])
    axes[1,0].set_title('Parts That Don\'t Like Water 💧', fontsize=14, color='#4A90E2')
    axes[1,0].set_xlabel('How Much They Don\'t Like Water', fontsize=12)
    axes[1,0].set_ylabel('How Many We Found', fontsize=12)
    
    # GC content - like counting special letters
    gc_content = [(str(seq.seq).count('G') + str(seq.seq).count('C')) / len(str(seq.seq)) * 100 
                 for seq in protein_sequences]
    sns.histplot(gc_content, ax=axes[1,1], color=colors[3])
    axes[1,1].set_title('Special Letters in Our Gene 🔤', fontsize=14, color='#4A90E2')
    axes[1,1].set_xlabel('How Many Special Letters', fontsize=12)
    axes[1,1].set_ylabel('How Many We Found', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def print_kid_friendly_conclusions():
    """Print conclusions in a child-friendly way"""
    print("\n" + "="*50)
    print("🌟 What We Learned About C21orf2! 🌟")
    print("="*50)
    print("\n1. 🧬 Our gene is like a special recipe book that tells our body how to make important things!")
    print("2. 🎨 It comes in different versions, just like how you can color a picture in different ways!")
    print("3. 💪 Some parts of our gene are very strong and don't like water, like a superhero's shield!")
    print("4. 🔍 We found lots of special letters (G and C) that help make our gene work properly!")
    print("5. 🎯 This gene might be important for helping our muscles work correctly!")
    print("\n" + "="*50)
    print("Remember: Just like how you're special and unique, this gene is special too!")
    print("It helps our body work properly, just like how you help make your family special! 🌈")
    print("="*50 + "\n")

def main():
    # Load sequences
    gene_sequences, protein_sequences = load_sequences()
    
    # Analyze protein properties
    protein_properties = [analyze_protein_properties(seq.seq) for seq in protein_sequences]
    
    # Create kid-friendly visualizations
    plot_kid_friendly_analysis(protein_sequences)
    
    # Print kid-friendly conclusions
    print_kid_friendly_conclusions()

if __name__ == "__main__":
    main() 