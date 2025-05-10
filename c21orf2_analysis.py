import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter, defaultdict
import warnings
from itertools import product
import re
warnings.filterwarnings('ignore')

def print_section_header(title, emoji="📊"):
    """Print a nicely formatted section header"""
    print("\n" + "="*80)
    print(f"{emoji} {title} {emoji}")
    print("="*80)

def print_subsection_header(title, emoji="📌"):
    """Print a nicely formatted subsection header"""
    print("\n" + "-"*70)
    print(f"{emoji} {title}")
    print("-"*70)

def load_sequences():
    """Load gene and protein sequences from files"""
    gene_sequences = list(SeqIO.parse("data/gene.fna", "fasta"))
    protein_sequences = list(SeqIO.parse("data/protein.faa", "fasta"))
    return gene_sequences, protein_sequences

def find_repeating_patterns(sequence, min_length=2, max_length=6):
    """Find repeating patterns in the sequence"""
    patterns = defaultdict(list)
    seq_str = str(sequence)
    
    # Find all possible patterns of given lengths
    for length in range(min_length, max_length + 1):
        for i in range(len(seq_str) - length + 1):
            pattern = seq_str[i:i+length]
            if len(pattern) == length:
                patterns[pattern].append(i)
    
    # Filter patterns that appear more than once
    repeats = {pattern: positions for pattern, positions in patterns.items() 
              if len(positions) > 1}
    
    return repeats

def find_palindromes(sequence, min_length=4, max_length=8):
    """Find palindromic sequences"""
    palindromes = []
    seq_str = str(sequence)
    
    for length in range(min_length, max_length + 1):
        for i in range(len(seq_str) - length + 1):
            substring = seq_str[i:i+length]
            if substring == substring[::-1]:
                palindromes.append((i, substring))
    
    return palindromes

def analyze_character_frequency(sequences):
    """Analyze character frequency in sequences"""
    print_section_header("Character Frequency Analysis", "🔤")
    
    # Get the first sequence for analysis
    seq = sequences[0]
    seq_str = str(seq.seq)
    
    # Count characters
    char_counts = Counter(seq_str)
    total_chars = len(seq_str)
    
    # Print character frequency table
    print("\nCharacter Distribution:")
    print("-" * 70)
    print(f"{'Character':<12} {'Count':<15} {'Percentage':<15} {'Frequency':<15}")
    print("-" * 70)
    
    # Sort by frequency (most common first)
    for char, count in char_counts.most_common():
        percentage = (count / total_chars) * 100
        frequency = f"1 in {total_chars/count:.1f}"
        print(f"{char:<12} {count:<15,} {percentage:<15.2f}% {frequency:<15}")
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie([count for _, count in char_counts.most_common()],
            labels=[char for char, _ in char_counts.most_common()],
            autopct='%1.1f%%',
            colors=sns.color_palette("pastel"),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title('Character Distribution in Gene Sequence', fontsize=16, pad=20)
    plt.show()
    
    return char_counts, total_chars

def analyze_sequence_patterns(sequences):
    """Analyze patterns in sequences"""
    print_section_header("Sequence Pattern Analysis", "🔍")
    
    # Analyze first sequence
    seq = sequences[0]
    seq_str = str(seq.seq)
    
    # Find repeating patterns
    repeats = find_repeating_patterns(seq)
    
    print_subsection_header("Repeating Patterns", "🔄")
    print("\nMost Common Repeating Patterns:")
    print("-" * 70)
    print(f"{'Pattern':<15} {'Occurrences':<15} {'Positions':<40}")
    print("-" * 70)
    
    # Sort patterns by number of occurrences
    sorted_patterns = sorted(repeats.items(), key=lambda x: len(x[1]), reverse=True)
    for pattern, positions in sorted_patterns[:10]:
        pos_str = ", ".join(str(p) for p in positions[:3])
        if len(positions) > 3:
            pos_str += f" ... (+{len(positions)-3} more)"
        print(f"{pattern:<15} {len(positions):<15} {pos_str:<40}")
    
    # Find palindromes
    palindromes = find_palindromes(seq)
    
    print_subsection_header("Palindromic Sequences", "🔄")
    print("\nFound Palindromic Sequences:")
    print("-" * 70)
    print(f"{'Position':<15} {'Sequence':<15} {'Length':<10}")
    print("-" * 70)
    
    for pos, palindrome in palindromes[:10]:
        print(f"{pos:<15} {palindrome:<15} {len(palindrome):<10}")
    
    return repeats, palindromes

def visualize_gene_structure(sequences):
    """Visualize the structure of the gene"""
    print_section_header("Gene Structure Visualization", "🧬")
    
    # Get the first sequence
    seq = sequences[0]
    seq_str = str(seq.seq)
    
    # Create a figure
    plt.figure(figsize=(15, 8))
    
    # Create a color map for different nucleotides
    color_map = {'A': '#FF9999', 'T': '#66B2FF', 'G': '#99FF99', 'C': '#FFCC99'}
    
    # Plot each nucleotide as a colored block
    for i, char in enumerate(seq_str):
        plt.bar(i, 1, color=color_map.get(char, '#CCCCCC'), width=1)
    
    # Add labels and title
    plt.title('Gene Structure Visualization', fontsize=16, pad=20)
    plt.xlabel('Position in Sequence', fontsize=12)
    plt.ylabel('Nucleotide', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=char) 
                      for char, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Show the plot
    plt.show()

def plot_pattern_analysis(sequences, repeats, palindromes):
    """Create visualizations for pattern analysis"""
    print_section_header("Pattern Analysis Visualizations", "📊")
    
    # Set a fun color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    # Set the style using seaborn's set_style
    sns.set_style("whitegrid")
    sns.set_palette(colors)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)
    
    # Main title
    fig.suptitle('Pattern Analysis of C21orf2 Gene! 🧬', fontsize=24, color='#FF6B6B', y=0.95)
    
    # 1. Pattern Length Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pattern_lengths = [len(pattern) for pattern in repeats.keys()]
    sns.histplot(pattern_lengths, ax=ax1, color=colors[0], bins=20)
    ax1.set_title('Distribution of Pattern Lengths 📏', fontsize=16, color='#4A90E2', pad=20)
    ax1.set_xlabel('Pattern Length', fontsize=14)
    ax1.set_ylabel('Number of Patterns', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. Pattern Frequency
    ax2 = fig.add_subplot(gs[0, 1])
    pattern_freq = [len(positions) for positions in repeats.values()]
    sns.histplot(pattern_freq, ax=ax2, color=colors[1], bins=20)
    ax2.set_title('Pattern Frequency Distribution 🔄', fontsize=16, color='#4A90E2', pad=20)
    ax2.set_xlabel('Number of Occurrences', fontsize=14)
    ax2.set_ylabel('Number of Patterns', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Palindrome Length Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    palindrome_lengths = [len(pal[1]) for pal in palindromes]
    sns.histplot(palindrome_lengths, ax=ax3, color=colors[2], bins=20)
    ax3.set_title('Palindrome Length Distribution 🔄', fontsize=16, color='#4A90E2', pad=20)
    ax3.set_xlabel('Palindrome Length', fontsize=14)
    ax3.set_ylabel('Number of Palindromes', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # 4. Pattern Position Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    # Create a binary matrix for pattern positions
    seq_length = len(str(sequences[0].seq))
    position_matrix = np.zeros((len(repeats), seq_length))
    for i, (pattern, positions) in enumerate(repeats.items()):
        for pos in positions:
            position_matrix[i, pos:pos+len(pattern)] = 1
    
    sns.heatmap(position_matrix, ax=ax4, cmap='YlOrRd', cbar_kws={'label': 'Pattern Presence'})
    ax4.set_title('Pattern Position Heatmap 🗺️', fontsize=16, color='#4A90E2', pad=20)
    ax4.set_xlabel('Sequence Position', fontsize=14)
    ax4.set_ylabel('Pattern Index', fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid lines for better readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def print_kid_friendly_conclusions():
    """Print conclusions in a child-friendly way"""
    print_section_header("What We Learned About C21orf2!", "🌟")
    
    print("\n1. 🧬 Gene Structure:")
    print("   • Our gene is like a special recipe book")
    print("   • It tells our body how to make important things")
    print("   • It has different versions, like different ways to color a picture")
    
    print("\n2. 💪 Special Features:")
    print("   • Some parts are very strong and don't like water")
    print("   • It has lots of special letters (G and C)")
    print("   • These letters help make our gene work properly")
    
    print("\n3. 🎯 Important Job:")
    print("   • This gene helps our muscles work correctly")
    print("   • It's like a tiny worker in our body")
    print("   • It's special and unique, just like you!")
    
    print("\n" + "="*60)
    print("Remember: This gene helps our body work properly,")
    print("just like how you help make your family special! 🌈")
    print("="*60 + "\n")

def main():
    # Load sequences
    gene_sequences, protein_sequences = load_sequences()
    
    # 1. Character Frequency Analysis
    char_counts, total_chars = analyze_character_frequency(gene_sequences)
    
    # 2. Sequence Pattern Analysis
    repeats, palindromes = analyze_sequence_patterns(gene_sequences)
    
    # 3. Gene Structure Visualization
    visualize_gene_structure(gene_sequences)
    
    # 4. Pattern Analysis Visualizations
    plot_pattern_analysis(gene_sequences, repeats, palindromes)
    
    # 5. Print conclusions
    print_kid_friendly_conclusions()

if __name__ == "__main__":
    main()